// Fast, byte-identical parquet writer for trace.parquet / message_trace.parquet, self-contained
// (no Arrow / libparquet at run time: loading and initialising those libraries cost ~10-15 ms per
// process, a visible share of a small unit's container window).
//
// It reproduces parquet-cpp 15.0.2 (cpp/src/parquet/column_writer.cc, encoding.cc, statistics.cc,
// metadata.cc, file_writer.cc at apache-arrow-15.0.2) driven by parquet::arrow::WriteTable with
// pyarrow.parquet.write_table's defaults, for the cases these files contain -- nullable INT64 /
// INT32 / UTF8 columns, one level, V1 data pages, dictionary encoding with fallback to PLAIN,
// SNAPPY, statistics on, no page index, no checksums. Each step names the function it reproduces:
//
//  - DoInBatches / WriteBatch: values go in batches of write_batch_size (1024). After each batch
//    the data page is cut if the encoder's estimated size reaches data_pagesize (1 MiB)
//    (CommitWriteAndCheckPageLimit), then the dictionary is abandoned for PLAIN if its encoded
//    size reaches dictionary_pagesize_limit (1 MiB) (CheckDictionarySizeLimit).
//  - DictEncoderImpl: indices in first-occurrence order; data page = bit width byte + RLE/bit-packed
//    hybrid (RleFast: arrow::util::RleEncoder's exact run decisions, inline); estimated size =
//    1 + MaxBufferSize(bit_width, n) + MinBufferSize(bit_width).
//  - Definition levels: RLE with a 4-byte length prefix (LevelEncoder = RleEncoder, bit width 1).
//  - Statistics: per page min / max / null_count (TypedStatisticsImpl::Update / Encode), merged into
//    the chunk statistics when a page is cut (ResetPageStatistics); ToThrift sets the deprecated
//    min/max too for signed (integer) columns.
//  - Pages are buffered while the dictionary is in use and written after the dictionary page
//    (Close / FallbackToPlainEncoding); after a fallback they are written eagerly.
//  - SerializedPageWriter: data pages and the dictionary page compressed with snappy 1.1.10's
//    RawCompress (the codec bundled in the pyarrow wheel, vendored in third_party/snappy and built
//    with its configuration); thrift compact PageHeader before each page; the ColumnChunk
//    metadata (ColumnChunkMetaDataBuilder::Finish) written after each chunk and again in the
//    footer; RowGroupMetaDataBuilder::Finish; FileMetaData with the schema, key-value metadata
//    (pandas + ARROW:schema), created_by and column orders, which do not depend on the data and
//    come from pq_footer_consts.h (tools/gen_footer_consts.py).
// Column chunks are independent until they reach the file, so each is encoded and compressed on a
// worker thread; the pages are then laid out in schema order on one thread.
// tests/test_pqfast.cpp compares this writer with the libparquet-backed one byte for byte, and
// scripts/check_identical.py compares every output against the baseline.
#include "pqfast.h"

#include "pq_footer_consts.h"
#include "rle_fast.h"
#include "sha256.h"
#include "third_party/snappy/snappy.h"

#include <fcntl.h>
#include <limits.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

namespace fastsim {

namespace {

constexpr int64_t kBatch = 1024;                 // WriterProperties::write_batch_size
constexpr int64_t kDataPageSize = 1024 * 1024;   // data_pagesize
constexpr int64_t kDictPageLimit = 1024 * 1024;  // dictionary_pagesize_limit
constexpr size_t kMaxStatSize = 4096;            // max_statistics_size
constexpr int64_t kRowGroup = 1024 * 1024;       // write_table(row_group_size=None) chunk

// parquet.thrift enum values
enum : int32_t { T_INT32 = 1, T_INT64 = 2, T_BYTE_ARRAY = 6 };                    // Type
enum : int32_t { E_PLAIN = 0, E_RLE = 3, E_RLE_DICTIONARY = 8 };                  // Encoding
enum : int32_t { P_DATA_PAGE = 0, P_DICTIONARY_PAGE = 2 };                        // PageType
constexpr int32_t kCodecSnappy = 1;                                               // CompressionCodec
// PARQUET_2_6: dictionary_index_encoding() = RLE_DICTIONARY, dictionary_page_encoding() = PLAIN
constexpr int32_t kDictIndexEncoding = E_RLE_DICTIONARY;
constexpr int32_t kDictPageEncoding = E_PLAIN;

int num_required_bits(uint64_t x) { return x == 0 ? 0 : 64 - __builtin_clzll(x); }

int bit_width_for(int num_entries) {  // DictEncoderImpl::bit_width
  if (num_entries == 0) return 0;
  if (num_entries == 1) return 1;
  return num_required_bits((uint64_t)num_entries - 1);  // bit_util::Log2
}

int64_t bytes_for_bits(int64_t bits) { return (bits + 7) >> 3; }

int rle_min_buffer_size(int bw) {  // RleEncoder::MinBufferSize
  const int max_literal_run_size = 1 + (int)bytes_for_bits(512 * bw);
  const int max_repeated_run_size = 5 + (int)bytes_for_bits(bw);
  return std::max(max_literal_run_size, max_repeated_run_size);
}

int rle_max_buffer_size(int bw, int num_values) {  // RleEncoder::MaxBufferSize
  const int num_runs = (num_values + 7) / 8;
  const int literal_max_size = num_runs + num_runs * bw;
  const int repeated_max_size = num_runs * (1 + (int)bytes_for_bits(bw));
  return std::max(literal_max_size, repeated_max_size);
}

// Thrift compact protocol (TCompactProtocol, thrift 0.16), only what the parquet structs need.
class ThriftWriter {
 public:
  explicit ThriftWriter(ByteBuf& b) : b_(b) {}
  void begin_struct() { stack_[depth_++] = last_; last_ = 0; }
  void end_struct() { b_.push(0); last_ = stack_[--depth_]; }
  void field(int16_t id, uint8_t type) {
    const int d = id - last_;
    if (d > 0 && d <= 15) {
      b_.push((uint8_t)(d << 4 | type));
    } else {
      b_.push(type);
      varint(zigzag32(id));
    }
    last_ = id;
  }
  void i16(int16_t id, int16_t v) { field(id, 4); varint(zigzag32(v)); }
  void i32(int16_t id, int32_t v) { field(id, 5); varint(zigzag32(v)); }
  void i64(int16_t id, int64_t v) { field(id, 6); varint(zigzag64(v)); }
  void boolean(int16_t id, bool v) { field(id, v ? 1 : 2); }
  void binary(int16_t id, const void* p, size_t n) { field(id, 8); bytes(p, n); }
  void binary(int16_t id, const std::string& s) { binary(id, s.data(), s.size()); }
  void struct_field(int16_t id) { field(id, 12); begin_struct(); }
  void list(int16_t id, uint8_t elem_type, size_t n) {
    field(id, 9);
    if (n < 15) {
      b_.push((uint8_t)(n << 4 | elem_type));
    } else {
      b_.push((uint8_t)(0xF0 | elem_type));
      varint(n);
    }
  }
  void elem_i32(int32_t v) { varint(zigzag32(v)); }
  void elem_string(const char* s) { bytes(s, std::strlen(s)); }
  void raw(const void* p, size_t n) { b_.append(p, n); }
  void varint(uint64_t v) {
    while (v >= 0x80) {
      b_.push((uint8_t)(v | 0x80));
      v >>= 7;
    }
    b_.push((uint8_t)v);
  }
  static uint64_t zigzag64(int64_t v) { return ((uint64_t)v << 1) ^ (uint64_t)(v >> 63); }

 private:
  static uint32_t zigzag32(int32_t v) { return ((uint32_t)v << 1) ^ (uint32_t)(v >> 31); }
  void bytes(const void* p, size_t n) { varint(n); b_.append(p, n); }
  ByteBuf& b_;
  int16_t last_ = 0;
  int16_t stack_[16];
  int depth_ = 0;
};

// EncodedStatistics as ToThrift sees it.
struct EncStats {
  bool has_min = false, has_max = false;
  std::string min, max;
  int64_t null_count = 0;
  bool is_signed = true;
};

void write_statistics(ThriftWriter& w, int16_t id, const EncStats& s) {  // ToThrift(EncodedStatistics)
  w.struct_field(id);
  if (s.has_max && s.is_signed) w.binary(1, s.max);
  if (s.has_min && s.is_signed) w.binary(2, s.min);
  w.i64(3, s.null_count);
  if (s.has_max) w.binary(5, s.max);
  if (s.has_min) w.binary(6, s.min);
  w.end_struct();
}

// Open-addressing map value -> first-occurrence index (the memo table's only observable property).
template <typename T>
class DictMap {
 public:
  void reset() {
    shift_ = 64 - 10;
    slots_.clear();
    values_.clear();
    values_.reserve(4096);
    monotone_ = true;
  }
  // returns {index, inserted}. While the values seen so far are strictly increasing no table is
  // kept: a value above the maximum is new and one equal to it is the last entry. The table is
  // built the first time a value goes below the maximum.
  __attribute__((always_inline)) inline std::pair<int32_t, bool> get_or_insert(T v) {
    if (monotone_) {
      if (values_.empty() || v > values_.back()) {
        values_.push_back(v);
        return {(int32_t)values_.size() - 1, true};
      }
      if (v == values_.back()) return {(int32_t)values_.size() - 1, false};
      build_table();
    }
    return lookup(v);
  }
  inline std::pair<int32_t, bool> lookup(T v) {
    const size_t mask = slots_.size() - 1;
    size_t h = hash(v);
    while (true) {
      Slot& s = slots_[h];
      if (s.idx < 0) break;
      if (s.key == v) return {s.idx, false};
      h = (h + 1) & mask;
    }
    const int32_t idx = (int32_t)values_.size();
    values_.push_back(v);
    slots_[h] = Slot{v, idx};
    if (values_.size() * 2 > slots_.size()) grow();
    return {idx, true};
  }
  const std::vector<T>& values() const { return values_; }
  int size() const { return (int)values_.size(); }

 private:
  struct Slot {
    T key;
    int32_t idx;
  };
  void build_table() {
    monotone_ = false;
    size_t cap = (size_t)1 << 10;
    shift_ = 64 - 10;
    while (values_.size() * 2 > cap) { cap *= 2; shift_--; }
    slots_.assign(cap, Slot{0, -1});
    const size_t mask = cap - 1;
    for (int32_t i = 0; i < (int32_t)values_.size(); i++) {
      size_t h = hash(values_[i]);
      while (slots_[h].idx >= 0) h = (h + 1) & mask;
      slots_[h] = Slot{values_[i], i};
    }
  }
  bool monotone_ = true;
  inline size_t hash(T v) const { return (size_t)(((uint64_t)(int64_t)v * 0x9E3779B97F4A7C15ULL) >> shift_); }
  void grow() {
    shift_--;
    std::vector<Slot> old;
    old.swap(slots_);
    slots_.assign(old.size() * 2, Slot{0, -1});
    const size_t mask = slots_.size() - 1;
    for (const Slot& s : old) {
      if (s.idx < 0) continue;
      size_t h = hash(s.key);
      while (slots_[h].idx >= 0) h = (h + 1) & mask;
      slots_[h] = s;
    }
  }
  int shift_ = 54;
  std::vector<Slot> slots_;
  std::vector<T> values_;
};

// Statistics of one page or of the chunk (TypedStatisticsImpl, with the values' order defined by
// `less`): min/max over non-null values, null count, number of non-null values.
template <typename T>
struct Stats {
  bool has_minmax = false;
  T min{}, max{};
  int64_t null_count = 0, num_values = 0;
  void reset() { *this = Stats(); }
};

// The file being laid out: page headers, metadata and footer go into `meta`; the file is the
// sequence of segments, each a range of `meta` (p == nullptr) or a page body held by an encoder.
struct Layout {
  struct Seg {
    const uint8_t* p;
    size_t off, n;
  };
  ByteBuf meta;
  std::vector<Seg> segs;
  int64_t pos = 0;  // file size so far
  void meta_since(size_t from) {  // the bytes appended to meta since `from`
    const size_t n = meta.size() - from;
    if (!segs.empty() && segs.back().p == nullptr && segs.back().off + segs.back().n == from) segs.back().n += n;
    else segs.push_back(Seg{nullptr, from, n});
    pos += (int64_t)n;
  }
  void body(const uint8_t* p, size_t n) {
    if (n == 0) return;
    segs.push_back(Seg{p, 0, n});
    pos += (int64_t)n;
  }
};

// A finished page, in file order: the compressed dictionary page or a compressed V1 data page.
struct PageRec {
  bool dict;
  ByteBuf data;  // compressed
  int32_t num_values;
  int32_t encoding;
  int64_t uncompressed_size;
  EncStats stats;  // data pages
};

// SerializedPageWriter::Compress (snappy::RawCompress, as arrow's SnappyCodec calls it)
ByteBuf compress(const uint8_t* data, size_t size) {
  ByteBuf b;
  b.resize(snappy::MaxCompressedLength(size));
  size_t len = 0;
  snappy::RawCompress((const char*)data, size, (char*)b.data(), &len);
  b.resize(len);
  return b;
}

// Encodes one column chunk into PageRecs. Independent of every other chunk, so chunks are encoded
// in parallel; ColumnEncoder::write then lays the chunk out in the file.
class ColumnEncoder {
 public:
  ColumnEncoder(const PqColumn& col, const char* name) : col_(col), name_(name) {
    stride_ = col_.stride ? col_.stride
              : col_.kind == PqColumn::I64 ? 8 : col_.kind == PqColumn::I32 ? 4 : (size_t)col_.code_bytes;
    is_signed_ = col_.kind != PqColumn::STR;  // INT32/INT64: SIGNED; UTF8 BYTE_ARRAY: UNSIGNED
    encoding_ = kDictIndexEncoding;
    i64_dict_.reset();
    i32_dict_.reset();
    if (col_.kind == PqColumn::STR) {
      str_index_.assign(col_.n_names, -1);
      // order of the names as BYTE_ARRAY statistics compare them (unsigned lexicographic)
      str_rank_.resize(col_.n_names);
      std::vector<int> order(col_.n_names);
      for (int k = 0; k < col_.n_names; k++) order[k] = k;
      std::sort(order.begin(), order.end(), [&](int a, int b) { return std::strcmp(col_.names[a], col_.names[b]) < 0; });
      for (int r = 0; r < col_.n_names; r++) str_rank_[order[r]] = r;
    }
  }

  // Encodes rows [begin, end) of the column and closes the chunk.
  void encode(int64_t begin, int64_t end) {
    const int64_t total = end - begin;
    const int64_t full = total / kBatch;
    for (int64_t b = 0; b < full; b++) write_batch(begin + b * kBatch, kBatch);
    if (total % kBatch) write_batch(begin + full * kBatch, total % kBatch);
    close();
  }

  // Lays the chunk out at the end of the file as SerializedPageWriter would (pages, then the
  // ColumnChunk metadata), records the metadata for the footer, and returns the total bytes
  // written (ColumnWriterImpl::Close's return value).
  int64_t write(Layout& L) {
    int64_t total = 0, num_values = 0, dict_offset = 0, data_offset = 0, comp = 0, uncomp = 0;
    int dict_pages = 0, plain_pages = 0, dict_index_pages = 0;
    bool first_data = true;
    ByteBuf& out = L.meta;
    for (const PageRec& r : recs_) {
      const int64_t start = L.pos;
      const size_t h0 = out.size();
      ThriftWriter w(out);
      w.begin_struct();  // PageHeader
      w.i32(1, r.dict ? P_DICTIONARY_PAGE : P_DATA_PAGE);
      w.i32(2, (int32_t)r.uncompressed_size);
      w.i32(3, (int32_t)r.data.size());
      if (r.dict) {
        w.struct_field(7);  // DictionaryPageHeader
        w.i32(1, r.num_values);
        w.i32(2, r.encoding);
        w.boolean(3, false);  // is_sorted
        w.end_struct();
      } else {
        w.struct_field(5);  // DataPageHeader
        w.i32(1, r.num_values);
        w.i32(2, r.encoding);
        w.i32(3, E_RLE);
        w.i32(4, E_RLE);
        write_statistics(w, 5, r.stats);
        w.end_struct();
      }
      w.end_struct();
      const int64_t header = (int64_t)(out.size() - h0);
      L.meta_since(h0);
      L.body(r.data.data(), r.data.size());
      if (r.dict) {
        if (dict_offset == 0) dict_offset = start;
        dict_pages++;
      } else {
        if (first_data) data_offset = start;
        first_data = false;
        num_values += r.num_values;
        (r.encoding == E_PLAIN ? plain_pages : dict_index_pages)++;
      }
      uncomp += r.uncompressed_size + header;
      comp += (int64_t)r.data.size() + header;
      total += r.uncompressed_size + header;
    }
    // ColumnChunkMetaDataBuilder::Finish (has_dictionary = true), then WriteTo
    const size_t meta_start = out.size();
    {
      ThriftWriter w(out);
      w.begin_struct();  // ColumnChunk
      w.i64(2, dict_offset > 0 ? dict_offset + comp : data_offset + comp);  // file_offset
      w.struct_field(3);                                                  // ColumnMetaData
      w.i32(1, col_.kind == PqColumn::I64 ? T_INT64 : col_.kind == PqColumn::I32 ? T_INT32 : T_BYTE_ARRAY);
      // encodings: dictionary page encodings, RLE (levels), data page encodings (map order)
      int32_t encs[4];
      int ne = 0;
      auto add = [&](int32_t e) {
        for (int k = 0; k < ne; k++)
          if (encs[k] == e) return;
        encs[ne++] = e;
      };
      if (dict_pages) add(kDictPageEncoding);
      add(E_RLE);
      if (plain_pages) add(E_PLAIN);
      if (dict_index_pages) add(kDictIndexEncoding);
      w.list(2, 5, (size_t)ne);
      for (int k = 0; k < ne; k++) w.elem_i32(encs[k]);
      w.list(3, 8, 1);
      w.elem_string(name_);
      w.i32(4, kCodecSnappy);
      w.i64(5, num_values);
      w.i64(6, uncomp);
      w.i64(7, comp);
      w.i64(9, data_offset);
      if (dict_offset > 0) w.i64(11, dict_offset);
      if (rows_written_ > 0) write_statistics(w, 12, chunk_encoded_);
      const size_t nstats = (dict_pages ? 1 : 0) + (plain_pages ? 1 : 0) + (dict_index_pages ? 1 : 0);
      w.list(13, 12, nstats);
      auto stat = [&](int32_t type, int32_t enc, int32_t count) {  // PageEncodingStats
        w.begin_struct();
        w.i32(1, type);
        w.i32(2, enc);
        w.i32(3, count);
        w.end_struct();
      };
      if (dict_pages) stat(P_DICTIONARY_PAGE, kDictPageEncoding, dict_pages);
      if (plain_pages) stat(P_DATA_PAGE, E_PLAIN, plain_pages);
      if (dict_index_pages) stat(P_DATA_PAGE, kDictIndexEncoding, dict_index_pages);
      w.end_struct();
      w.end_struct();
    }
    meta_.assign((const char*)out.data() + meta_start, out.size() - meta_start);
    L.meta_since(meta_start);
    first_page_offset_ = dict_offset > 0 ? dict_offset : data_offset;
    total_compressed_ = comp;
    return total;
  }

  // Streaming use: point the column at records that moved (`rel` holds offsets from base).
  void rebase(const PqColumn& rel, const uint8_t* base) {
    auto at = [base](const void* p) -> const void* { return p || base ? (const void*)(base + (uintptr_t)p) : nullptr; };
    col_.data = rel.derive == PqColumn::NONE ? at(rel.data) : nullptr;
    col_.valid = rel.valid ? (const uint8_t*)at(rel.valid) : nullptr;
    col_.a = rel.a ? at(rel.a) : nullptr;
    col_.b = rel.b ? at(rel.b) : nullptr;
    col_.cond = rel.cond ? (const uint8_t*)at(rel.cond) : nullptr;
  }
  void feed(int64_t off, int64_t n) { write_batch(off, n); }  // rows [off, off + n), n <= kBatch
  void finish() { close(); }

  const std::string& meta() const { return meta_; }  // serialized ColumnChunk
  int64_t first_page_offset() const { return first_page_offset_; }
  int64_t total_compressed() const { return total_compressed_; }

 private:
  PqColumn col_;
  const char* name_;
  size_t stride_;
  std::vector<PageRec> recs_, pending_;  // written order; data pages held back while the dictionary is open
  EncStats chunk_encoded_;
  bool is_signed_ = true;
  bool fallback_ = false;
  int32_t encoding_;
  std::string meta_;
  int64_t first_page_offset_ = 0, total_compressed_ = 0;

  DictMap<int64_t> i64_dict_;
  DictMap<int32_t> i32_dict_;
  std::vector<int32_t> str_index_, str_rank_, str_dict_;  // code -> dict index; code -> rank; dict index -> code
  int dict_encoded_size_ = 0;
  std::vector<int32_t> indices_;
  std::vector<uint8_t> plain_;
  std::vector<uint8_t> def_levels_;  // only for columns with a validity vector
  int64_t page_nulls_ = 0;
  std::vector<uint8_t> page_;          // uncompressed page image

  int64_t num_buffered_values_ = 0, num_buffered_rows_ = 0, rows_written_ = 0;
  Stats<int64_t> page_stats_, chunk_stats_;  // ints as int64; strings as rank

  int dict_entries() const {
    switch (col_.kind) {
      case PqColumn::I64: return i64_dict_.size();
      case PqColumn::I32: return i32_dict_.size();
      default: return (int)str_dict_.size();
    }
  }

  int64_t estimated_size() const {  // EstimatedDataEncodedSize of the current encoder
    if (!fallback_) {
      const int bw = bit_width_for(dict_entries());
      return 1 + rle_max_buffer_size(bw, (int)indices_.size()) + rle_min_buffer_size(bw);
    }
    return (int64_t)plain_.size();
  }
  void put_code(int32_t c) {  // one string value (by code)
    if (fallback_) {
      const uint32_t len = (uint32_t)std::strlen(col_.names[c]);
      const size_t o = plain_.size();
      plain_.resize(o + 4 + len);
      std::memcpy(plain_.data() + o, &len, 4);
      std::memcpy(plain_.data() + o + 4, col_.names[c], len);
    } else {
      if (str_index_[c] < 0) {
        str_index_[c] = (int32_t)str_dict_.size();
        str_dict_.push_back(c);
        dict_encoded_size_ += (int)std::strlen(col_.names[c]) + 4;
      }
      indices_.push_back(str_index_[c]);
    }
    update_stats(str_rank_[c]);
  }

  void update_stats(int64_t v) {
    if (!page_stats_.has_minmax) {
      page_stats_.has_minmax = true;
      page_stats_.min = page_stats_.max = v;
    } else {
      if (v < page_stats_.min) page_stats_.min = v;
      if (v > page_stats_.max) page_stats_.max = v;
    }
  }

  // One batch of an integer column (value k at p + k * stride): dictionary indices (or PLAIN bytes
  // after a fallback) and the page statistics, in a loop specialised on type and nullability.
  template <typename T, bool kNullable>
  int64_t put_ints(const uint8_t* p, size_t stride, const uint8_t* valid, size_t vstride, int64_t n, DictMap<T>& dict) {
    int64_t nv = 0;
    bool any = page_stats_.has_minmax;
    int64_t mn = page_stats_.min, mx = page_stats_.max;
    if (!fallback_) {
      const size_t base = indices_.size();
      indices_.resize(base + (size_t)n);
      int32_t* out = indices_.data() + base;
      T last{};
      int32_t last_idx = -1;
      for (int64_t i = 0; i < n; i++) {
        if (kNullable) {
          const uint8_t ok = valid[(size_t)i * vstride] != 0;
          def_levels_.push_back(ok);
          if (!ok) continue;
        }
        T v;
        std::memcpy(&v, p + (size_t)i * stride, sizeof(T));
        int32_t idx;
        if (v == last && last_idx >= 0) {
          idx = last_idx;
        } else {
          auto r = dict.get_or_insert(v);
          if (r.second) dict_encoded_size_ += (int)sizeof(T);
          idx = r.first;
          last = v;
          last_idx = idx;
        }
        out[nv++] = idx;
        if (!any) { any = true; mn = mx = v; }
        else { mn = v < mn ? v : mn; mx = v > mx ? v : mx; }
      }
      indices_.resize(base + (size_t)nv);
    } else {
      const size_t o = plain_.size();
      plain_.resize(o + (size_t)n * sizeof(T));
      uint8_t* w = plain_.data() + o;
      for (int64_t i = 0; i < n; i++) {
        if (kNullable) {
          const uint8_t ok = valid[(size_t)i * vstride] != 0;
          def_levels_.push_back(ok);
          if (!ok) continue;
        }
        T v;
        std::memcpy(&v, p + (size_t)i * stride, sizeof(T));
        std::memcpy(w + (size_t)nv * sizeof(T), &v, sizeof(T));
        nv++;
        if (!any) { any = true; mn = mx = v; }
        else { mn = v < mn ? v : mn; mx = v > mx ? v : mx; }
      }
      plain_.resize(o + (size_t)nv * sizeof(T));
    }
    page_stats_.has_minmax = any;
    page_stats_.min = mn;
    page_stats_.max = mx;
    return nv;
  }

  void write_batch(int64_t off, int64_t n) {
    const size_t vs = col_.valid_stride;
    const uint8_t* valid = col_.valid ? col_.valid + (size_t)off * vs : nullptr;
    const uint8_t* data = col_.data ? (const uint8_t*)col_.data + (size_t)off * stride_ : nullptr;
    int64_t nv = 0;
    switch (col_.kind) {
      case PqColumn::I64:
        if (col_.derive != PqColumn::NONE) {
          int64_t buf[kBatch];
          if (col_.derive == PqColumn::SEQ) {
            for (int64_t k = 0; k < n; k++) buf[k] = off + k;
          } else {
            const uint8_t* a = (const uint8_t*)col_.a + (size_t)off * stride_;
            const uint8_t* b = (const uint8_t*)col_.b + (size_t)off * stride_;
            const uint8_t* c = col_.cond + (size_t)off * stride_;
            for (int64_t k = 0; k < n; k++) {
              int64_t x, y;
              std::memcpy(&x, a + (size_t)k * stride_, 8);
              std::memcpy(&y, b + (size_t)k * stride_, 8);
              buf[k] = c[(size_t)k * stride_] ? x - y : 0;
            }
          }
          data = (const uint8_t*)buf;
          nv = valid ? put_ints<int64_t, true>(data, 8, valid, vs, n, i64_dict_)
                     : put_ints<int64_t, false>(data, 8, valid, vs, n, i64_dict_);
        } else {
          nv = valid ? put_ints<int64_t, true>(data, stride_, valid, vs, n, i64_dict_)
                     : put_ints<int64_t, false>(data, stride_, valid, vs, n, i64_dict_);
        }
        break;
      case PqColumn::I32:
        nv = valid ? put_ints<int32_t, true>(data, stride_, valid, vs, n, i32_dict_)
                   : put_ints<int32_t, false>(data, stride_, valid, vs, n, i32_dict_);
        break;
      default:
        for (int64_t k = 0; k < n; k++) {
          if (valid) {
            const uint8_t ok = valid[(size_t)k * vs] != 0;
            def_levels_.push_back(ok);
            if (!ok) continue;
          }
          int32_t c;
          if (col_.code_bytes == 1) c = (int8_t)data[(size_t)k * stride_];
          else std::memcpy(&c, data + (size_t)k * stride_, 4);
          put_code(c);
          nv++;
        }
    }
    page_nulls_ += n - nv;
    rows_written_ += n;
    num_buffered_rows_ += n;
    page_stats_.null_count += n - nv;
    page_stats_.num_values += nv;
    num_buffered_values_ += n;
    if (estimated_size() >= kDataPageSize) add_data_page();
    if (!fallback_ && dict_encoded_size_ >= kDictPageLimit) fall_back();
  }

  std::string encode_stat(int64_t v) const {
    if (col_.kind == PqColumn::I64) return std::string((const char*)&v, 8);
    if (col_.kind == PqColumn::I32) {
      const int32_t x = (int32_t)v;
      return std::string((const char*)&x, 4);
    }
    // rank -> the name
    for (int c = 0; c < col_.n_names; c++)
      if (str_rank_[c] == v) return std::string(col_.names[c]);
    return std::string();
  }

  EncStats encode(const Stats<int64_t>& s) const {  // TypedStatisticsImpl::Encode
    EncStats e;
    if (s.has_minmax) {
      e.has_min = e.has_max = true;
      e.min = encode_stat(s.min);
      e.max = encode_stat(s.max);
    }
    e.null_count = s.null_count;
    // ApplyStatSizeLimits(max_statistics_size)
    if (e.max.size() > kMaxStatSize) e.has_max = false;
    if (e.min.size() > kMaxStatSize) e.has_min = false;
    e.is_signed = is_signed_;
    return e;
  }

  void add_data_page() {  // ColumnWriterImpl::AddDataPage + BuildDataPageV1
    const int nbv = (int)num_buffered_values_;
    page_.clear();
    page_.reserve((size_t)(32 + (int64_t)def_levels_.size() / 4 + rle_max_buffer_size(1, nbv) +
                           (fallback_ ? (int64_t)plain_.size() : estimated_size())));
    page_.resize(4);
    // definition levels, RLE with a 4-byte length prefix (RleEncodeLevels)
    if (page_nulls_ == 0) {  // nbv ones: one repeated run, VLQ(2 * nbv) then the value byte 1
      uint32_t v = (uint32_t)nbv << 1;
      while (v & 0xFFFFFF80u) {
        page_.push_back((uint8_t)((v & 0x7F) | 0x80));
        v >>= 7;
      }
      page_.push_back((uint8_t)v);
      page_.push_back(1);
    } else {
      RleFast d(page_, 1);
      for (uint8_t x : def_levels_) d.put(x);
      d.flush();
    }
    const int32_t def_len = (int32_t)page_.size() - 4;
    std::memcpy(page_.data(), &def_len, 4);
    // values (GetValuesBuffer -> FlushValues)
    if (!fallback_) {
      const int bw = bit_width_for(dict_entries());
      page_.push_back((uint8_t)bw);
      RleFast enc(page_, bw);
      for (int32_t idx : indices_) enc.put((uint64_t)idx);
      enc.flush();
      indices_.clear();
    } else {
      page_.insert(page_.end(), plain_.begin(), plain_.end());
      plain_.clear();
    }
    const int64_t uncompressed_size = (int64_t)page_.size();
    page_nulls_ = 0;

    EncStats page_stats = encode(page_stats_);
    // ResetPageStatistics: merge into the chunk statistics, then reset
    chunk_stats_.num_values += page_stats_.num_values;
    chunk_stats_.null_count += page_stats_.null_count;
    if (page_stats_.has_minmax) {
      if (!chunk_stats_.has_minmax) {
        chunk_stats_.has_minmax = true;
        chunk_stats_.min = page_stats_.min;
        chunk_stats_.max = page_stats_.max;
      } else {
        chunk_stats_.min = std::min(chunk_stats_.min, page_stats_.min);
        chunk_stats_.max = std::max(chunk_stats_.max, page_stats_.max);
      }
    }
    page_stats_.reset();

    PageRec rec{false, compress(page_.data(), (size_t)uncompressed_size), nbv, encoding_, uncompressed_size,
                std::move(page_stats)};
    // dictionary mode: keep until the dictionary page is written; after a fallback: in order
    (fallback_ ? recs_ : pending_).push_back(std::move(rec));
    def_levels_.clear();
    num_buffered_values_ = 0;
    num_buffered_rows_ = 0;
  }

  void write_dictionary_page() {  // WriteDictionaryPage, compressed here rather than in the pager
    page_.resize((size_t)dict_encoded_size_ + 1);
    uint8_t* p = page_.data();
    int entries;
    switch (col_.kind) {
      case PqColumn::I64:
        std::memcpy(p, i64_dict_.values().data(), i64_dict_.values().size() * 8);
        entries = i64_dict_.size();
        break;
      case PqColumn::I32:
        std::memcpy(p, i32_dict_.values().data(), i32_dict_.values().size() * 4);
        entries = i32_dict_.size();
        break;
      default:
        for (int32_t c : str_dict_) {
          const uint32_t len = (uint32_t)std::strlen(col_.names[c]);
          std::memcpy(p, &len, 4);
          std::memcpy(p + 4, col_.names[c], len);
          p += 4 + len;
        }
        entries = (int)str_dict_.size();
    }
    recs_.push_back(PageRec{true, compress(page_.data(), (size_t)dict_encoded_size_), entries, kDictPageEncoding,
                            dict_encoded_size_, EncStats()});
  }

  void flush_buffered_pages() {  // FlushBufferedDataPages
    if (num_buffered_values_ > 0) add_data_page();
    for (PageRec& r : pending_) recs_.push_back(std::move(r));
    pending_.clear();
  }

  void fall_back() {  // FallbackToPlainEncoding
    write_dictionary_page();
    flush_buffered_pages();
    fallback_ = true;
    encoding_ = E_PLAIN;
  }

  void close() {  // ColumnWriterImpl::Close
    if (!fallback_) write_dictionary_page();
    flush_buffered_pages();
    chunk_encoded_ = encode(chunk_stats_);
  }
};

const char* const kTraceNames[] = {"t_ns", "agent_id", "msg_type", "side", "price", "size", "order_id"};
const PqColumn::Kind kTraceKinds[] = {PqColumn::I64, PqColumn::I32, PqColumn::STR, PqColumn::STR,
                                      PqColumn::I64, PqColumn::I64, PqColumn::I64};
const char* const kLedgerNames[] = {"seq",    "t_recv_ns",  "t_send_ns", "latency_ns", "src_id",
                                    "dst_id", "message_id", "msg_type",  "order_id",   "causal_parent"};
const PqColumn::Kind kLedgerKinds[] = {PqColumn::I64, PqColumn::I64, PqColumn::I64, PqColumn::I64, PqColumn::I32,
                                       PqColumn::I32, PqColumn::I64, PqColumn::STR, PqColumn::I64, PqColumn::I64};

}  // namespace

struct PqImage::Impl {
  std::vector<std::unique_ptr<ColumnEncoder>> enc;  // own the page bodies
  Layout L;
  const uint8_t* seg_ptr(const Layout::Seg& g) const { return g.p ? g.p : L.meta.data() + g.off; }
};

PqImage::PqImage() : impl(new Impl) {}
PqImage::~PqImage() { delete impl; }
size_t PqImage::size() const { return (size_t)impl->L.pos; }

void PqImage::copy_to(ByteBuf& out) const {
  out.clear();
  out.reserve(size());
  for (const auto& g : impl->L.segs) out.append(impl->seg_ptr(g), g.n);
}

bool PqImage::write_file(const std::string& path, std::string* sha, std::string& err) const {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    err = "cannot open " + path;
    return false;
  }
  Sha256 h;
  const auto& segs = impl->L.segs;
  bool ok = true;
  std::vector<iovec> iov;
  for (size_t s0 = 0; s0 < segs.size() && ok;) {
    const size_t s1 = std::min(segs.size(), s0 + 512);
    iov.clear();
    for (size_t k = s0; k < s1; k++) {
      const uint8_t* p = impl->seg_ptr(segs[k]);
      if (sha) h.update(p, segs[k].n);
      iov.push_back(iovec{(void*)p, segs[k].n});
    }
    size_t i = 0;
    while (i < iov.size()) {  // writev may write less than asked
      const ssize_t w = ::writev(fd, iov.data() + i, (int)std::min<size_t>(iov.size() - i, IOV_MAX));
      if (w < 0) {
        if (errno == EINTR) continue;
        ok = false;
        break;
      }
      size_t left = (size_t)w;
      while (i < iov.size() && left >= iov[i].iov_len) left -= iov[i++].iov_len;
      if (left) {
        iov[i].iov_base = (uint8_t*)iov[i].iov_base + left;
        iov[i].iov_len -= left;
      }
    }
    s0 = s1;
  }
  if (::close(fd) != 0) ok = false;
  if (!ok) {
    err = "cannot write " + path;
    return false;
  }
  if (sha) *sha = h.hexdigest();
  return true;
}

// Lays the encoded chunks (row-group major) out in file order (FileSerializer / RowGroupSerializer
// sequence) and appends the footer.
static void lay_out(PqImage* out, bool trace, int64_t nrg, int ncol, int64_t n) {
  auto& enc = out->impl->enc;
  // lay out in file order (FileSerializer / RowGroupSerializer sequence)
  const unsigned char* prefix = trace ? kTraceFooterPrefix : kLedgerFooterPrefix;
  const unsigned char* suffix = trace ? kTraceFooterSuffix : kLedgerFooterSuffix;
  const size_t prefix_len = trace ? sizeof(kTraceFooterPrefix) : sizeof(kLedgerFooterPrefix);
  const size_t suffix_len = trace ? sizeof(kTraceFooterSuffix) : sizeof(kLedgerFooterSuffix);
  Layout& L = out->impl->L;
  L = Layout();
  ByteBuf& o = L.meta;
  o.reserve(4096 + prefix_len + suffix_len + enc.size() * 2048);
  o.append("PAR1", 4);  // FileSerializer::StartFile
  L.meta_since(0);
  std::vector<int64_t> rg_bytes((size_t)nrg);
  for (int64_t g = 0; g < nrg; g++) {
    int64_t total_bytes = 0;
    for (int i = 0; i < ncol; i++) total_bytes += enc[(size_t)(g * ncol + i)]->write(L);
    rg_bytes[(size_t)g] = total_bytes;
  }
  // FileMetaData: constant prefix (version, schema), num_rows, row_groups, constant suffix
  const size_t footer_start = o.size();
  o.append(prefix, prefix_len);
  {
    ThriftWriter w(o);
    o.push(0x16);  // field 3, i64
    w.varint(ThriftWriter::zigzag64(n));
    o.push(0x19);  // field 4, list
    if (nrg < 15) {
      o.push((uint8_t)(nrg << 4 | 12));
    } else {
      o.push(0xF0 | 12);
      w.varint((uint64_t)nrg);
    }
    for (int64_t g = 0; g < nrg; g++) {  // RowGroupMetaDataBuilder::Finish
      w.begin_struct();
      w.list(1, 12, (size_t)ncol);
      int64_t total_compressed = 0;
      for (int i = 0; i < ncol; i++) {
        const ColumnEncoder& e = *enc[(size_t)(g * ncol + i)];
        w.raw(e.meta().data(), e.meta().size());
        total_compressed += e.total_compressed();
      }
      w.i64(2, rg_bytes[(size_t)g]);
      w.i64(3, std::min(n, (g + 1) * kRowGroup) - g * kRowGroup);
      w.i64(5, enc[(size_t)(g * ncol)]->first_page_offset());
      w.i64(6, total_compressed);
      w.i16(7, (int16_t)g);
      w.end_struct();
    }
  }
  o.append(suffix, suffix_len);
  const uint32_t footer_len = (uint32_t)(o.size() - footer_start);
  o.append(&footer_len, 4);
  o.append("PAR1", 4);
  L.meta_since(footer_start);
}

bool pq_write_fast(PqSchema schema, const std::vector<PqColumn>& cols, int64_t n, PqImage* out, std::string& err,
                   int threads) {
  const bool trace = schema == PqSchema::TRACE;
  const char* const* names = trace ? kTraceNames : kLedgerNames;
  const PqColumn::Kind* kinds = trace ? kTraceKinds : kLedgerKinds;
  const int ncol = trace ? 7 : 10;
  if (n <= 0 || (int)cols.size() != ncol) {
    err = "fast writer: unsupported shape";
    return false;
  }
  for (int i = 0; i < ncol; i++)
    if (cols[i].kind != kinds[i] || (cols[i].kind == PqColumn::STR && cols[i].code_bytes != 1 && cols[i].code_bytes != 4)) {
      err = "fast writer: unexpected column type";
      return false;
    }
  try {
    // encode every (row group, column) chunk, on up to `threads` threads
    const int64_t nrg = (n + kRowGroup - 1) / kRowGroup;
    auto& enc = out->impl->enc;
    enc.clear();
    enc.resize((size_t)(nrg * ncol));
    for (int64_t g = 0; g < nrg; g++)
      for (int i = 0; i < ncol; i++) enc[(size_t)(g * ncol + i)].reset(new ColumnEncoder(cols[i], names[i]));
    std::atomic<size_t> next{0};
    std::string worker_err;
    std::mutex err_mu;
    auto work = [&]() {
      for (size_t k; (k = next.fetch_add(1)) < enc.size();) {
        const int64_t g = (int64_t)k / ncol;
        try {
          enc[k]->encode(g * kRowGroup, std::min(n, (g + 1) * kRowGroup));
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> l(err_mu);
          worker_err = e.what();
        }
      }
    };
    const int nthreads = (int)std::min<size_t>((size_t)std::max(1, threads), enc.size());
    const auto tp0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 1; t < nthreads; t++) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    if (!worker_err.empty()) {
      err = "fast writer: " + worker_err;
      return false;
    }

    const auto tp1 = std::chrono::steady_clock::now();
    lay_out(out, trace, nrg, ncol, n);
    const auto tp2 = std::chrono::steady_clock::now();
    if (std::getenv("FASTSIM_TIMING"))
      std::fprintf(stderr, "pq: encode %.1f ms (%d threads), layout %.1f ms\n",
                   std::chrono::duration<double, std::milli>(tp1 - tp0).count(), nthreads,
                   std::chrono::duration<double, std::milli>(tp2 - tp1).count());
    return true;
  } catch (const std::exception& e) {
    err = std::string("fast writer: ") + e.what();
    return false;
  }
}

// ---- streaming: encode a growing array of records while it grows (see pqfast.h) ----------------
struct PqStream::Impl {
  PqSchema schema;
  std::vector<PqColumn> rel;  // offsets from the records' base
  std::vector<const RowFeed*> feeds;  // per column
  int ncol;
  struct ColState {
    std::atomic<int> busy{0};
    int64_t pos = 0;  // rows fed
    bool open = false;  // the last encoder in rgs is still being fed
    std::atomic<bool> complete{false};
    const uint8_t* base = nullptr;
    std::vector<std::unique_ptr<ColumnEncoder>> rgs;  // one encoder per row group
  };
  std::unique_ptr<ColState[]> cols;
  std::atomic<int> n_complete{0};
  std::vector<std::thread> threads;
  std::mutex err_mu;
  std::string err;

  Impl(PqSchema s, const std::vector<PqColumn>& r, const std::vector<const RowFeed*>& f)
      : schema(s), rel(r), feeds(f), ncol((int)r.size()) {
    cols.reset(new ColState[(size_t)ncol]);
  }

  // Feeds column c every complete batch available; returns whether it did anything.
  bool advance(int c, const uint8_t* base, int64_t avail, bool done) {
    ColState& st = cols[(size_t)c];
    const char* const* names = schema == PqSchema::TRACE ? kTraceNames : kLedgerNames;
    bool worked = false;
    while (!st.complete) {
      if (!st.open) {  // the next row group starts at pos (a multiple of kRowGroup)
        if (st.pos >= avail) {
          if (done) {
            st.complete = true;
            n_complete.fetch_add(1);
          }
          break;
        }
        st.rgs.emplace_back(new ColumnEncoder(rel[(size_t)c], names[c]));
        st.open = true;
        st.base = nullptr;
      }
      ColumnEncoder& e = *st.rgs.back();
      if (base != st.base) {
        e.rebase(rel[(size_t)c], base);
        st.base = base;
      }
      const int64_t rg_end = (int64_t)st.rgs.size() * kRowGroup;
      const int64_t lim = std::min(avail, rg_end);
      if (st.pos + kBatch <= lim) {
        e.feed(st.pos, kBatch);
        st.pos += kBatch;
        worked = true;
        continue;
      }
      if (done && st.pos < lim) {  // the final, partial batch (lim == avail: rg_end is batch-aligned)
        e.feed(st.pos, lim - st.pos);
        st.pos = lim;
        worked = true;
      }
      if (st.pos == rg_end || (done && st.pos == avail)) {
        e.finish();  // close the row group
        st.open = false;
        worked = true;
        if (done && st.pos == avail) {
          st.complete = true;
          n_complete.fetch_add(1);
          break;
        }
        continue;
      }
      break;  // wait for more rows
    }
    return worked;
  }

  void work(int id) {
    try {
      while (n_complete.load() < ncol) {
        bool worked = false, all_done = true;
        for (int k = 0; k < ncol; k++) {
          const int c = (k + id) % ncol;
          const RowFeed& feed = *feeds[(size_t)c];
          const bool done = feed.done.load(std::memory_order_acquire);
          all_done &= done;
          if (cols[(size_t)c].complete.load(std::memory_order_relaxed) || cols[(size_t)c].busy.exchange(1, std::memory_order_acquire)) continue;
          const int64_t avail = feed.avail.load(std::memory_order_acquire);
          const uint8_t* base = feed.base.load(std::memory_order_relaxed);
          worked |= advance(c, base, avail, done);
          cols[(size_t)c].busy.store(0, std::memory_order_release);
        }
        if (!worked && !all_done) {
          timespec ts{0, 50000};
          nanosleep(&ts, nullptr);
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> l(err_mu);
      err = e.what();
      for (int c = 0; c < ncol; c++) cols[(size_t)c].complete = true;
      n_complete.store(ncol);
    }
  }
};

PqStream::PqStream(PqSchema schema, const std::vector<PqColumn>& rel_cols, const RowFeed& feed, int threads)
    : PqStream(schema, rel_cols, std::vector<const RowFeed*>(rel_cols.size(), &feed), threads) {}

PqStream::PqStream(PqSchema schema, const std::vector<PqColumn>& rel_cols, const std::vector<const RowFeed*>& feeds,
                   int threads)
    : impl(new Impl(schema, rel_cols, feeds)) {
  for (int t = 0; t < std::max(1, threads); t++) impl->threads.emplace_back([this, t] { impl->work(t); });
}

PqStream::~PqStream() {
  for (auto& t : impl->threads)
    if (t.joinable()) t.join();
  delete impl;
}

bool PqStream::finish(PqImage* out, std::string& err) {
  for (auto& t : impl->threads)
    if (t.joinable()) t.join();
  if (!impl->err.empty()) {
    err = "fast writer: " + impl->err;
    return false;
  }
  const bool trace = impl->schema == PqSchema::TRACE;
  const int ncol = impl->ncol;
  if (ncol != (trace ? 7 : 10)) {
    err = "fast writer: unsupported shape";
    return false;
  }
  const int64_t n = impl->feeds[0]->avail.load(std::memory_order_acquire);
  const int64_t nrg = (n + kRowGroup - 1) / kRowGroup;
  if (n <= 0) {
    err = "fast writer: unsupported shape";
    return false;
  }
  auto& enc = out->impl->enc;
  enc.clear();
  for (int64_t g = 0; g < nrg; g++)
    for (int c = 0; c < ncol; c++) {
      auto& rgs = impl->cols[(size_t)c].rgs;
      if ((int64_t)rgs.size() != nrg || impl->cols[(size_t)c].pos != n) {
        err = "fast writer: stream incomplete";
        return false;
      }
      enc.push_back(std::move(rgs[(size_t)g]));
    }
  lay_out(out, trace, nrg, ncol, n);
  return true;
}

}  // namespace fastsim
