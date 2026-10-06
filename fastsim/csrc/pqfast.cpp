// Fast, byte-identical parquet writer for trace.parquet / message_trace.parquet.
//
// parquet::arrow::WriteTable spends most of its time in generic per-value machinery (Arrow memo
// tables, virtual encoders, statistics comparators, zero-filled scratch buffers). This file
// re-implements only the per-value part of parquet-cpp 15.0.2's TypedColumnWriterImpl for the
// cases these files contain -- nullable INT64 / INT32 / UTF8 columns, one level, V1 data pages,
// dictionary encoding with fallback to PLAIN -- and hands everything else to libparquet itself:
// page headers, compression, column-chunk and row-group metadata, schema and footer are produced
// by the library's own PageWriter, *MetaDataBuilder and WriteFileMetaData, so the file structure
// cannot drift. Each step below names the parquet-cpp function whose behaviour it reproduces
// (cpp/src/parquet/column_writer.cc, encoding.cc, statistics.cc at apache-arrow-15.0.2):
//
//  - DoInBatches / WriteBatch: values go in batches of write_batch_size (1024). After each batch
//    the data page is cut if the encoder's estimated size reaches data_pagesize (1 MiB)
//    (CommitWriteAndCheckPageLimit), then the dictionary is abandoned for PLAIN if its encoded
//    size reaches dictionary_pagesize_limit (1 MiB) (CheckDictionarySizeLimit).
//  - DictEncoderImpl: indices in first-occurrence order; data page = bit width byte + RLE/bit-packed
//    hybrid (RleFast: arrow::util::RleEncoder's exact run decisions, inline); estimated size =
//    1 + RlePreserveBufferSize(n, bit_width).
//  - Definition levels: RLE with a 4-byte length prefix (LevelEncoder = RleEncoder, bit width 1).
//  - Statistics: per page min / max / null_count (TypedStatisticsImpl::Update / Encode), merged into
//    the chunk statistics when a page is cut (ResetPageStatistics).
//  - Pages are buffered while the dictionary is in use and written after the dictionary page
//    (Close / FallbackToPlainEncoding); after a fallback they are written eagerly.
// scripts/check_identical.py compares every output byte for byte against the baseline, and
// FASTSIM_LIBPARQUET_WRITER=1 switches back to WriteTable for A/B checks.
#include "pqfast.h"

#include "rle_fast.h"

#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <arrow/util/base64.h>
#include <arrow/util/rle_encoding.h>
#include <parquet/arrow/schema.h>
#include <parquet/column_page.h>
#include <parquet/column_writer.h>
#include <parquet/file_writer.h>
#include <parquet/metadata.h>
#include <parquet/properties.h>
#include <parquet/schema.h>
#include <parquet/statistics.h>

#include <algorithm>
#include <cstring>

namespace fastsim {

namespace {

constexpr int64_t kBatch = 1024;                 // WriterProperties::write_batch_size
constexpr int64_t kDataPageSize = 1024 * 1024;   // data_pagesize
constexpr int64_t kDictPageLimit = 1024 * 1024;  // dictionary_pagesize_limit
constexpr size_t kMaxStatSize = 4096;            // max_statistics_size
constexpr int64_t kRowGroup = 1024 * 1024;       // write_table(row_group_size=None) chunk

std::shared_ptr<parquet::WriterProperties> writer_props() {
  parquet::WriterProperties::Builder pb;
  pb.data_page_version(parquet::ParquetDataPageVersion::V1)
      ->version(parquet::ParquetVersion::PARQUET_2_6)
      ->compression(parquet::Compression::SNAPPY)
      ->enable_dictionary()
      ->enable_statistics()
      ->max_row_group_length(64LL * 1024 * 1024)
      ->disable_page_checksum()
      ->disable_write_page_index();
  return pb.build();
}

std::shared_ptr<parquet::ArrowWriterProperties> arrow_props() {
  parquet::ArrowWriterProperties::Builder ab;
  ab.store_schema()->disable_deprecated_int96_timestamps()->disallow_truncated_timestamps()->enable_compliant_nested_types();
  return ab.build();
}

int bit_width_for(int num_entries) {  // DictEncoderImpl::bit_width
  if (num_entries == 0) return 0;
  if (num_entries == 1) return 1;
  return arrow::bit_util::Log2((uint64_t)num_entries);
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

class ColumnWriter {
 public:
  ColumnWriter(const PqColumn& col, const parquet::ColumnDescriptor* descr, parquet::ColumnChunkMetaDataBuilder* meta,
               std::unique_ptr<parquet::PageWriter> pager, const parquet::WriterProperties& props)
      : col_(col), descr_(descr), meta_(meta), pager_(std::move(pager)), props_(props) {
    is_signed_ = descr_->sort_order() == parquet::SortOrder::SIGNED;
    encoding_ = props_.dictionary_index_encoding();
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
    compressed_ = *arrow::AllocateResizableBuffer(0);
  }

  // Writes rows [begin, end) of the column, then closes the chunk. Returns total bytes written
  // (ColumnWriterImpl::Close's return value).
  int64_t write(int64_t begin, int64_t end) {
    const int64_t total = end - begin;
    const int64_t full = total / kBatch;
    for (int64_t b = 0; b < full; b++) write_batch(begin + b * kBatch, kBatch);
    if (total % kBatch) write_batch(begin + full * kBatch, total % kBatch);
    close();
    return total_bytes_written_;
  }

 private:
  const PqColumn& col_;
  const parquet::ColumnDescriptor* descr_;
  parquet::ColumnChunkMetaDataBuilder* meta_;
  std::unique_ptr<parquet::PageWriter> pager_;
  const parquet::WriterProperties& props_;
  bool is_signed_ = true;
  bool fallback_ = false;
  parquet::Encoding::type encoding_;

  DictMap<int64_t> i64_dict_;
  DictMap<int32_t> i32_dict_;
  std::vector<int32_t> str_index_, str_rank_, str_dict_;  // code -> dict index; code -> rank; dict index -> code
  int dict_encoded_size_ = 0;
  std::vector<int32_t> indices_;
  std::vector<uint8_t> plain_;
  std::vector<uint8_t> def_levels_;  // only for columns with a validity vector
  int64_t page_nulls_ = 0;
  std::vector<uint8_t> page_;          // uncompressed page image

  int64_t num_buffered_values_ = 0, num_buffered_rows_ = 0, rows_written_ = 0, total_bytes_written_ = 0;
  Stats<int64_t> page_stats_, chunk_stats_;  // ints as int64; strings as rank

  std::shared_ptr<arrow::ResizableBuffer> compressed_;
  std::vector<std::unique_ptr<parquet::DataPageV1>> pages_;

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
      return 1 + arrow::util::RleEncoder::MaxBufferSize(bw, (int)indices_.size()) + arrow::util::RleEncoder::MinBufferSize(bw);
    }
    return (int64_t)plain_.size();
  }

  void put_value(int64_t i) {
    switch (col_.kind) {
      case PqColumn::I64: {
        const int64_t v = col_.i64[i];
        if (fallback_) {
          const size_t o = plain_.size();
          plain_.resize(o + 8);
          std::memcpy(plain_.data() + o, &v, 8);
        } else {
          auto r = i64_dict_.get_or_insert(v);
          if (r.second) dict_encoded_size_ += 8;
          indices_.push_back(r.first);
        }
        update_stats(v);
        break;
      }
      case PqColumn::I32: {
        const int32_t v = col_.i32[i];
        if (fallback_) {
          const size_t o = plain_.size();
          plain_.resize(o + 4);
          std::memcpy(plain_.data() + o, &v, 4);
        } else {
          auto r = i32_dict_.get_or_insert(v);
          if (r.second) dict_encoded_size_ += 4;
          indices_.push_back(r.first);
        }
        update_stats(v);
        break;
      }
      default: {
        const int32_t c = col_.codes[i];
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
    }
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

  // One batch of an integer column: dictionary indices (or PLAIN bytes after a fallback) and the
  // page statistics, in a loop specialised on type and nullability.
  template <typename T, bool kNullable>
  int64_t put_ints(const T* vals, int64_t off, int64_t n, DictMap<T>& dict) {
    const uint8_t* valid = col_.valid;
    int64_t nv = 0;
    bool any = page_stats_.has_minmax;
    int64_t mn = page_stats_.min, mx = page_stats_.max;
    if (!fallback_) {
      const size_t base = indices_.size();
      indices_.resize(base + (size_t)n);
      int32_t* out = indices_.data() + base;
      T last{};
      int32_t last_idx = -1;
      for (int64_t i = off; i < off + n; i++) {
        if (kNullable) {
          const uint8_t ok = valid[i] != 0;
          def_levels_.push_back(ok);
          if (!ok) continue;
        }
        const T v = vals[i];
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
      for (int64_t i = off; i < off + n; i++) {
        if (kNullable) {
          const uint8_t ok = valid[i] != 0;
          def_levels_.push_back(ok);
          if (!ok) continue;
        }
        const T v = vals[i];
        const size_t o = plain_.size();
        plain_.resize(o + sizeof(T));
        std::memcpy(plain_.data() + o, &v, sizeof(T));
        nv++;
        if (!any) { any = true; mn = mx = v; }
        else { mn = v < mn ? v : mn; mx = v > mx ? v : mx; }
      }
    }
    page_stats_.has_minmax = any;
    page_stats_.min = mn;
    page_stats_.max = mx;
    return nv;
  }

  int64_t put_strings(int64_t off, int64_t n) {  // never null in these files
    for (int64_t i = off; i < off + n; i++) put_value(i);
    return n;
  }

  void write_batch(int64_t off, int64_t n) {
    int64_t nv;
    switch (col_.kind) {
      case PqColumn::I64:
        nv = col_.valid ? put_ints<int64_t, true>(col_.i64, off, n, i64_dict_) : put_ints<int64_t, false>(col_.i64, off, n, i64_dict_);
        break;
      case PqColumn::I32:
        nv = col_.valid ? put_ints<int32_t, true>(col_.i32, off, n, i32_dict_) : put_ints<int32_t, false>(col_.i32, off, n, i32_dict_);
        break;
      default:
        if (col_.valid) {  // not produced by writer.cpp; keep the generic path correct anyway
          nv = 0;
          for (int64_t i = off; i < off + n; i++) {
            const uint8_t ok = col_.valid[i] != 0;
            def_levels_.push_back(ok);
            if (ok) { put_value(i); nv++; }
          }
        } else {
          nv = put_strings(off, n);
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

  parquet::EncodedStatistics encode(const Stats<int64_t>& s) const {  // TypedStatisticsImpl::Encode
    parquet::EncodedStatistics e;
    if (s.has_minmax) {
      e.set_min(encode_stat(s.min));
      e.set_max(encode_stat(s.max));
    }
    e.set_null_count(s.null_count);
    e.all_null_value = s.num_values == 0;
    e.ApplyStatSizeLimits(kMaxStatSize);
    e.set_is_signed(is_signed_);
    return e;
  }

  void add_data_page() {  // ColumnWriterImpl::AddDataPage + BuildDataPageV1
    const int nbv = (int)num_buffered_values_;
    page_.clear();
    page_.reserve((size_t)(16 + parquet::LevelEncoder::MaxBufferSize(parquet::Encoding::RLE, 1, nbv) +
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
    const arrow::Buffer uncompressed(page_.data(), (int64_t)page_.size());
    page_nulls_ = 0;

    parquet::EncodedStatistics page_stats = encode(page_stats_);
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

    const int64_t first_row_index = rows_written_ - num_buffered_rows_;
    if (!fallback_) {  // keep until the dictionary page is written
      std::shared_ptr<arrow::ResizableBuffer> c = *arrow::AllocateResizableBuffer(0);
      pager_->Compress(uncompressed, c.get());
      pages_.push_back(std::make_unique<parquet::DataPageV1>(c, nbv, encoding_, parquet::Encoding::RLE,
                                                             parquet::Encoding::RLE, uncompressed_size, page_stats,
                                                             first_row_index));
    } else {
      pager_->Compress(uncompressed, compressed_.get());
      parquet::DataPageV1 page(compressed_, nbv, encoding_, parquet::Encoding::RLE, parquet::Encoding::RLE,
                               uncompressed_size, page_stats, first_row_index);
      total_bytes_written_ += pager_->WriteDataPage(page);
    }
    def_levels_.clear();
    num_buffered_values_ = 0;
    num_buffered_rows_ = 0;
  }

  void write_dictionary_page() {
    std::shared_ptr<arrow::ResizableBuffer> b = *arrow::AllocateResizableBuffer(dict_encoded_size_);
    uint8_t* p = b->mutable_data();
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
    parquet::DictionaryPage page(b, entries, props_.dictionary_page_encoding());
    total_bytes_written_ += pager_->WriteDictionaryPage(page);
  }

  void flush_buffered_pages() {  // FlushBufferedDataPages
    if (num_buffered_values_ > 0) add_data_page();
    for (const auto& p : pages_) total_bytes_written_ += pager_->WriteDataPage(*p);
    pages_.clear();
  }

  void fall_back() {  // FallbackToPlainEncoding
    write_dictionary_page();
    flush_buffered_pages();
    fallback_ = true;
    encoding_ = parquet::Encoding::PLAIN;
  }

  void close() {  // ColumnWriterImpl::Close
    if (!fallback_) write_dictionary_page();
    flush_buffered_pages();
    parquet::EncodedStatistics cs = encode(chunk_stats_);
    if (rows_written_ > 0 && cs.is_set()) meta_->SetStatistics(cs);
    pager_->Close(/*has_dictionary=*/true, fallback_);
  }
};

}  // namespace

bool pq_write_fast(const std::shared_ptr<arrow::Schema>& schema, const std::vector<PqColumn>& cols, int64_t n,
                   std::shared_ptr<arrow::Buffer>* out, std::string& err) {
  if (n <= 0 || (int)cols.size() != schema->num_fields()) {
    err = "fast writer: unsupported shape";
    return false;
  }
  try {
    auto props = writer_props();
    auto aprops = arrow_props();
    std::shared_ptr<parquet::SchemaDescriptor> descr;
    auto st = parquet::arrow::ToParquetSchema(schema.get(), *props, *aprops, &descr);
    if (!st.ok()) {
      err = st.ToString();
      return false;
    }
    for (int i = 0; i < descr->num_columns(); i++) {
      const auto* c = descr->Column(i);
      if (c->max_definition_level() != 1 || c->max_repetition_level() != 0 || !props->dictionary_enabled(c->path())) {
        err = "fast writer: unexpected column layout";
        return false;
      }
    }
    // GetSchemaMetadata: the schema's own metadata, then ARROW:schema
    std::shared_ptr<arrow::KeyValueMetadata> kv =
        schema->metadata() ? schema->metadata()->Copy() : arrow::key_value_metadata({}, {});
    auto ser = arrow::ipc::SerializeSchema(*schema, arrow::default_memory_pool());
    if (!ser.ok()) {
      err = ser.status().ToString();
      return false;
    }
    kv->Append("ARROW:schema", arrow::util::base64_encode((*ser)->ToString()));

    auto sink = *arrow::io::BufferOutputStream::Create(1 << 20);
    PARQUET_THROW_NOT_OK(sink->Write("PAR1", 4));  // FileSerializer::StartFile
    auto file_meta = parquet::FileMetaDataBuilder::Make(descr.get(), props);
    int16_t rg_ordinal = 0;
    for (int64_t off = 0; off < n; off += kRowGroup, rg_ordinal++) {
      const int64_t end = std::min(n, off + kRowGroup);
      parquet::RowGroupMetaDataBuilder* rg = file_meta->AppendRowGroup();
      int64_t total_bytes = 0;
      for (int i = 0; i < descr->num_columns(); i++) {
        parquet::ColumnChunkMetaDataBuilder* cm = rg->NextColumnChunk();
        auto pager = parquet::PageWriter::Open(sink, props->compression(cm->descr()->path()), cm, rg_ordinal, (int16_t)i,
                                               props->memory_pool(), false, nullptr, nullptr,
                                               props->page_checksum_enabled(), nullptr, nullptr, parquet::CodecOptions());
        ColumnWriter w(cols[i], cm->descr(), cm, std::move(pager), *props);
        total_bytes += w.write(off, end);
      }
      rg->set_num_rows(end - off);
      rg->Finish(total_bytes, rg_ordinal);
    }
    auto meta = file_meta->Finish(kv);
    parquet::WriteFileMetaData(*meta, sink.get());
    auto buf = sink->Finish();
    if (!buf.ok()) {
      err = buf.status().ToString();
      return false;
    }
    *out = *buf;
    return true;
  } catch (const std::exception& e) {
    err = std::string("fast writer: ") + e.what();
    return false;
  }
}

}  // namespace fastsim
