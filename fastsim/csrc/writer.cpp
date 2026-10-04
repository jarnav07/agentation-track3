#include "writer.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace fastsim {

namespace {

// json.dumps(...) of the pandas metadata that pandas 1.5.3 + pyarrow 15.0.2 store (fastsim/output.py).
const char* kTracePandasMeta =
    "{\"index_columns\": [], \"column_indexes\": [], \"columns\": ["
    "{\"name\": \"t_ns\", \"field_name\": \"t_ns\", \"pandas_type\": \"int64\", \"numpy_type\": \"int64\", \"metadata\": null}, "
    "{\"name\": \"agent_id\", \"field_name\": \"agent_id\", \"pandas_type\": \"int32\", \"numpy_type\": \"int32\", \"metadata\": null}, "
    "{\"name\": \"msg_type\", \"field_name\": \"msg_type\", \"pandas_type\": \"unicode\", \"numpy_type\": \"string\", \"metadata\": null}, "
    "{\"name\": \"side\", \"field_name\": \"side\", \"pandas_type\": \"unicode\", \"numpy_type\": \"string\", \"metadata\": null}, "
    "{\"name\": \"price\", \"field_name\": \"price\", \"pandas_type\": \"int64\", \"numpy_type\": \"int64\", \"metadata\": null}, "
    "{\"name\": \"size\", \"field_name\": \"size\", \"pandas_type\": \"int64\", \"numpy_type\": \"int64\", \"metadata\": null}, "
    "{\"name\": \"order_id\", \"field_name\": \"order_id\", \"pandas_type\": \"int64\", \"numpy_type\": \"int64\", \"metadata\": null}], "
    "\"creator\": {\"library\": \"pyarrow\", \"version\": \"15.0.2\"}, \"pandas_version\": \"1.5.3\"}";

const char* kLedgerPandasMeta =
    "{\"index_columns\": [], \"column_indexes\": [], \"columns\": ["
    "{\"name\": \"seq\", \"field_name\": \"seq\", \"pandas_type\": \"int64\", \"numpy_type\": \"int64\", \"metadata\": null}, "
    "{\"name\": \"t_recv_ns\", \"field_name\": \"t_recv_ns\", \"pandas_type\": \"int64\", \"numpy_type\": \"int64\", \"metadata\": null}, "
    "{\"name\": \"t_send_ns\", \"field_name\": \"t_send_ns\", \"pandas_type\": \"int64\", \"numpy_type\": \"Int64\", \"metadata\": null}, "
    "{\"name\": \"latency_ns\", \"field_name\": \"latency_ns\", \"pandas_type\": \"int64\", \"numpy_type\": \"int64\", \"metadata\": null}, "
    "{\"name\": \"src_id\", \"field_name\": \"src_id\", \"pandas_type\": \"int32\", \"numpy_type\": \"int32\", \"metadata\": null}, "
    "{\"name\": \"dst_id\", \"field_name\": \"dst_id\", \"pandas_type\": \"int32\", \"numpy_type\": \"int32\", \"metadata\": null}, "
    "{\"name\": \"message_id\", \"field_name\": \"message_id\", \"pandas_type\": \"int64\", \"numpy_type\": \"int64\", \"metadata\": null}, "
    "{\"name\": \"msg_type\", \"field_name\": \"msg_type\", \"pandas_type\": \"unicode\", \"numpy_type\": \"string\", \"metadata\": null}, "
    "{\"name\": \"order_id\", \"field_name\": \"order_id\", \"pandas_type\": \"int64\", \"numpy_type\": \"Int64\", \"metadata\": null}, "
    "{\"name\": \"causal_parent\", \"field_name\": \"causal_parent\", \"pandas_type\": \"int64\", \"numpy_type\": \"Int64\", \"metadata\": null}], "
    "\"creator\": {\"library\": \"pyarrow\", \"version\": \"15.0.2\"}, \"pandas_version\": \"1.5.3\"}";

const char* kTraceMsgNames[] = {"ORDER_SUBMITTED", "ORDER_ACCEPTED", "ORDER_FILLED", "PARTIAL_FILL", "ORDER_CANCELLED", "QUOTE_UPDATE"};
const char* kSideNames[] = {"BID", "ASK"};
const char* kKindNames[] = {"AGENT_WAKEUP", "MarketClosePriceRequestMsg", "MarketHoursRequestMsg", "MarketHoursMsg",
                            "QuerySpreadMsg", "QuerySpreadResponseMsg", "LimitOrderMsg", "CancelOrderMsg",
                            "OrderAcceptedMsg", "OrderExecutedMsg", "OrderCancelledMsg", "MarketClosedMsg",
                            "MarketClosePriceMsg"};

template <typename T>
std::shared_ptr<arrow::Buffer> wrap(const std::vector<T>& v) {
  return arrow::Buffer::Wrap(v.data(), v.size());
}

template <typename ArrowType, typename T>
std::shared_ptr<arrow::Array> prim(const std::vector<T>& v) {
  auto data = arrow::ArrayData::Make(std::make_shared<ArrowType>(), (int64_t)v.size(), {nullptr, wrap(v)}, 0);
  return arrow::MakeArray(data);
}

// int64 with a validity vector (1 = valid); bitmap only when something is null.
std::shared_ptr<arrow::Array> nullable_i64(const std::vector<int64_t>& v, const std::vector<uint8_t>& valid,
                                           std::vector<uint8_t>& bitmap_storage) {
  const int64_t n = (int64_t)v.size();
  int64_t nulls = 0;
  for (uint8_t b : valid) nulls += (b == 0);
  std::shared_ptr<arrow::Buffer> bitmap;
  if (nulls) {
    bitmap_storage.assign((size_t)((n + 7) / 8), 0);
    for (int64_t i = 0; i < n; i++)
      if (valid[i]) bitmap_storage[i >> 3] |= (uint8_t)(1u << (i & 7));
    bitmap = wrap(bitmap_storage);
  }
  auto data = arrow::ArrayData::Make(arrow::int64(), n, {bitmap, wrap(v)}, nulls);
  return arrow::MakeArray(data);
}

struct StrStorage {
  std::vector<int32_t> offsets;
  std::vector<char> chars;
};

// Plain utf8 array from codes into a name table (no nulls: every code is valid here).
std::shared_ptr<arrow::Array> strings(const std::vector<int32_t>& codes, const char* const* names, int n_names,
                                      StrStorage& st) {
  std::vector<int32_t> lens(n_names);
  for (int k = 0; k < n_names; k++) lens[k] = (int32_t)std::strlen(names[k]);
  const size_t n = codes.size();
  st.offsets.resize(n + 1);
  st.offsets[0] = 0;
  size_t total = 0;
  for (size_t i = 0; i < n; i++) {
    total += (size_t)lens[codes[i]];
    st.offsets[i + 1] = (int32_t)total;
  }
  st.chars.resize(total ? total : 1);
  char* w = st.chars.data();
  for (size_t i = 0; i < n; i++) {
    const int32_t c = codes[i];
    std::memcpy(w, names[c], (size_t)lens[c]);
    w += lens[c];
  }
  auto data = arrow::ArrayData::Make(arrow::utf8(), (int64_t)n, {nullptr, wrap(st.offsets), arrow::Buffer::Wrap(st.chars.data(), total)}, 0);
  return arrow::MakeArray(data);
}

bool use_threads() {
  const char* v = std::getenv("FASTSIM_WRITER_THREADS");
  return v && v[0] == '1';  // off by default: pool start-up costs more than it saves at these sizes
}

bool write_table(const std::shared_ptr<arrow::Table>& table, const std::string& path, std::string& err) {
  // pyarrow.parquet.write_table defaults (pyarrow/_parquet.pyx, 15.0.2)
  parquet::WriterProperties::Builder pb;
  pb.data_page_version(parquet::ParquetDataPageVersion::V1)
      ->version(parquet::ParquetVersion::PARQUET_2_6)
      ->compression(parquet::Compression::SNAPPY)
      ->enable_dictionary()
      ->enable_statistics()
      ->max_row_group_length(64LL * 1024 * 1024)
      ->disable_page_checksum()
      ->disable_write_page_index();
  parquet::ArrowWriterProperties::Builder ab;
  ab.store_schema()
      ->disable_deprecated_int96_timestamps()
      ->disallow_truncated_timestamps()
      ->enable_compliant_nested_types()
      // Encode the columns of a row group on Arrow's CPU pool. Column chunks are still serialised in
      // schema order, so the file bytes do not change (verified by the oracle).
      ->set_use_threads(use_threads());
  auto out = arrow::io::FileOutputStream::Open(path);
  if (!out.ok()) { err = out.status().ToString(); return false; }
  const int64_t rows = table->num_rows();
  const int64_t chunk = std::min<int64_t>(rows, 1024 * 1024);  // ParquetWriter.write_table(row_group_size=None)
  auto st = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, chunk, pb.build(), ab.build());
  if (!st.ok()) { err = st.ToString(); return false; }
  st = (*out)->Close();
  if (!st.ok()) { err = st.ToString(); return false; }
  return true;
}

}  // namespace

bool write_trace_parquet(const Output& o, const std::string& path, std::string& err) {
  StrStorage s_msg, s_side;
  auto schema = arrow::schema(
      {arrow::field("t_ns", arrow::int64()), arrow::field("agent_id", arrow::int32()),
       arrow::field("msg_type", arrow::utf8()), arrow::field("side", arrow::utf8()),
       arrow::field("price", arrow::int64()), arrow::field("size", arrow::int64()),
       arrow::field("order_id", arrow::int64())},
      arrow::key_value_metadata({"pandas"}, {kTracePandasMeta}));
  auto table = arrow::Table::Make(
      schema, {prim<arrow::Int64Type>(o.t_ns), prim<arrow::Int32Type>(o.agent_id),
               strings(o.msg_code, kTraceMsgNames, 6, s_msg), strings(o.side_code, kSideNames, 2, s_side),
               prim<arrow::Int64Type>(o.price), prim<arrow::Int64Type>(o.size), prim<arrow::Int64Type>(o.order_id)});
  return write_table(table, path, err);
}

bool write_ledger_parquet(const Output& o, const std::string& path, std::string& err) {
  const size_t n = o.l_t_recv.size();
  std::vector<int64_t> seq(n);
  for (size_t i = 0; i < n; i++) seq[i] = (int64_t)i;
  std::vector<uint8_t> bm1, bm2, bm3;
  StrStorage s_kind;
  auto schema = arrow::schema(
      {arrow::field("seq", arrow::int64()), arrow::field("t_recv_ns", arrow::int64()),
       arrow::field("t_send_ns", arrow::int64()), arrow::field("latency_ns", arrow::int64()),
       arrow::field("src_id", arrow::int32()), arrow::field("dst_id", arrow::int32()),
       arrow::field("message_id", arrow::int64()), arrow::field("msg_type", arrow::utf8()),
       arrow::field("order_id", arrow::int64()), arrow::field("causal_parent", arrow::int64())},
      arrow::key_value_metadata({"pandas"}, {kLedgerPandasMeta}));
  auto table = arrow::Table::Make(
      schema, {prim<arrow::Int64Type>(seq), prim<arrow::Int64Type>(o.l_t_recv),
               nullable_i64(o.l_t_send, o.l_t_send_valid, bm1), prim<arrow::Int64Type>(o.l_latency),
               prim<arrow::Int32Type>(o.l_src), prim<arrow::Int32Type>(o.l_dst), prim<arrow::Int64Type>(o.l_msg_id),
               strings(o.l_kind, kKindNames, 13, s_kind), nullable_i64(o.l_order_id, o.l_order_valid, bm2),
               nullable_i64(o.l_causal, o.l_causal_valid, bm3)});
  return write_table(table, path, err);
}

}  // namespace fastsim
