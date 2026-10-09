// Randomised byte-for-byte check of the self-contained writer (csrc/pqfast.cpp) against the
// libparquet-backed one it replaced (tests/pqfast_libparquet.cpp, which is itself checked against
// WriteTable): null-heavy and all-null columns, few and many distinct values (dictionary fallback
// to PLAIN, several data pages), lengths around batch and row-group boundaries; and the streaming
// encoder (PqStream) on the same tables, fed by a producer thread whose buffer grows and moves.
//   PA=<site-packages>/pyarrow
//   g++ -O2 -std=c++17 -I$PA/include -DHAVE_CONFIG_H tests/test_pqfast.cpp tests/pqfast_libparquet.cpp \
//       csrc/pqfast.cpp csrc/third_party/snappy/snappy*.cc -L$PA -l:libarrow.so.1500 -l:libparquet.so.1500 \
//       -Wl,-rpath,$PA -pthread -o /tmp/test_pqfast && /tmp/test_pqfast
#include <arrow/api.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <thread>

#include "../csrc/pqfast.h"
#include "pqfast_libparquet.h"

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

const char* kMsg[] = {"ORDER_SUBMITTED", "ORDER_ACCEPTED", "ORDER_FILLED", "PARTIAL_FILL", "ORDER_CANCELLED", "QUOTE_UPDATE"};
const char* kSide[] = {"BID", "ASK"};
const char* kKind[] = {"AGENT_WAKEUP", "MarketClosePriceRequestMsg", "MarketHoursRequestMsg", "MarketHoursMsg",
                       "QuerySpreadMsg", "QuerySpreadResponseMsg", "LimitOrderMsg", "CancelOrderMsg",
                       "OrderAcceptedMsg", "OrderExecutedMsg", "OrderCancelledMsg", "MarketClosedMsg",
                       "MarketClosePriceMsg"};

std::shared_ptr<arrow::Schema> trace_schema() {
  return arrow::schema({arrow::field("t_ns", arrow::int64()), arrow::field("agent_id", arrow::int32()),
                        arrow::field("msg_type", arrow::utf8()), arrow::field("side", arrow::utf8()),
                        arrow::field("price", arrow::int64()), arrow::field("size", arrow::int64()),
                        arrow::field("order_id", arrow::int64())},
                       arrow::key_value_metadata({"pandas"}, {kTracePandasMeta}));
}
std::shared_ptr<arrow::Schema> ledger_schema() {
  return arrow::schema({arrow::field("seq", arrow::int64()), arrow::field("t_recv_ns", arrow::int64()),
                        arrow::field("t_send_ns", arrow::int64()), arrow::field("latency_ns", arrow::int64()),
                        arrow::field("src_id", arrow::int32()), arrow::field("dst_id", arrow::int32()),
                        arrow::field("message_id", arrow::int64()), arrow::field("msg_type", arrow::utf8()),
                        arrow::field("order_id", arrow::int64()), arrow::field("causal_parent", arrow::int64())},
                       arrow::key_value_metadata({"pandas"}, {kLedgerPandasMeta}));
}

struct Col {
  std::vector<int64_t> i64;
  std::vector<int32_t> i32;
  std::vector<uint8_t> valid;
};

std::mt19937_64 rng(11);

// values with `distinct` possible values, optionally sorted runs, nulls with probability p_null
void fill(Col& c, int kind, int64_t n, int64_t distinct, double p_null, bool sorted, int n_names) {
  std::uniform_real_distribution<double> u(0, 1);
  c.i64.resize(n);
  c.i32.resize(n);
  c.valid.assign(n, 1);
  int64_t base = (int64_t)(rng() % 1000000) - 500000;
  for (int64_t i = 0; i < n; i++) {
    int64_t v = sorted ? base + i / std::max<int64_t>(1, n / std::max<int64_t>(1, distinct)) : base + (int64_t)(rng() % (uint64_t)distinct);
    if (kind == 2) v = (int64_t)(rng() % (uint64_t)n_names);
    c.i64[i] = v * (kind == 0 ? 1000003 : 1);
    c.i32[i] = (int32_t)v;
    if (u(rng) < p_null) c.valid[i] = 0;
  }
}

}  // namespace

int main() {
  int cases = 0, bad = 0;
  const int64_t lens[] = {1, 2, 7, 1023, 1024, 1025, 5000, 70000, 300000, 1048575, 1048576, 1048577, 1300000};
  for (int t = 0; t < 60; t++) {
    const bool trace = t % 2 == 0;
    const int64_t n = lens[t % 13] + (t >= 26 ? (int64_t)(rng() % 3000) : 0);
    const int ncol = trace ? 7 : 10;
    // kinds: 0 = i64, 1 = i32, 2 = string
    const int tk[] = {0, 1, 2, 2, 0, 0, 0};
    const int lk[] = {0, 0, 0, 0, 1, 1, 0, 2, 0, 0};
    std::vector<Col> data(ncol);
    std::vector<fastsim::PqColumn> a(ncol);
    std::vector<fastsim_ref::PqColumn> b(ncol);
    for (int i = 0; i < ncol; i++) {
      const int kind = trace ? tk[i] : lk[i];
      const int n_names = trace ? (i == 2 ? 6 : 2) : 13;
      const bool nullable = !trace && (i == 2 || i == 8 || i == 9);
      const double p_null = !nullable ? 0.0 : (t % 5 == 0 ? 1.0 : t % 5 == 1 ? 0.99 : t % 5 == 2 ? 0.3 : 0.001);
      const int64_t distinct = (t % 3 == 0) ? 3 : (t % 3 == 1) ? 1 + n / 2 : 1 + n * 4;
      fill(data[i], kind, n, distinct, p_null, (t / 3) % 2 == 0, n_names);
      auto& ca = a[i];
      auto& cb = b[i];
      ca.kind = kind == 0 ? fastsim::PqColumn::I64 : kind == 1 ? fastsim::PqColumn::I32 : fastsim::PqColumn::STR;
      cb.kind = (fastsim_ref::PqColumn::Kind)ca.kind;
      cb.i64 = data[i].i64.data();
      cb.i32 = data[i].i32.data();
      cb.codes = data[i].i32.data();
      ca.data = kind == 0 ? (const void*)data[i].i64.data() : (const void*)data[i].i32.data();
      ca.names = cb.names = trace ? (i == 2 ? kMsg : kSide) : kKind;
      ca.n_names = cb.n_names = n_names;
      ca.valid = cb.valid = nullable ? data[i].valid.data() : nullptr;
    }
    fastsim::ByteBuf img;
    std::string e1, e2;
    fastsim::PqImage pimg;
    const bool ok1 = fastsim::pq_write_fast(trace ? fastsim::PqSchema::TRACE : fastsim::PqSchema::LEDGER, a, n, &pimg, e1, 1 + t % 4);
    if (ok1) pimg.copy_to(img);
    std::shared_ptr<arrow::Buffer> ref;
    const bool ok2 = fastsim_ref::pq_write_fast(trace ? trace_schema() : ledger_schema(), b, n, &ref, e2, 1);
    const bool same = ok1 && ok2 && (int64_t)img.size() == ref->size() && std::memcmp(img.data(), ref->data(), img.size()) == 0;
    if (!same) {
      std::printf("case %d (%s, n=%lld): ok %d/%d sizes %zu/%lld %s %s\n", t, trace ? "trace" : "ledger", (long long)n, ok1,
                  ok2, img.size(), ok2 ? (long long)ref->size() : -1LL, e1.c_str(), e2.c_str());
      bad++;
    }
    cases++;
    if (!same || t % 3 != 0) continue;
    // streaming: the same table as records {i64 per column, valid per column}, appended by a
    // producer in uneven steps into a buffer that is reallocated as it grows
    const size_t rec = (size_t)ncol * 9;
    std::vector<std::vector<uint8_t>*> bufs;  // superseded buffers stay alive until the end
    std::vector<uint8_t>* cur = new std::vector<uint8_t>(rec * 64);
    bufs.push_back(cur);
    fastsim::RowFeed feed;
    std::vector<fastsim::PqColumn> rel(ncol);
    for (int i = 0; i < ncol; i++) {
      rel[i] = a[i];
      rel[i].data = (const void*)(uintptr_t)(i * 8);
      rel[i].stride = rec;
      if (rel[i].kind == fastsim::PqColumn::STR) rel[i].code_bytes = 4;
      rel[i].valid = a[i].valid ? (const uint8_t*)(uintptr_t)(ncol * 8 + i) : nullptr;
      rel[i].valid_stride = rec;
    }
    fastsim::PqStream stream(trace ? fastsim::PqSchema::TRACE : fastsim::PqSchema::LEDGER, rel, feed, 1 + t % 3);
    int64_t have = 0;
    std::mt19937_64 step(t);
    while (have < n) {
      const int64_t k = std::min<int64_t>(n - have, 1 + (int64_t)(step() % 40000));
      if ((size_t)(have + k) * rec > cur->size()) {
        auto* nb = new std::vector<uint8_t>(std::max(cur->size() * 2, (size_t)(have + k) * rec));
        std::memcpy(nb->data(), cur->data(), (size_t)have * rec);
        bufs.push_back(nb);
        cur = nb;
      }
      for (int64_t r = have; r < have + k; r++)
        for (int i = 0; i < ncol; i++) {
          uint8_t* p = cur->data() + (size_t)r * rec;
          const int kind = trace ? tk[i] : lk[i];
          int64_t v = kind == 0 ? data[i].i64[r] : (int64_t)data[i].i32[r];
          if (kind == 0) std::memcpy(p + i * 8, &v, 8);
          else { int32_t w = (int32_t)v; std::memcpy(p + i * 8, &w, 4); }
          p[ncol * 8 + i] = data[i].valid[r];
        }
      have += k;
      feed.publish(cur->data(), have);
    }
    feed.done.store(true, std::memory_order_release);
    fastsim::PqImage simg;
    std::string e3;
    fastsim::ByteBuf sb;
    const bool ok3 = stream.finish(&simg, e3);
    if (ok3) simg.copy_to(sb);
    if (!ok3 || sb.size() != img.size() || std::memcmp(sb.data(), img.data(), img.size()) != 0) {
      std::printf("stream case %d (n=%lld): ok %d sizes %zu/%zu %s\n", t, (long long)n, ok3, sb.size(), img.size(), e3.c_str());
      bad++;
    }
    cases++;
    for (auto* b : bufs) delete b;
  }
  std::printf("%d cases, %d mismatches\n", cases, bad);
  return bad != 0;
}
