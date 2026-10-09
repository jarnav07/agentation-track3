#include "writer.h"


#include <cstdio>
#include <cstdlib>

namespace fastsim {

namespace {

// The baseline's string values, in code order (engine.h). The pandas metadata string the
// baseline stores is part of the constant footer (pq_footer_consts.h).
const char* kTraceMsgNames[] = {"ORDER_SUBMITTED", "ORDER_ACCEPTED", "ORDER_FILLED", "PARTIAL_FILL", "ORDER_CANCELLED", "QUOTE_UPDATE"};
const char* kSideNames[] = {"BID", "ASK"};
const char* kKindNames[] = {"AGENT_WAKEUP", "MarketClosePriceRequestMsg", "MarketHoursRequestMsg", "MarketHoursMsg",
                            "QuerySpreadMsg", "QuerySpreadResponseMsg", "LimitOrderMsg", "CancelOrderMsg",
                            "OrderAcceptedMsg", "OrderExecutedMsg", "OrderCancelledMsg", "MarketClosedMsg",
                            "MarketClosePriceMsg"};

PqColumn i64(const bvec<int64_t>& v, const bvec<uint8_t>* valid = nullptr) {
  PqColumn c;
  c.kind = PqColumn::I64;
  c.data = v.data();
  c.valid = valid ? valid->data() : nullptr;
  return c;
}
PqColumn i32(const bvec<int32_t>& v) {
  PqColumn c;
  c.kind = PqColumn::I32;
  c.data = v.data();
  return c;
}
PqColumn str(const bvec<int32_t>& v, const char* const* names, int k) {
  PqColumn c;
  c.kind = PqColumn::STR;
  c.data = v.data();
  c.names = names;
  c.n_names = k;
  return c;
}
// a field of the LRow records
PqColumn rec(PqColumn::Kind kind, const void* field) {
  PqColumn c;
  c.kind = kind;
  c.data = field;
  c.stride = sizeof(LRow);
  return c;
}

}  // namespace

bool trace_image(const Output& o, PqImage* img, int threads) {
  std::vector<PqColumn> cols = {i64(o.t_ns), i32(o.agent_id), str(o.msg_code, kTraceMsgNames, 6),
                                str(o.side_code, kSideNames, 2), i64(o.price), i64(o.size), i64(o.order_id)};
  std::string why;
  if (pq_write_fast(PqSchema::TRACE, cols, (int64_t)o.t_ns.size(), img, why, threads)) return true;
  if (std::getenv("FASTSIM_VERBOSE")) std::fprintf(stderr, "fast writer declined: %s\n", why.c_str());
  return false;
}

// The ledger's columns as fields of the LRow records at `r`.
static std::vector<PqColumn> ledger_columns(const LRow* r) {
  PqColumn seq;
  seq.kind = PqColumn::I64;
  seq.derive = PqColumn::SEQ;
  PqColumn t_send = rec(PqColumn::I64, &r->t_send);
  t_send.valid = (const uint8_t*)&r->has_send;
  t_send.valid_stride = sizeof(LRow);
  PqColumn lat;  // latency_ns = t_recv - t_send when there is a send time, else 0
  lat.kind = PqColumn::I64;
  lat.derive = PqColumn::DIFF;
  lat.a = &r->t_recv;
  lat.b = &r->t_send;
  lat.cond = (const uint8_t*)&r->has_send;
  lat.stride = sizeof(LRow);
  PqColumn kind = rec(PqColumn::STR, &r->kind);
  kind.code_bytes = 1;
  kind.names = kKindNames;
  kind.n_names = 13;
  PqColumn oid = rec(PqColumn::I64, &r->oid);
  oid.valid = (const uint8_t*)&r->has_oid;
  oid.valid_stride = sizeof(LRow);
  PqColumn causal = rec(PqColumn::I64, &r->causal);
  causal.valid = (const uint8_t*)&r->has_causal;
  causal.valid_stride = sizeof(LRow);
  return {seq, rec(PqColumn::I64, &r->t_recv), t_send, lat, rec(PqColumn::I32, &r->src),
          rec(PqColumn::I32, &r->dst), rec(PqColumn::I64, &r->mid), kind, oid, causal};
}

PqStream* ledger_stream(const RowFeed& feed, int threads) {
  // offsets from the records' base: the columns of a record at address 0
  return new PqStream(PqSchema::LEDGER, ledger_columns((const LRow*)nullptr), feed, threads);
}

bool ledger_image(const Output& o, PqImage* img, int threads) {
  std::vector<PqColumn> cols;
  int64_t n;
  if (o.ledger_rows) {
    n = (int64_t)o.lrows.size();
    if (n == 0) return false;
    cols = ledger_columns(o.lrows.data());
  } else {
    n = (int64_t)o.l_t_recv.size();
    PqColumn seq;
    seq.kind = PqColumn::I64;
    seq.derive = PqColumn::SEQ;
    cols = {seq, i64(o.l_t_recv), i64(o.l_t_send, &o.l_t_send_valid), i64(o.l_latency), i32(o.l_src), i32(o.l_dst),
            i64(o.l_msg_id), str(o.l_kind, kKindNames, 13), i64(o.l_order_id, &o.l_order_valid),
            i64(o.l_causal, &o.l_causal_valid)};
  }
  std::string why;
  if (pq_write_fast(PqSchema::LEDGER, cols, n, img, why, threads)) return true;
  if (std::getenv("FASTSIM_VERBOSE")) std::fprintf(stderr, "fast writer declined: %s\n", why.c_str());
  return false;
}

bool write_trace_parquet(const Output& o, const std::string& path, std::string& err, std::string* sha) {
  PqImage img;
  if (trace_image(o, &img, 1)) return img.write_file(path, sha, err);
  err = "trace: writer declined";
  return false;
}

bool write_ledger_parquet(const Output& o, const std::string& path, std::string& err, std::string* sha) {
  PqImage img;
  if (ledger_image(o, &img, 1)) return img.write_file(path, sha, err);
  err = "ledger: writer declined";
  return false;
}

}  // namespace fastsim
