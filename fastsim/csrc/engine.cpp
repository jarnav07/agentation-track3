// See engine.h. Section comments name the pyengine.py method each block ports.
#include "engine.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <immintrin.h>
#include <sys/mman.h>
#include <time.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <memory>

namespace fastsim {

namespace {

// ----------------------------------------------------------------------------- spec parsing
struct Reader {
  const char* p;
  const char* end;
  bool ok = true;
  template <typename T>
  T get() {
    T v{};
    if (p + sizeof(T) > end) {
      ok = false;
      return v;
    }
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
};

// Python floor division for ints.
inline int64_t floordiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

// Python round(float) -> int (round half to even), as int64.
inline int64_t py_round(double x) { return (int64_t)std::nearbyint(x); }

// ----------------------------------------------------------------------------- engine state
constexpr uint64_t kSign = (uint64_t)1 << 63;
// Heap entry keyed on (t, sender, recipient, mid): the tuple order of ABIDES's PriorityQueue
// entries, packed as k1 = t with its sign bit flipped (unsigned order == signed order) and
// k2 = sender << 32 | recipient (agent ids are non-negative); mid only breaks exact ties.
struct Ev {
  uint64_t k1, k2;
  int64_t mid;
  int32_t payload;  // message slot, or -1 for a wait-list proxy (see the main loop)
  uint32_t ver;     // proxy version
  int64_t t() const { return (int64_t)(k1 ^ kSign); }
  int32_t sender() const { return (int32_t)(k2 >> 32); }
  int32_t recipient() const { return (int32_t)(uint32_t)k2; }
};
inline Ev make_ev(int64_t t, int32_t sender, int32_t recipient, int64_t mid, int32_t payload) {
  return Ev{(uint64_t)t ^ kSign, ((uint64_t)(uint32_t)sender << 32) | (uint32_t)recipient, mid, payload, 0};
}
inline bool ev_greater(const Ev& a, const Ev& b) {
  if (__builtin_expect(a.k1 != b.k1, 1)) return a.k1 > b.k1;
  if (a.k2 != b.k2) return a.k2 > b.k2;
  return a.mid > b.mid;
}
// Same order, evaluated without branches (for the heap's data-dependent child selection).
inline bool ev_less_bf(const Ev& a, const Ev& b) {
  return (a.k1 < b.k1) | ((a.k1 == b.k1) & ((a.k2 < b.k2) | ((a.k2 == b.k2) & (a.mid < b.mid))));
}

struct Msg {
  int8_t kind;
  int8_t is_bid;
  int8_t has_order_id;
  int8_t closed;
  int8_t has_causal;
  int32_t agent;
  int64_t oid, price, qty;
  int64_t bid_p, ask_p;  // spread response; 0 = None (agents treat a 0 price as None too)
  int64_t t_send, t_recv, causal;
};

struct BookOrder {
  int64_t oid;
  int32_t agent;
  int64_t qty;
  int32_t next, prev;
  int64_t price;
  bool is_bid;
};

struct Level {
  int64_t price;
  int32_t head, tail;
  int64_t total;  // visible quantity (PriceLevel.total_quantity)
};

// Pre-drawn per-agent RNG values (see Engine::act): an agent's stream is consumed in a fixed
// per-action pattern, so a block can be drawn ahead in one tight loop.
struct NoiseDraw {
  int64_t size, offset;
  bool buy;
};
constexpr int kAgentBlock = 16;

struct Trader {
  NoiseDraw nblk[kAgentBlock];
  double gblk[kAgentBlock];
  int npos = kAgentBlock, gpos = kAgentBlock;
  bool hours_known = false, first_wake = true, mkt_closed = false, awaiting_spread = false;
  int64_t known_bid = 0, known_ask = 0;
  std::vector<int64_t> open;   // oids in insertion order (lazy deletion)
  std::vector<double> hist;    // MomentumTrader mid history
};

enum TType : int8_t { T_SUBMITTED = 0, T_ACCEPTED = 1, T_EXEC = 2, T_CANCELLED = 3 };

class Engine {
 public:
  explicit Engine(Spec& s) : S(s) {}
  ~Engine() { pf_finish(); }
  bool run(Output& out, double* sim_sec);

 private:
  Spec& S;
  int64_t now = 0;
  int64_t next_mid = 1;
  int64_t next_oid = 0;
  std::vector<Ev> heap;  // 4-ary min-heap on ev_greater
  // Message payload pool in fixed-size chunks, so references stay valid while handlers allocate.
  static constexpr int kChunkBits = 12;
  std::vector<std::unique_ptr<Msg[]>> chunks;
  int32_t pool_size = 0;
  std::vector<int32_t> free_slots;
  Msg& msg(int32_t i) { return chunks[(size_t)i >> kChunkBits][i & ((1 << kChunkBits) - 1)]; }
  std::vector<int64_t> agent_time, comp_delay;
  // Busy-recipient wait lists (see the main loop): per agent, a min-heap on (sender, mid) of the
  // events ABIDES would keep requeueing, and the version of its one live proxy in the main heap.
  std::vector<std::vector<Ev>> waitq;
  std::vector<uint32_t> wver;
  std::vector<uint8_t> prox_live;
  std::vector<Ev> prox_cur;
  static bool wait_greater(const Ev& a, const Ev& b) { return a.k2 != b.k2 ? a.k2 > b.k2 : a.mid > b.mid; }
  void push_proxy(int32_t r, uint64_t k1) {
    const Ev& w = waitq[r].front();
    Ev p{k1, w.k2, w.mid, -1, ++wver[r]};
    prox_cur[r] = p;
    prox_live[r] = 1;
    enqueue(p);
  }
  int64_t causal = 0;
  bool has_causal = false;
  std::string error;

  // exchange
  std::vector<int32_t> close_price_subs;
  std::vector<Level> bids, asks;  // best level at the BACK (bids ascending, asks descending)
  std::vector<BookOrder> borders;
  std::vector<int32_t> bfree;
  bvec<int32_t> book_idx;  // oid -> BookOrder index or -1

  // traders
  std::vector<Trader> tr;
  // oid -> trader-side order state (oids are dense and handed out in order)
  struct TrOrder {
    int64_t qty;    // remaining qty
    int64_t price;  // limit price
    uint8_t open;   // still in the owner's open-order dict
    uint8_t side;   // is_bid
  };
  RecBuf<TrOrder> tro;

  // oracle
  bool pt_float = false;
  int64_t pt_i = 0;
  double pt_d = 0.0;
  int64_t pv = 0;
  double mst = 0.0, msv = 0.0;

  // trace records (processing order), array-of-structs: one append per event
  struct Row { int64_t t, price, qty, oid; int32_t agent; int8_t type, bid; };
  struct QRow { int64_t t, price, qty; int8_t bid; };
  struct LRow { int64_t mid, t_send, t_recv, oid, causal; int32_t src, dst; int8_t kind, has_send, has_oid, has_causal; };
  RecBuf<Row> rows;
  RecBuf<QRow> qrows;
  RecBuf<LRow> lrows;

  Output* O = nullptr;

  // -------------------------------------------------------------- kernel
  int32_t alloc_msg() {
    if (!free_slots.empty()) {
      int32_t i = free_slots.back();
      free_slots.pop_back();
      return i;
    }
    if ((pool_size >> kChunkBits) == (int32_t)chunks.size())
      chunks.emplace_back(new Msg[(size_t)1 << kChunkBits]);
    return pool_size++;
  }

  // 4-ary min-heaps (shallower than binary for the few hundred pending events a run keeps).
  static void hpush(std::vector<Ev>& heap, const Ev& e) {
    size_t i = heap.size();
    heap.push_back(e);
    Ev* h = heap.data();
    while (i > 0) {
      size_t parent = (i - 1) >> 2;
      if (!ev_greater(h[parent], e)) break;
      h[i] = h[parent];
      i = parent;
    }
    h[i] = e;
  }
  // Place e at hole i and sift it down. Child selection is branch-free: which child is smallest
  // is data-dependent and would otherwise mispredict on most levels.
  static void sift_down(std::vector<Ev>& heap, size_t i, const Ev& e) {
    const size_t n = heap.size();
    Ev* h = heap.data();
    while (true) {
      const size_t c = 4 * i + 1;
      size_t best;
      if (c + 3 < n) {
        const size_t a = c + ev_less_bf(h[c + 1], h[c]);
        const size_t b = c + 2 + ev_less_bf(h[c + 3], h[c + 2]);
        best = ev_less_bf(h[b], h[a]) ? b : a;
      } else {
        if (c >= n) break;
        best = c;
        for (size_t k = c + 1; k < n; k++)
          if (ev_less_bf(h[k], h[best])) best = k;
      }
      if (!ev_greater(e, h[best])) break;
      h[i] = h[best];
      i = best;
    }
    h[i] = e;
  }
  static void hpop(std::vector<Ev>& heap) {
    Ev last = heap.back();
    heap.pop_back();
    if (!heap.empty()) sift_down(heap, 0, last);
  }

  // Pending events: one total order (ev_greater) over two containers. Events due within
  // kCalN * 2^kCalShift ns of `now` (in-flight messages, wait-list proxies, short wakeups) go in a
  // calendar queue: a ring of 64 ns buckets, each a list sorted by the full key, with a bitmap of
  // non-empty buckets. Everything later (typically the agents' next wakeups) goes in `heap`. All
  // calendar events lie in [now, now + window) -- events are never due before `now` and `now` only
  // advances -- so scanning the ring from now's bucket visits them in time order. The next event
  // is the smaller of the calendar's first and the heap's top.
  static constexpr int kCalShift = 6;
  static constexpr size_t kCalN = (size_t)1 << 12;
  struct CNode {
    Ev e;
    int32_t next;
  };
  std::vector<CNode> cnodes;
  std::vector<int32_t> cfree;
  std::vector<int32_t> chead;  // kCalN bucket heads, -1 = empty
  uint64_t cbits[kCalN / 64] = {};
  size_t ccount = 0;
  int64_t corigin = 0;  // start_time; no event is due before it
  uint64_t cslot(int64_t t) const { return (uint64_t)(t - corigin) >> kCalShift; }

  void enqueue(const Ev& e) {
    const int64_t t = e.t();
    if (cslot(t) - cslot(now) >= kCalN) {
      hpush(heap, e);
      return;
    }
    int32_t i;
    if (!cfree.empty()) {
      i = cfree.back();
      cfree.pop_back();
    } else {
      i = (int32_t)cnodes.size();
      cnodes.push_back(CNode{});
    }
    cnodes[i].e = e;
    const size_t b = cslot(t) & (kCalN - 1);
    int32_t* link = &chead[b];
    while (*link >= 0 && ev_greater(e, cnodes[*link].e)) link = &cnodes[*link].next;
    cnodes[i].next = *link;
    *link = i;
    cbits[b >> 6] |= (uint64_t)1 << (b & 63);
    ccount++;
  }
  // bucket holding the calendar's first event (requires ccount > 0)
  size_t cal_first_bucket() const {
    size_t b = cslot(now) & (kCalN - 1);
    size_t w = b >> 6;
    uint64_t m = cbits[w] & (~(uint64_t)0 << (b & 63));
    while (!m) {
      w = (w + 1) & (kCalN / 64 - 1);
      m = cbits[w];
    }
    return (w << 6) | (size_t)__builtin_ctzll(m);
  }
  void cal_pop(size_t b) {
    const int32_t i = chead[b];
    chead[b] = cnodes[i].next;
    if (chead[b] < 0) cbits[b >> 6] &= ~((uint64_t)1 << (b & 63));
    cfree.push_back(i);
    ccount--;
  }

  // One draw of the latency stream (rngs[2]).
  int64_t draw_latency() {
    double value;
    MT19937& rs = S.rngs[2];
    switch (S.lat_model) {
      case L_LOGNORMAL: value = rs.lognormal(S.lat_mu, S.lat_sigma); break;
      case L_UNIFORM: value = rs.uniform(S.lat_min, S.lat_max); break;
      case L_PARETO: {
        double base = S.lat_min > 0 ? S.lat_min : 1.0;
        value = base * (1.0 + rs.pareto(S.lat_alpha));
        break;
      }
      default: value = S.lat_mean;
    }
    // np.clip == minimum(maximum(v, lo), hi), NaN-propagating
    double v1 = (std::isnan(value) || value >= S.lat_min) ? value : S.lat_min;
    double v2 = (std::isnan(v1) || v1 <= S.lat_max) ? v1 : S.lat_max;
    return py_round(v2);
  }

  // Page pre-faulting: the record buffers are reserved up front but only touched as the run fills
  // them, so every new page is a fault taken inside the event loop. A helper thread keeps the pages
  // a few MiB ahead of each fill point resident (MADV_POPULATE_WRITE maps pages without changing
  // their contents, so racing the writer is harmless).
  struct PfBuf { char* base; size_t cap_bytes, rec; size_t done; };
  static constexpr int kPf = 5;
  PfBuf pf_buf[kPf];
  alignas(64) std::atomic<size_t> pf_fill[kPf];  // published record counts
  alignas(64) std::atomic<bool> pf_stop{false};
  bool pf_on = false;
  std::thread pf_thread;
  void pf_publish() {
    pf_fill[0].store(rows.size(), std::memory_order_relaxed);
    pf_fill[1].store(lrows.size(), std::memory_order_relaxed);
    pf_fill[2].store(qrows.size(), std::memory_order_relaxed);
    pf_fill[3].store(tro.size(), std::memory_order_relaxed);
    pf_fill[4].store(book_idx.size(), std::memory_order_relaxed);
  }
  void pf_worker() {
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
    constexpr size_t kStep = (size_t)1 << 20, kLead = (size_t)8 << 20;
    while (!pf_stop.load(std::memory_order_relaxed)) {
      bool worked = false;
      for (int b = 0; b < kPf; b++) {
        PfBuf& B = pf_buf[b];
        const size_t want = std::min(B.cap_bytes, pf_fill[b].load(std::memory_order_relaxed) * B.rec + kLead);
        if (B.done < want) {
          const size_t len = std::min(kStep, B.cap_bytes - B.done);
          if (madvise(B.base + B.done, len, MADV_POPULATE_WRITE) != 0) B.cap_bytes = B.done;  // unsupported / gone
          else B.done += len;
          worked = true;
        }
      }
      if (!worked) {
        timespec ts{0, 20000};
        nanosleep(&ts, nullptr);
      }
    }
  }
  void pf_start() {
    auto set = [](PfBuf& B, void* p, size_t cap, size_t rec) {
      // whole pages only: the helper never touches a page shared with another allocation
      uintptr_t a = ((uintptr_t)p + 4095) & ~(uintptr_t)4095, e = ((uintptr_t)p + cap * rec) & ~(uintptr_t)4095;
      B = PfBuf{(char*)a, e > a ? (size_t)(e - a) : 0, rec, 0};
    };
    set(pf_buf[0], rows.data(), rows.capacity(), sizeof(Row));
    set(pf_buf[1], lrows.data(), lrows.capacity(), sizeof(LRow));
    set(pf_buf[2], qrows.data(), qrows.capacity(), sizeof(QRow));
    set(pf_buf[3], tro.data(), tro.capacity(), sizeof(TrOrder));
    set(pf_buf[4], book_idx.data(), book_idx.capacity(), sizeof(int32_t));
    pf_publish();
    pf_on = true;
    pf_thread = std::thread([this] { pf_worker(); });
  }
  void pf_finish() {
    if (!pf_on) return;
    pf_stop.store(true, std::memory_order_relaxed);
    pf_thread.join();
    pf_on = false;
  }

  // The latency stream is consumed in a fixed order, independent of the simulation state, so it
  // is drawn in blocks: a tight loop of independent draws overlaps the libm calls far better than
  // one draw per message interleaved with event handling. Only the consumption order matters.
  static constexpr int kLatBlock = 256;
  int64_t lat_blk[kLatBlock];
  double lat_g[kLatBlock + 2 * MT19937::kBulk + 2];  // gaussian FIFO for the lognormal model
  size_t lat_g0 = 0, lat_ng = 0;
  int lat_pos = kLatBlock;
  __attribute__((noinline)) void lat_refill() {
    if (S.lat_model == L_LOGNORMAL) {
      // lognormal(mu, sigma) == exp(mu + sigma * std_gauss()); the gaussians come in bulk
      if (lat_ng - lat_g0 < (size_t)kLatBlock) {
        std::memmove(lat_g, lat_g + lat_g0, (lat_ng - lat_g0) * sizeof(double));
        lat_ng -= lat_g0;
        lat_g0 = 0;
        lat_ng += S.rngs[2].gauss_bulk(lat_g + lat_ng, kLatBlock - lat_ng);
      }
      const double mu = S.lat_mu, sigma = S.lat_sigma, lo = S.lat_min, hi = S.lat_max;
      for (int i = 0; i < kLatBlock; i++) {
        const double value = rng_exp(mu + sigma * lat_g[lat_g0 + i]);
        const double v1 = (std::isnan(value) || value >= lo) ? value : lo;
        const double v2 = (std::isnan(v1) || v1 <= hi) ? v1 : hi;
        lat_blk[i] = py_round(v2);
      }
      lat_g0 += kLatBlock;
    } else {
      for (int i = 0; i < kLatBlock; i++) lat_blk[i] = draw_latency();
    }
    lat_pos = 0;
  }
  int64_t get_latency(int32_t s, int32_t r) {
    if (S.lat_model == L_MATRIX) return S.lat_matrix[(size_t)s * S.n_agents + r];
    if (s == r) return 0;
    if (S.lat_model == L_DET) return draw_latency();
    if (__builtin_expect(lat_pos == kLatBlock, 0)) lat_refill();
    return lat_blk[lat_pos++];
  }

  // Kernel.send_message (+ ledger patch). The payload's message fields must already be set.
  void send(int32_t sender, int32_t recipient, int64_t mid, int32_t slot, int64_t delay) {
    int64_t sent_time = now + comp_delay[sender] + delay;
    int64_t deliver_at = sent_time + get_latency(sender, recipient);
    Msg& m = msg(slot);
    m.t_send = sent_time;
    m.t_recv = deliver_at;
    m.causal = causal;
    m.has_causal = has_causal;
    enqueue(make_ev(deliver_at, sender, recipient, mid, slot));
  }

  bool set_wakeup(int32_t agent, int64_t t) {
    if (t < now) {
      error = "set_wakeup() called with requested time not in future";
      return false;
    }
    int32_t slot = alloc_msg();
    msg(slot).kind = K_WAKEUP;
    enqueue(make_ev(t, agent, agent, next_mid++, slot));
    return true;
  }

  void ledger_row(int64_t mid, int32_t src, int32_t dst, const Msg& m) {
    if (m.kind == K_WAKEUP)
      lrows.push_back(LRow{mid, 0, now, 0, 0, src, dst, K_WAKEUP, 0, 0, 0});
    else
      lrows.push_back(LRow{mid, m.t_send, m.t_recv, m.has_order_id ? m.oid : 0, m.has_causal ? m.causal : 0,
                           src, dst, m.kind, 1, m.has_order_id, m.has_causal});
  }

  // -------------------------------------------------------------- exchange
  // ExchangeAgent.send_message: pipeline delay on accept / execute / cancel reports.
  void ex_send(int32_t recipient, int8_t kind, int32_t agent, int64_t oid, int8_t is_bid,
               int64_t price, int64_t qty, bool has_oid) {
    int32_t slot = alloc_msg();
    Msg& m = msg(slot);
    m.kind = kind;
    m.agent = agent;
    m.oid = oid;
    m.is_bid = is_bid;
    m.price = price;
    m.qty = qty;
    m.has_order_id = has_oid;
    int64_t delay = (kind == K_ACCEPTED || kind == K_EXECUTED || kind == K_CANCELLED) ? S.pipeline_delay : 0;
    send(0, recipient, next_mid++, slot, delay);
  }

  void ex_send_plain(int32_t recipient, int8_t kind) { ex_send(recipient, kind, 0, 0, 0, 0, 0, false); }

  void exchange_wakeup(int64_t t) {
    if (t >= S.mkt_close) {
      int64_t mid = next_mid++;  // one MarketClosePriceMsg delivered to every subscriber
      for (int32_t a : close_price_subs) {
        int32_t slot = alloc_msg();
        Msg& m = msg(slot);
        m.kind = K_CLOSE_PRICE;
        m.has_order_id = 0;
        send(0, a, mid, slot, 0);
      }
    }
  }

  // book helpers
  std::vector<Level>& side_book(bool is_bid) { return is_bid ? bids : asks; }
  // position of the level with `price` in a side book, or the insertion point (keeps order).
  // bids ascending, asks descending.
  size_t level_pos(std::vector<Level>& book, bool is_bid, int64_t price, bool& found) {
    size_t lo = 0, hi = book.size();
    while (lo < hi) {
      size_t mid = (lo + hi) / 2;
      bool before = is_bid ? (book[mid].price < price) : (book[mid].price > price);
      if (before) lo = mid + 1; else hi = mid;
    }
    found = lo < book.size() && book[lo].price == price;
    return lo;
  }

  static bool is_match(const Level& lvl, bool is_bid, int64_t price) {
    return is_bid ? price >= lvl.price : price <= lvl.price;
  }

  int32_t new_book_order(int64_t oid, int32_t agent, int64_t qty, int64_t price, bool is_bid) {
    int32_t i;
    if (!bfree.empty()) {
      i = bfree.back();
      bfree.pop_back();
    } else {
      borders.emplace_back();
      i = (int32_t)borders.size() - 1;
    }
    borders[i] = BookOrder{oid, agent, qty, -1, -1, price, is_bid};
    if ((size_t)oid >= book_idx.size()) book_idx.resize((size_t)oid + 1, -1);
    book_idx[oid] = i;
    return i;
  }

  void unlink(Level& lvl, int32_t i) {
    BookOrder& o = borders[i];
    if (o.prev >= 0) borders[o.prev].next = o.next; else lvl.head = o.next;
    if (o.next >= 0) borders[o.next].prev = o.prev; else lvl.tail = o.prev;
    lvl.total -= o.qty;
    book_idx[o.oid] = -1;
    bfree.push_back(i);
  }

  void log_quotes() {
    if (!bids.empty()) {
      qrows.push_back(QRow{now, bids.back().price, bids.back().total, 1});
    }
    if (!asks.empty()) {
      qrows.push_back(QRow{now, asks.back().price, asks.back().total, 0});
    }
  }

  // OrderBook.enter_order
  void enter_order(int64_t oid, int32_t agent, bool is_bid, int64_t price, int64_t qty) {
    auto& book = side_book(is_bid);
    bool found;
    size_t pos = level_pos(book, is_bid, price, found);
    if (!found) book.insert(book.begin() + pos, Level{price, -1, -1, 0});
    Level& lvl = book[pos];
    int32_t i = new_book_order(oid, agent, qty, price, is_bid);
    borders[i].prev = lvl.tail;
    if (lvl.tail >= 0) borders[lvl.tail].next = i; else lvl.head = i;
    lvl.tail = i;
    lvl.total += qty;
  }

  // OrderBook.cancel_order; returns success.
  bool cancel_order(int64_t oid, int32_t agent, bool is_bid, int64_t price, bool quiet) {
    auto& book = side_book(is_bid);
    if (book.empty()) return false;
    if ((size_t)oid >= book_idx.size() || book_idx[oid] < 0) return false;
    int32_t i = book_idx[oid];
    // ABIDES searches only the level at the request's price on the request's side.
    if (borders[i].price != price || borders[i].is_bid != is_bid) return false;
    bool found;
    size_t pos = level_pos(book, is_bid, price, found);
    if (!found) return false;
    Level& lvl = book[pos];
    BookOrder c = borders[i];
    unlink(lvl, i);
    if (lvl.head < 0) book.erase(book.begin() + pos);
    if (!quiet) ex_send(agent, K_CANCELLED, c.agent, c.oid, is_bid, price, c.qty, true);
    return true;
  }

  // OrderBook.execute_order; order fields passed by reference (qty decremented).
  bool execute_order(int64_t oid, int32_t agent, bool is_bid, int64_t price, int64_t& qty) {
    auto& book = side_book(!is_bid);
    if (book.empty()) return false;
    Level& lvl = book.back();
    if (!is_match(lvl, is_bid, price)) return false;
    int32_t ri = lvl.head;
    BookOrder& resting = borders[ri];
    int64_t m_oid = resting.oid, fill_qty;
    int32_t m_agent = resting.agent;
    int64_t fill_price = lvl.price;
    if (qty >= resting.qty) {
      fill_qty = resting.qty;
      unlink(lvl, ri);
      if (lvl.head < 0) book.pop_back();
    } else {
      fill_qty = qty;
      resting.qty -= fill_qty;
      lvl.total -= fill_qty;
    }
    qty -= fill_qty;
    ex_send(m_agent, K_EXECUTED, m_agent, m_oid, !is_bid, fill_price, fill_qty, true);
    ex_send(agent, K_EXECUTED, agent, oid, is_bid, fill_price, fill_qty, true);
    return true;
  }

  // OrderBook.handle_limit_order (with the opt-in STP patch)
  void handle_limit_order(int64_t oid, int32_t agent, bool is_bid, int64_t price, int64_t qty) {
    if (qty <= 0 || price < 0) return;
    while (true) {
      if (S.stp != STP_NONE) {
        auto& opp = side_book(!is_bid);
        if (!opp.empty() && is_match(opp.back(), is_bid, price)) {
          const BookOrder& resting = borders[opp.back().head];
          if (resting.agent == agent) {
            if (S.stp == STP_OLDEST &&
                cancel_order(resting.oid, resting.agent, !is_bid, opp.back().price, false))
              continue;
            if (S.stp != STP_OLDEST) {
              ex_send(agent, K_CANCELLED, agent, oid, is_bid, price, qty, true);
              break;
            }
          }
        }
      }
      if (execute_order(oid, agent, is_bid, price, qty)) {
        if (qty <= 0) break;
      } else {
        enter_order(oid, agent, is_bid, price, qty);
        ex_send(agent, K_ACCEPTED, agent, oid, is_bid, price, qty, true);
        break;
      }
    }
    log_quotes();
  }

  void exchange_receive(int64_t t, int32_t sender, const Msg& m) {
    comp_delay[0] = S.ex_comp_delay;
    int8_t kind = m.kind;
    if (t > S.mkt_close && kind != K_QUERY_SPREAD) {
      ex_send_plain(sender, K_MKT_CLOSED);
      return;
    }
    switch (kind) {
      case K_HOURS_REQ:
        comp_delay[0] = 0;
        ex_send_plain(sender, K_HOURS);
        break;
      case K_CLOSE_PRICE_REQ:
        close_price_subs.push_back(sender);
        break;
      case K_QUERY_SPREAD: {
        int32_t slot = alloc_msg();
        Msg& r = msg(slot);
        r.kind = K_SPREAD_RESP;
        r.has_order_id = 0;
        r.bid_p = bids.empty() ? 0 : bids.back().price;
        r.ask_p = asks.empty() ? 0 : asks.back().price;
        r.closed = t > S.mkt_close;
        send(0, sender, next_mid++, slot, 0);
        break;
      }
      case K_LIMIT:
        handle_limit_order(m.oid, m.agent, m.is_bid, m.price, m.qty);
        break;
      case K_CANCEL:
        cancel_order(m.oid, m.agent, m.is_bid, m.price, false);
        break;
      default:
        break;
    }
  }

  // -------------------------------------------------------------- traders
  void tr_send(int32_t a, int8_t kind, int64_t oid, bool is_bid, int64_t price, int64_t qty, bool has_oid) {
    int32_t slot = alloc_msg();
    Msg& m = msg(slot);
    m.kind = kind;
    m.agent = a;
    m.oid = oid;
    m.is_bid = is_bid;
    m.price = price;
    m.qty = qty;
    m.has_order_id = has_oid;
    send(a, 0, next_mid++, slot, 0);
  }

  void row(int64_t t, int32_t agent, int8_t tt, bool is_bid, int64_t price, int64_t qty, int64_t oid) {
    rows.push_back(Row{t, price, qty, oid, agent, tt, (int8_t)is_bid});
  }


  bool trader_wakeup(int32_t a, int64_t t) {
    Trader& T = tr[a];
    if (T.first_wake) {
      T.first_wake = false;
      tr_send(a, K_CLOSE_PRICE_REQ, 0, false, 0, 0, false);
    }
    if (!T.hours_known) tr_send(a, K_HOURS_REQ, 0, false, 0, 0, false);
    if (!T.hours_known || T.mkt_closed) return true;
    if (!set_wakeup(a, t + S.agents[a].interval_ns)) return false;
    tr_send(a, K_QUERY_SPREAD, 0, false, 0, 0, false);
    T.awaiting_spread = true;
    return true;
  }

  bool trader_receive(int32_t a, int64_t t, const Msg& m) {
    Trader& T = tr[a];
    bool had_hours = T.hours_known;
    switch (m.kind) {
      case K_HOURS: T.hours_known = true; break;
      case K_MKT_CLOSED: T.mkt_closed = true; break;
      case K_EXECUTED:
        row(t, m.agent, T_EXEC, m.is_bid, m.price, m.qty, m.oid);
        if (tro[m.oid].open) {
          if (m.qty >= tro[m.oid].qty) tro[m.oid].open = 0;
          else tro[m.oid].qty -= m.qty;
        }
        break;
      case K_ACCEPTED: row(t, m.agent, T_ACCEPTED, m.is_bid, m.price, m.qty, m.oid); break;
      case K_CANCELLED:
        row(t, m.agent, T_CANCELLED, m.is_bid, m.price, m.qty, m.oid);
        tro[m.oid].open = 0;
        break;
      case K_SPREAD_RESP:
        if (m.closed) T.mkt_closed = true;
        T.known_bid = m.bid_p;
        T.known_ask = m.ask_p;
        break;
      default: break;
    }
    if (T.hours_known && !had_hours)
      if (!set_wakeup(a, S.mkt_open + 0)) return false;
    if (T.awaiting_spread && m.kind == K_SPREAD_RESP) {
      if (!T.mkt_closed) act(a, t);
      T.awaiting_spread = false;
    }
    return true;
  }

  void place_limit_order(int32_t a, int64_t t, int64_t qty, bool is_bid, int64_t price) {
    int64_t oid = next_oid++;
    tro.push_back(TrOrder{qty, price, 1, (uint8_t)is_bid});  // index oid
    tr[a].open.push_back(oid);
    tr_send(a, K_LIMIT, oid, is_bid, price, qty, true);
    row(t, a, T_SUBMITTED, is_bid, price, qty, oid);
  }

  void act(int32_t a, int64_t t) {
    const AgentParams& P = S.agents[a];
    Trader& T = tr[a];
    const int64_t bid = T.known_bid, ask = T.known_ask;  // 0 == None
    MT19937& rs = S.rngs[2 + a];
    switch (P.kind) {
      case A_NOISE: {
        // One action draws normal, randint(0, 2), randint(0, offset + 1), always in that order.
        if (T.npos == kAgentBlock) {
          for (int k = 0; k < kAgentBlock; k++) {
            NoiseDraw& d = T.nblk[k];
            d.size = py_round(rs.normal(P.order_size_mean, P.order_size_std));
            if (d.size < 1) d.size = 1;
            d.buy = rs.randint(0, 2) != 0;
            d.offset = rs.randint(0, P.price_offset_ticks + 1);
          }
          T.npos = 0;
        }
        const NoiseDraw& d = T.nblk[T.npos++];
        const int64_t size = d.size, offset = d.offset;
        const bool buy = d.buy;
        if (buy) {
          int64_t anchor = ask ? ask : (bid ? bid : P.reference_price);
          place_limit_order(a, t, size, true, anchor + offset);
        } else {
          int64_t anchor = bid ? bid : (ask ? ask : P.reference_price);
          place_limit_order(a, t, size, false, anchor - offset);
        }
        break;
      }
      case A_MM: {
        int64_t mid = (bid && ask) ? floordiv(bid + ask, 2) : P.reference_price;
        // cancel_all_orders: insertion order over the still-open orders (compacting as we go)
        auto& open = T.open;
        size_t w = 0;
        for (size_t r = 0; r < open.size(); r++) {
          int64_t oid = open[r];
          const TrOrder& o = tro[oid];
          if (!o.open) continue;
          open[w++] = oid;
          tr_send(a, K_CANCEL, oid, o.side, o.price, o.qty, true);
        }
        open.resize(w);
        int64_t half = floordiv(P.spread_ticks, 2);
        for (int64_t lvl = 0; lvl < P.depth_levels; lvl++) {
          place_limit_order(a, t, P.size_per_level, true, mid - half - lvl);
          place_limit_order(a, t, P.size_per_level, false, mid + half + lvl);
        }
        break;
      }
      case A_VALUE: {
        double mid;
        if (bid && ask) mid = (double)(bid + ask) / 2.0;
        else if (bid) mid = (double)bid;
        else if (ask) mid = (double)ask;
        else return;
        int64_t fundamental = observe_price(t, T, rs, P.sigma_n);
        if (mid < (double)(fundamental - P.threshold_ticks) && ask)
          place_limit_order(a, t, P.size, true, ask);
        else if (mid > (double)(fundamental + P.threshold_ticks) && bid)
          place_limit_order(a, t, P.size, false, bid);
        break;
      }
      default: {  // A_MOMENTUM
        double mid;
        if (bid && ask) mid = (double)(bid + ask) / 2.0;
        else if (bid) mid = (double)bid;
        else if (ask) mid = (double)ask;
        else return;
        auto& h = T.hist;
        h.push_back(mid);
        if ((int64_t)h.size() > P.lookback + 1) h.erase(h.begin());
        if ((int64_t)h.size() <= P.lookback) return;
        double past = h[0];
        if (mid > past + (double)P.threshold_ticks && ask)
          place_limit_order(a, t, P.size, true, ask);
        else if (mid < past - (double)P.threshold_ticks && bid)
          place_limit_order(a, t, P.size, false, bid);
        break;
      }
    }
  }

  // -------------------------------------------------------------- oracle
  // Types follow the baseline: timestamps are Python ints until a megashock makes them
  // np.float64; int-vs-np.float64 comparisons convert the int to float64 (NumPy semantics).
  int64_t compute_fundamental(bool ts_float, int64_t ts_i, double ts_d, bool adj_float, double adj) {
    double d;
    if (!ts_float && !pt_float) d = (double)(ts_i - pt_i);
    else d = (ts_float ? ts_d : (double)ts_i) - (pt_float ? pt_d : (double)pt_i);
    const double mu = (double)S.r_bar;
    const double gamma = S.kappa, theta = S.fund_vol;
    const double loc = mu + (double)(pv - S.r_bar) * std::exp(-gamma * d);
    const double scale = std::sqrt((std::pow(theta, 2.0) / (2 * gamma)) * (1 - std::exp((-2 * gamma) * d)));
    double v = S.rngs[1].normal(loc, scale);
    if (adj_float) v += adj;
    int64_t vi = (v > 0) ? py_round(v) : 0;
    for (Jump& j : S.jumps) {
      bool reached = ts_float ? (ts_d >= (double)j.time_ns) : (ts_i >= j.time_ns);
      if (!j.consumed && reached) {
        vi = std::max<int64_t>(0, vi + j.magnitude);
        j.consumed = true;
      }
    }
    pt_float = ts_float;
    pt_i = ts_i;
    pt_d = ts_d;
    pv = vi;
    return vi;
  }

  int64_t advance_fundamental(int64_t current_time) {
    bool le = pt_float ? ((double)current_time <= pt_d) : (current_time <= pt_i);
    if (le) return pv;
    while (mst < (double)current_time) {
      compute_fundamental(true, 0, mst, true, msv);
      mst = pt_d + std::trunc(S.rngs[0].exponential(S.ms_scale));
      msv = S.rngs[1].normal(S.ms_mean, S.ms_sd);
      if (S.rngs[1].randint(0, 2) != 0) msv = -msv;
    }
    return compute_fundamental(false, current_time, 0.0, false, 0.0);
  }

  int64_t observe_price(int64_t t, Trader& T, MT19937& rs, double sigma_n) {
    int64_t r_t = advance_fundamental(t >= S.mkt_close ? S.mkt_close - 1 : t);
    if (sigma_n == 0) return r_t;
    // normal(loc, scale) == loc + scale * std_gauss(); the agent's stream is only ever gaussians
    if (T.gpos == kAgentBlock) {
      for (int k = 0; k < kAgentBlock; k++) T.gblk[k] = rs.std_gauss();
      T.gpos = 0;
    }
    return py_round((double)r_t + std::sqrt(sigma_n) * T.gblk[T.gpos++]);
  }

  void assemble();
};

bool Engine::run(Output& out, double* sim_sec) {
  O = &out;
  const auto t_start = std::chrono::steady_clock::now();
  const int32_t n = S.n_agents;
  now = S.start_time;
  agent_time.assign(n, S.start_time);
  waitq.assign(n, {});
  wver.assign(n, 0);
  prox_live.assign(n, 0);
  prox_cur.assign(n, Ev{});
  comp_delay.assign(n, 50);
  tr.assign(n, Trader{});
  pt_float = false;
  pt_i = S.mkt_open;
  pv = S.r_bar;
  mst = S.first_mst;
  msv = S.first_msv;
  heap.reserve(4096);
  cnodes.reserve(4096);
  chead.assign(kCalN, -1);
  corigin = S.start_time;
  // Capacity estimates from the scheduled wakeup count and the orders those wakeups can place
  // (one per trader wakeup, 2 x depth per market-maker rebalance). Reservations are address space
  // only (pages are touched as records are written), so they are generous: growing a big buffer
  // mid-run means copying it and faulting in a fresh one.
  {
    double wakeups = 0, orders = 0;
    const double horizon = (double)(S.mkt_close - S.mkt_open);
    for (int32_t a = 1; a < n; a++) {
      const double w = horizon / (double)std::max<int64_t>(1, S.agents[a].interval_ns);
      wakeups += w;
      orders += S.agents[a].kind == A_MM ? w * 2.0 * (double)std::max<int64_t>(1, S.agents[a].depth_levels) : w;
    }
    auto cap = [](double v, size_t rec) {  // at most 4 GiB of address space per buffer
      return (size_t)std::min(v + 4096.0, (double)((size_t)4 << 30) / (double)rec);
    };
    tro.reserve(cap(orders * 1.25, sizeof(TrOrder)));
    book_idx.reserve(cap(orders * 1.25, sizeof(int32_t)));
    rows.reserve(cap(orders * 4.0, sizeof(Row)));
    qrows.reserve(cap(orders * 2.5, sizeof(QRow)));
    lrows.reserve(cap(wakeups * 4.0 + orders * 6.0, sizeof(LRow)));
  }

  // (a thread costs more than it saves on small runs)
  const size_t reserved = rows.capacity() * sizeof(Row) + lrows.capacity() * sizeof(LRow) +
                          qrows.capacity() * sizeof(QRow) + tro.capacity() * sizeof(TrOrder);
  if (S.helper_threads > 0 && reserved > ((size_t)16 << 20) && !std::getenv("FASTSIM_NO_PREFAULT")) pf_start();

  if (!set_wakeup(0, S.mkt_close)) { out.error = error; return false; }
  for (int32_t a = 0; a < n; a++)
    if (!set_wakeup(a, S.start_time)) { out.error = error; return false; }
  now = S.start_time;

  // ABIDES requeues an event whose recipient is busy at (agent_time, sender, recipient, mid) and
  // pops it again, possibly many times (a burst of k messages to one agent costs O(k^2) pops).
  // Requeue pops have no effect beyond setting `now` to a time the loop has already reached, so
  // the waiting events are parked in a per-recipient heap instead, represented in the main heap
  // by one proxy carrying exactly the key the first of them would have there:
  // (agent_time[r], min sender, r, min mid). The proxy pops where that event would; it is either
  // requeued (still busy) or hands over the minimum waiting event, which is processed at the
  // proxy's time just as the requeued original would be. A proxy superseded by a smaller key at
  // the same time is stale and dropped without touching `now` (the event it stood for would only
  // have been requeued, at a time already reached).
  while ((!heap.empty() || ccount) && now != 0 && now <= S.stop_time) {
    size_t cb = 0;
    bool from_cal = false;
    if (ccount) {
      cb = cal_first_bucket();
      from_cal = heap.empty() || ev_greater(heap[0], cnodes[chead[cb]].e);
    }
    const Ev top = from_cal ? cnodes[chead[cb]].e : heap[0];
    if (from_cal) cal_pop(cb); else hpop(heap);
    Ev e;
    const int32_t r = top.recipient();
    if (top.payload < 0) {  // wait-list proxy
      if (top.ver != wver[r]) continue;
      now = top.t();
      if (agent_time[r] > now) {  // still busy: requeue with the current minimum
        Ev p = top;
        const Ev& w = waitq[r].front();
        p.k1 = (uint64_t)agent_time[r] ^ kSign;
        p.k2 = w.k2;
        p.mid = w.mid;
        prox_cur[r] = p;
        enqueue(p);
        continue;
      }
      prox_live[r] = 0;
      auto& q = waitq[r];
      e = q.front();
      std::pop_heap(q.begin(), q.end(), wait_greater);
      q.pop_back();
      e.k1 = top.k1;
    } else {
      now = top.t();
      if (agent_time[r] > now) {  // recipient busy: park the event
        auto& q = waitq[r];
        q.push_back(top);
        std::push_heap(q.begin(), q.end(), wait_greater);
        const uint64_t k1 = (uint64_t)agent_time[r] ^ kSign;
        if (!prox_live[r]) push_proxy(r, k1);
        else if (prox_cur[r].k1 == k1 && wait_greater(prox_cur[r], top)) push_proxy(r, k1);
        // (a live proxy at an earlier time will requeue to agent_time with the new minimum)
        continue;
      }
      e = top;
    }
    agent_time[r] = e.t();
    const Msg& m = msg(e.payload);
    bool ok = true;
    if (m.kind == K_WAKEUP) {
      causal = e.mid;
      has_causal = true;
      ledger_row(e.mid, r, r, m);
      if (r == 0) exchange_wakeup(e.t());
      else ok = trader_wakeup(r, e.t());
      agent_time[r] += comp_delay[r];
    } else {
      agent_time[r] += comp_delay[r];
      causal = e.mid;
      has_causal = true;
      ledger_row(e.mid, e.sender(), r, m);
      if (r == 0) exchange_receive(e.t(), e.sender(), m);
      else ok = trader_receive(r, e.t(), m);
    }
    free_slots.push_back(e.payload);
    if (!ok) { out.error = error; return false; }
    if (!waitq[r].empty() && !prox_live[r]) push_proxy(r, (uint64_t)agent_time[r] ^ kSign);
    if (pf_on) pf_publish();
  }
  pf_finish();
  if (sim_sec) *sim_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  assemble();
  return true;
}

// See assemble.py for the row-order rules this reproduces.
void Engine::assemble() {
  const size_t n = rows.size();
  // last execution per oid -> ORDER_FILLED
  std::vector<int32_t> last_exec((size_t)next_oid + 1, -1);
  for (size_t i = 0; i < n; i++)
    if (rows[i].type == T_EXEC) last_exec[rows[i].oid] = (int32_t)i;

  // Rows are recorded in processing order, so t never decreases; a stable sort by (t, oid) then
  // only reorders inside runs of equal t. Sort those runs (they are tiny) instead of everything.
  std::vector<int32_t> oidx(n);
  for (size_t i = 0; i < n; i++) oidx[i] = (int32_t)i;
  bool monotone = true;
  for (size_t i = 1; i < n; i++)
    if (rows[i].t < rows[i - 1].t) { monotone = false; break; }
  auto by_t_oid = [&](int32_t a, int32_t b) {
    if (rows[a].t != rows[b].t) return rows[a].t < rows[b].t;
    return rows[a].oid < rows[b].oid;
  };
  if (monotone) {
    for (size_t s0 = 0; s0 < n;) {
      size_t e = s0 + 1;
      while (e < n && rows[e].t == rows[s0].t) e++;
      if (e - s0 > 1) std::stable_sort(oidx.begin() + s0, oidx.begin() + e, by_t_oid);
      s0 = e;
    }
  } else {
    std::stable_sort(oidx.begin(), oidx.end(), by_t_oid);
  }

  // quotes: stable by t, then per t keep last per side, ordered by first appearance
  const size_t nq = qrows.size();
  std::vector<int32_t> qi(nq);
  for (size_t i = 0; i < nq; i++) qi[i] = (int32_t)i;
  bool qmono = true;
  for (size_t i = 1; i < nq; i++)
    if (qrows[i].t < qrows[i - 1].t) { qmono = false; break; }
  if (!qmono) std::stable_sort(qi.begin(), qi.end(), [&](int32_t a, int32_t b) { return qrows[a].t < qrows[b].t; });
  struct Q { int64_t t; int32_t side; int64_t price, qty; };
  std::vector<Q> quotes;
  quotes.reserve(nq);
  for (size_t s0 = 0; s0 < nq;) {
    size_t e = s0;
    const int64_t t = qrows[qi[s0]].t;
    while (e < nq && qrows[qi[e]].t == t) e++;
    int first_side = -1, second_side = -1;
    int64_t lp[2] = {0, 0}, lq[2] = {0, 0};
    for (size_t k = s0; k < e; k++) {
      const QRow& q = qrows[qi[k]];
      int side = q.bid ? 0 : 1;  // side code: 0 BID, 1 ASK
      if (first_side < 0) first_side = side;
      else if (side != first_side && second_side < 0) second_side = side;
      lp[side] = q.price;
      lq[side] = q.qty;
    }
    quotes.push_back(Q{t, first_side, lp[first_side], lq[first_side]});
    if (second_side >= 0) quotes.push_back(Q{t, second_side, lp[second_side], lq[second_side]});
    s0 = e;
  }

  // merge: at equal t, quotes (order_id -1) precede order rows
  const size_t total = n + quotes.size();
  O->t_ns.resize(total); O->agent_id.resize(total); O->msg_code.resize(total); O->side_code.resize(total);
  O->price.resize(total); O->size.resize(total); O->order_id.resize(total);
  size_t a = 0, b = 0;
  for (size_t w = 0; w < total; w++) {
    bool take_quote;
    if (b >= quotes.size()) take_quote = false;
    else if (a >= n) take_quote = true;
    else take_quote = quotes[b].t <= rows[oidx[a]].t;
    if (take_quote) {
      const Q& q = quotes[b++];
      O->t_ns[w] = q.t; O->agent_id[w] = 0; O->msg_code[w] = 5; O->side_code[w] = q.side;
      O->price[w] = q.price; O->size[w] = q.qty; O->order_id[w] = -1;
    } else {
      const int32_t i = oidx[a++];
      const Row& r = rows[i];
      // msg codes: 0 SUBMITTED, 1 ACCEPTED, 2 FILLED, 3 PARTIAL, 4 CANCELLED, 5 QUOTE
      int32_t code;
      switch (r.type) {
        case T_SUBMITTED: code = 0; break;
        case T_ACCEPTED: code = 1; break;
        case T_EXEC: code = (last_exec[r.oid] == i) ? 2 : 3; break;
        default: code = 4;
      }
      O->t_ns[w] = r.t; O->agent_id[w] = r.agent; O->msg_code[w] = code;
      O->side_code[w] = r.bid ? 0 : 1; O->price[w] = r.price; O->size[w] = r.qty; O->order_id[w] = r.oid;
    }
  }

  // ledger: transpose the delivery-ordered rows into columns
  const size_t nl = lrows.size();
  O->l_msg_id.resize(nl); O->l_src.resize(nl); O->l_dst.resize(nl); O->l_kind.resize(nl);
  O->l_t_send.resize(nl); O->l_t_send_valid.resize(nl); O->l_t_recv.resize(nl); O->l_latency.resize(nl);
  O->l_order_id.resize(nl); O->l_order_valid.resize(nl); O->l_causal.resize(nl); O->l_causal_valid.resize(nl);
  for (size_t i = 0; i < nl; i++) {
    const LRow& l = lrows[i];
    O->l_msg_id[i] = l.mid; O->l_src[i] = l.src; O->l_dst[i] = l.dst; O->l_kind[i] = l.kind;
    O->l_t_send[i] = l.t_send; O->l_t_send_valid[i] = l.has_send; O->l_t_recv[i] = l.t_recv;
    O->l_latency[i] = l.has_send ? l.t_recv - l.t_send : 0;
    O->l_order_id[i] = l.oid; O->l_order_valid[i] = l.has_oid;
    O->l_causal[i] = l.causal; O->l_causal_valid[i] = l.has_causal;
  }
}

}  // namespace

bool parse_spec(const char* buf, size_t len, Spec& s, std::string& err) {
  Reader R{buf, buf + len};
  uint32_t magic = R.get<uint32_t>();
  if (magic != 0x46535631u) {  // "FSV1"
    err = "bad spec magic";
    return false;
  }
  s.start_time = R.get<int64_t>();
  s.stop_time = R.get<int64_t>();
  s.mkt_open = R.get<int64_t>();
  s.mkt_close = R.get<int64_t>();
  s.pipeline_delay = R.get<int64_t>();
  s.ex_comp_delay = R.get<int64_t>();
  s.stp = R.get<int32_t>();
  s.n_agents = R.get<int32_t>();
  s.lat_model = R.get<int32_t>();
  s.lat_mean = R.get<double>();
  s.lat_sigma = R.get<double>();
  s.lat_min = R.get<double>();
  s.lat_max = R.get<double>();
  s.lat_alpha = R.get<double>();
  s.lat_mu = R.get<double>();
  if (s.lat_model == L_MATRIX) {
    s.lat_matrix.resize((size_t)s.n_agents * s.n_agents);
    for (auto& v : s.lat_matrix) v = R.get<int64_t>();
  }
  s.r_bar = R.get<int64_t>();
  s.kappa = R.get<double>();
  s.fund_vol = R.get<double>();
  s.ms_scale = R.get<double>();
  s.ms_mean = R.get<double>();
  s.ms_sd = R.get<double>();
  s.first_mst = R.get<double>();
  s.first_msv = R.get<double>();
  int32_t nj = R.get<int32_t>();
  for (int32_t i = 0; i < nj && R.ok; i++) {
    Jump j;
    j.time_ns = R.get<int64_t>();
    j.magnitude = R.get<int64_t>();
    j.consumed = R.get<int32_t>() != 0;
    s.jumps.push_back(j);
  }
  s.agents.assign(s.n_agents, AgentParams{});
  for (int32_t a = 1; a < s.n_agents && R.ok; a++) {
    AgentParams& p = s.agents[a];
    p.kind = R.get<int32_t>();
    p.interval_ns = R.get<int64_t>();
    p.order_size_mean = R.get<double>();
    p.order_size_std = R.get<double>();
    p.price_offset_ticks = R.get<int64_t>();
    p.reference_price = R.get<int64_t>();
    p.spread_ticks = R.get<int64_t>();
    p.depth_levels = R.get<int64_t>();
    p.size_per_level = R.get<int64_t>();
    p.threshold_ticks = R.get<int64_t>();
    p.sigma_n = R.get<double>();
    p.lookback = R.get<int64_t>();
    p.size = R.get<int64_t>();
  }
  int32_t nr = R.get<int32_t>();
  if (nr != s.n_agents + 2) {
    err = "rng stream count mismatch";
    return false;
  }
  s.rngs.resize(nr);
  for (int32_t i = 0; i < nr && R.ok; i++) {
    MT19937& m = s.rngs[i];
    for (int k = 0; k < MT19937::N; k++) m.key[k] = R.get<uint32_t>();
    m.pos = R.get<int32_t>();
    m.has_gauss = R.get<int32_t>();
    m.gauss = R.get<double>();
    m.temper_all();
  }
  if (!R.ok || R.p != R.end) {
    err = "spec length mismatch";
    return false;
  }
  return true;
}

bool run_engine(Spec& spec, Output& out, double* sim_sec) {
  Engine e(spec);
  return e.run(out, sim_sec);
}

}  // namespace fastsim
