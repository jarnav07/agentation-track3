// See engine.h. Section comments name the pyengine.py method each block ports.
#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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
struct Ev {
  int64_t t;
  int32_t sender, recipient;
  int64_t mid;
  int32_t payload;
};
// min-heap on (t, sender, recipient, mid): the tuple order of ABIDES's PriorityQueue entries.
inline bool ev_greater(const Ev& a, const Ev& b) {
  if (a.t != b.t) return a.t > b.t;
  if (a.sender != b.sender) return a.sender > b.sender;
  if (a.recipient != b.recipient) return a.recipient > b.recipient;
  return a.mid > b.mid;
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

struct Trader {
  bool hours_known = false, first_wake = true, mkt_closed = false, awaiting_spread = false;
  int64_t known_bid = 0, known_ask = 0;
  std::vector<int64_t> open;   // oids in insertion order (lazy deletion)
  std::vector<double> hist;    // MomentumTrader mid history
};

enum TType : int8_t { T_SUBMITTED = 0, T_ACCEPTED = 1, T_EXEC = 2, T_CANCELLED = 3 };

class Engine {
 public:
  explicit Engine(Spec& s) : S(s) {}
  bool run(Output& out);

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
  int64_t causal = 0;
  bool has_causal = false;
  std::string error;

  // exchange
  std::vector<int32_t> close_price_subs;
  std::vector<Level> bids, asks;  // best level at the BACK (bids ascending, asks descending)
  std::vector<BookOrder> borders;
  std::vector<int32_t> bfree;
  std::vector<int32_t> book_idx;  // oid -> BookOrder index or -1

  // traders
  std::vector<Trader> tr;
  std::vector<int64_t> tr_qty;     // oid -> trader-side remaining qty
  std::vector<uint8_t> tr_open;    // oid -> still in the owner's open-order dict
  std::vector<uint8_t> tr_side;    // oid -> is_bid
  std::vector<int64_t> tr_price;   // oid -> limit price

  // oracle
  bool pt_float = false;
  int64_t pt_i = 0;
  double pt_d = 0.0;
  int64_t pv = 0;
  double mst = 0.0, msv = 0.0;

  // trace records (processing order)
  std::vector<int64_t> r_t, r_price, r_qty, r_oid;
  std::vector<int32_t> r_agent;
  std::vector<int8_t> r_type, r_bid;
  std::vector<int64_t> q_t, q_price, q_qty;
  std::vector<int8_t> q_bid;

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

  // 4-ary heap: shallower than binary for the few hundred pending events a run keeps.
  void push(const Ev& e) {
    size_t i = heap.size();
    heap.push_back(e);
    while (i > 0) {
      size_t parent = (i - 1) >> 2;
      if (!ev_greater(heap[parent], e)) break;
      heap[i] = heap[parent];
      i = parent;
    }
    heap[i] = e;
  }
  Ev pop() {
    Ev top = heap[0];
    Ev last = heap.back();
    heap.pop_back();
    const size_t n = heap.size();
    if (n) {
      size_t i = 0;
      while (true) {
        size_t c = 4 * i + 1;
        if (c >= n) break;
        size_t best = c;
        size_t end = c + 4 < n ? c + 4 : n;
        for (size_t k = c + 1; k < end; k++)
          if (ev_greater(heap[best], heap[k])) best = k;
        if (!ev_greater(last, heap[best])) break;
        heap[i] = heap[best];
        i = best;
      }
      heap[i] = last;
    }
    return top;
  }

  int64_t get_latency(int32_t s, int32_t r) {
    if (S.lat_model == L_MATRIX) return S.lat_matrix[(size_t)s * S.n_agents + r];
    if (s == r) return 0;
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

  // Kernel.send_message (+ ledger patch). The payload's message fields must already be set.
  void send(int32_t sender, int32_t recipient, int64_t mid, int32_t slot, int64_t delay) {
    int64_t sent_time = now + comp_delay[sender] + delay;
    int64_t deliver_at = sent_time + get_latency(sender, recipient);
    Msg& m = msg(slot);
    m.t_send = sent_time;
    m.t_recv = deliver_at;
    m.causal = causal;
    m.has_causal = has_causal;
    push(Ev{deliver_at, sender, recipient, mid, slot});
  }

  bool set_wakeup(int32_t agent, int64_t t) {
    if (t < now) {
      error = "set_wakeup() called with requested time not in future";
      return false;
    }
    int32_t slot = alloc_msg();
    msg(slot).kind = K_WAKEUP;
    push(Ev{t, agent, agent, next_mid++, slot});
    return true;
  }

  void ledger_row(int64_t mid, int32_t src, int32_t dst, const Msg& m) {
    O->l_msg_id.push_back(mid);
    O->l_src.push_back(src);
    O->l_dst.push_back(dst);
    O->l_kind.push_back(m.kind);
    if (m.kind == K_WAKEUP) {
      O->l_t_send.push_back(0);
      O->l_t_send_valid.push_back(0);
      O->l_t_recv.push_back(now);
      O->l_latency.push_back(0);
      O->l_order_id.push_back(0);
      O->l_order_valid.push_back(0);
      O->l_causal.push_back(0);
      O->l_causal_valid.push_back(0);
    } else {
      O->l_t_send.push_back(m.t_send);
      O->l_t_send_valid.push_back(1);
      O->l_t_recv.push_back(m.t_recv);
      O->l_latency.push_back(m.t_recv - m.t_send);
      O->l_order_id.push_back(m.has_order_id ? m.oid : 0);
      O->l_order_valid.push_back(m.has_order_id);
      O->l_causal.push_back(m.has_causal ? m.causal : 0);
      O->l_causal_valid.push_back(m.has_causal);
    }
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
    if ((size_t)oid >= book_idx.size()) book_idx.resize((size_t)oid + 1 + (book_idx.size() >> 1), -1);
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
      q_t.push_back(now); q_bid.push_back(1); q_price.push_back(bids.back().price); q_qty.push_back(bids.back().total);
    }
    if (!asks.empty()) {
      q_t.push_back(now); q_bid.push_back(0); q_price.push_back(asks.back().price); q_qty.push_back(asks.back().total);
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
    r_t.push_back(t); r_agent.push_back(agent); r_type.push_back(tt); r_bid.push_back(is_bid);
    r_price.push_back(price); r_qty.push_back(qty); r_oid.push_back(oid);
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
        if (tr_open[m.oid]) {
          if (m.qty >= tr_qty[m.oid]) tr_open[m.oid] = 0;
          else tr_qty[m.oid] -= m.qty;
        }
        break;
      case K_ACCEPTED: row(t, m.agent, T_ACCEPTED, m.is_bid, m.price, m.qty, m.oid); break;
      case K_CANCELLED:
        row(t, m.agent, T_CANCELLED, m.is_bid, m.price, m.qty, m.oid);
        tr_open[m.oid] = 0;
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
    if ((size_t)oid >= tr_qty.size()) {
      size_t n = (size_t)oid + 1 + (tr_qty.size() >> 1);
      tr_qty.resize(n); tr_open.resize(n); tr_side.resize(n); tr_price.resize(n);
    }
    tr_qty[oid] = qty; tr_open[oid] = 1; tr_side[oid] = is_bid; tr_price[oid] = price;
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
        int64_t size = py_round(rs.normal(P.order_size_mean, P.order_size_std));
        if (size < 1) size = 1;
        bool buy = rs.randint(0, 2) != 0;
        int64_t offset = rs.randint(0, P.price_offset_ticks + 1);
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
          if (!tr_open[oid]) continue;
          open[w++] = oid;
          tr_send(a, K_CANCEL, oid, tr_side[oid], tr_price[oid], tr_qty[oid], true);
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
        int64_t fundamental = observe_price(t, rs, P.sigma_n);
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

  int64_t observe_price(int64_t t, MT19937& rs, double sigma_n) {
    int64_t r_t = advance_fundamental(t >= S.mkt_close ? S.mkt_close - 1 : t);
    if (sigma_n == 0) return r_t;
    return py_round(rs.normal((double)r_t, std::sqrt(sigma_n)));
  }

  void assemble();
};

bool Engine::run(Output& out) {
  O = &out;
  const int32_t n = S.n_agents;
  now = S.start_time;
  agent_time.assign(n, S.start_time);
  comp_delay.assign(n, 50);
  tr.assign(n, Trader{});
  pt_float = false;
  pt_i = S.mkt_open;
  pv = S.r_bar;
  mst = S.first_mst;
  msv = S.first_msv;
  heap.reserve(4096);
  const size_t guess = 1 << 16;
  r_t.reserve(guess); r_price.reserve(guess); r_qty.reserve(guess); r_oid.reserve(guess);
  r_agent.reserve(guess); r_type.reserve(guess); r_bid.reserve(guess);
  O->l_t_recv.reserve(guess); O->l_t_send.reserve(guess); O->l_latency.reserve(guess); O->l_msg_id.reserve(guess);
  O->l_order_id.reserve(guess); O->l_causal.reserve(guess); O->l_src.reserve(guess); O->l_dst.reserve(guess);
  O->l_kind.reserve(guess); O->l_t_send_valid.reserve(guess); O->l_order_valid.reserve(guess);
  O->l_causal_valid.reserve(guess);

  if (!set_wakeup(0, S.mkt_close)) { out.error = error; return false; }
  for (int32_t a = 0; a < n; a++)
    if (!set_wakeup(a, S.start_time)) { out.error = error; return false; }
  now = S.start_time;

  while (!heap.empty() && now != 0 && now <= S.stop_time) {
    Ev e = pop();
    now = e.t;
    const int32_t r = e.recipient;
    if (agent_time[r] > e.t) {
      e.t = agent_time[r];
      push(e);
      continue;
    }
    agent_time[r] = e.t;
    const Msg& m = msg(e.payload);
    bool ok = true;
    if (m.kind == K_WAKEUP) {
      causal = e.mid;
      has_causal = true;
      ledger_row(e.mid, r, r, m);
      if (r == 0) exchange_wakeup(e.t);
      else ok = trader_wakeup(r, e.t);
      agent_time[r] += comp_delay[r];
    } else {
      agent_time[r] += comp_delay[r];
      causal = e.mid;
      has_causal = true;
      ledger_row(e.mid, e.sender, r, m);
      if (r == 0) exchange_receive(e.t, e.sender, m);
      else ok = trader_receive(r, e.t, m);
    }
    free_slots.push_back(e.payload);
    if (!ok) { out.error = error; return false; }
  }
  assemble();
  return true;
}

// See assemble.py for the row-order rules this reproduces.
void Engine::assemble() {
  const size_t n = r_t.size();
  // last execution per oid -> ORDER_FILLED
  std::vector<int32_t> last_exec((size_t)next_oid + 1, -1);
  for (size_t i = 0; i < n; i++)
    if (r_type[i] == T_EXEC) last_exec[r_oid[i]] = (int32_t)i;
  // msg codes: 0 SUBMITTED, 1 ACCEPTED, 2 FILLED, 3 PARTIAL, 4 CANCELLED, 5 QUOTE
  std::vector<int32_t> code(n);
  for (size_t i = 0; i < n; i++) {
    switch (r_type[i]) {
      case T_SUBMITTED: code[i] = 0; break;
      case T_ACCEPTED: code[i] = 1; break;
      case T_EXEC: code[i] = (last_exec[r_oid[i]] == (int32_t)i) ? 2 : 3; break;
      default: code[i] = 4;
    }
  }
  std::vector<int32_t> oidx(n);
  for (size_t i = 0; i < n; i++) oidx[i] = (int32_t)i;
  std::stable_sort(oidx.begin(), oidx.end(), [&](int32_t a, int32_t b) {
    if (r_t[a] != r_t[b]) return r_t[a] < r_t[b];
    return r_oid[a] < r_oid[b];
  });

  // quotes: stable by t, then per t keep last per side, ordered by first appearance
  const size_t nq = q_t.size();
  std::vector<int32_t> qi(nq);
  for (size_t i = 0; i < nq; i++) qi[i] = (int32_t)i;
  std::stable_sort(qi.begin(), qi.end(), [&](int32_t a, int32_t b) { return q_t[a] < q_t[b]; });
  struct Q { int64_t t; int32_t side; int64_t price, qty; };
  std::vector<Q> quotes;
  quotes.reserve(nq);
  for (size_t s = 0; s < nq;) {
    size_t e = s;
    while (e < nq && q_t[qi[e]] == q_t[qi[s]]) e++;
    int first_side = -1, second_side = -1;
    int64_t lp[2] = {0, 0}, lq[2] = {0, 0};
    for (size_t k = s; k < e; k++) {
      int32_t j = qi[k];
      int side = q_bid[j] ? 0 : 1;  // side code: 0 BID, 1 ASK
      if (first_side < 0) first_side = side;
      else if (side != first_side && second_side < 0) second_side = side;
      lp[side] = q_price[j];
      lq[side] = q_qty[j];
    }
    int64_t t = q_t[qi[s]];
    quotes.push_back(Q{t, first_side, lp[first_side], lq[first_side]});
    if (second_side >= 0) quotes.push_back(Q{t, second_side, lp[second_side], lq[second_side]});
    s = e;
  }

  // merge: at equal t, quotes (order_id -1) precede order rows
  const size_t total = n + quotes.size();
  O->t_ns.reserve(total); O->agent_id.reserve(total); O->msg_code.reserve(total); O->side_code.reserve(total);
  O->price.reserve(total); O->size.reserve(total); O->order_id.reserve(total);
  size_t a = 0, b = 0;
  while (a < n || b < quotes.size()) {
    bool take_quote;
    if (b >= quotes.size()) take_quote = false;
    else if (a >= n) take_quote = true;
    else take_quote = quotes[b].t <= r_t[oidx[a]];
    if (take_quote) {
      const Q& q = quotes[b++];
      O->t_ns.push_back(q.t); O->agent_id.push_back(0); O->msg_code.push_back(5); O->side_code.push_back(q.side);
      O->price.push_back(q.price); O->size.push_back(q.qty); O->order_id.push_back(-1);
    } else {
      int32_t i = oidx[a++];
      O->t_ns.push_back(r_t[i]); O->agent_id.push_back(r_agent[i]); O->msg_code.push_back(code[i]);
      O->side_code.push_back(r_bid[i] ? 0 : 1); O->price.push_back(r_price[i]); O->size.push_back(r_qty[i]);
      O->order_id.push_back(r_oid[i]);
    }
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
  }
  if (!R.ok || R.p != R.end) {
    err = "spec length mismatch";
    return false;
  }
  return true;
}

bool run_engine(Spec& spec, Output& out) {
  Engine e(spec);
  return e.run(out);
}

}  // namespace fastsim
