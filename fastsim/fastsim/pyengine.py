"""Pure-Python reference engine: the ABIDES baseline's semantics, without ABIDES's overheads.

This is a line-by-line re-statement of what the pinned ABIDES kernel, ``ExchangeAgent``,
``OrderBook``/``PriceLevel``, ``TradingAgent`` and the ``abides_fork`` scheduled agents *do* for
the configurations a Track-3 scenario can express — and nothing else. It keeps every observable:

* the kernel's priority order ``(deliver_time, sender_id, recipient_id, message_id)``, including
  re-queueing a message for an agent that is still "in the future" (computation delay);
* every random draw, in the same order, from the same NumPy ``RandomState`` objects;
* message-id and order-id assignment (both are creation-order counters in ABIDES);
* the exact rows of ``trace.parquet`` and ``message_trace.parquet``.

It drops what has no observable effect: logging and its eager f-strings, deep copies of objects
that are never mutated afterwards, holdings / cash bookkeeping (no agent reads it), the exchange's
order history, book_log2 snapshots and end-of-run metrics, pandas.

It is the specification the compiled engine is checked against, and the fallback if the compiled
extension is unavailable. Comments cite the ABIDES code each block mirrors.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

from .scenario import (
    DEFAULT_COMPUTATION_DELAY,
    LAT_DETERMINISTIC,
    LAT_LOGNORMAL,
    LAT_MATRIX,
    LAT_PARETO,
    LAT_UNIFORM,
    MM,
    MOMENTUM,
    NOISE,
    VALUE,
    EngineSpec,
)

# Message kinds. Ledger names are the ABIDES class names (``Message.type()``).
K_WAKEUP = 0
K_CLOSE_PRICE_REQ = 1
K_HOURS_REQ = 2
K_HOURS = 3
K_QUERY_SPREAD = 4
K_SPREAD_RESP = 5
K_LIMIT = 6
K_CANCEL = 7
K_ACCEPTED = 8
K_EXECUTED = 9
K_CANCELLED = 10
K_MKT_CLOSED = 11
K_CLOSE_PRICE = 12
KIND_NAMES = (
    "AGENT_WAKEUP",
    "MarketClosePriceRequestMsg",
    "MarketHoursRequestMsg",
    "MarketHoursMsg",
    "QuerySpreadMsg",
    "QuerySpreadResponseMsg",
    "LimitOrderMsg",
    "CancelOrderMsg",
    "OrderAcceptedMsg",
    "OrderExecutedMsg",
    "OrderCancelledMsg",
    "MarketClosedMsg",
    "MarketClosePriceMsg",
)

# Trace order-row types (EXEC becomes PARTIAL_FILL / ORDER_FILLED at the end).
T_SUBMITTED, T_ACCEPTED, T_EXEC, T_CANCELLED = 0, 1, 2, 3

# Order fields (mutable lists inside the book / exchange; tuples inside messages).
O_ID, O_AGENT, O_BID, O_PRICE, O_QTY = 0, 1, 2, 3, 4


class Result:
    """Columnar outputs of one run."""

    def __init__(self) -> None:
        self.order_rows: list[tuple] = []  # (t, agent, ttype, is_bid, price, qty, oid)
        self.quotes: list[tuple] = []  # (t, is_bid, price, qty) in emission order
        self.ledger: list[tuple] = []  # delivered rows, in delivery (seq) order


class PyEngine:
    def __init__(self, spec: EngineSpec) -> None:
        self.spec = spec
        self.n = spec.n_agents
        self.current_time = spec.start_time
        self.stop_time = spec.stop_time
        self.mkt_open = spec.mkt_open
        self.mkt_close = spec.mkt_close
        self.heap: list[tuple] = []
        self.msg_id = 1  # Message.__message_id_counter after reset
        self.order_id = 0  # Order._order_id_counter after reset
        self.agent_time = [spec.start_time] * self.n
        self.comp_delay = [DEFAULT_COMPUTATION_DELAY] * self.n
        self.causal: int | None = None
        self.res = Result()

        # exchange (agent 0)
        self.ex_computation_delay = spec.exchange_computation_delay
        self.pipeline_delay = spec.pipeline_delay
        self.stp = spec.stp_policy
        self.close_price_subs: list[int] = []
        self.bids: list[list] = []  # price levels, best first: [price, [orders...]]
        self.asks: list[list] = []

        # traders (index = agent id; slot 0 unused)
        self.agents = [None] + spec.agents
        self.mkt_hours_known = [False] * self.n
        self.first_wake = [True] * self.n
        self.mkt_closed = [False] * self.n
        self.awaiting_spread = [False] * self.n
        self.known_bid: list[Any] = [None] * self.n  # best (price, qty) or None
        self.known_ask: list[Any] = [None] * self.n
        self.open_orders: list[dict] = [dict() for _ in range(self.n)]  # oid -> [qty, is_bid, price]
        self.mid_history: list[list] = [[] for _ in range(self.n)]

        # oracle state (SparseMeanRevertingOracle)
        o = spec.oracle
        self.o_rs = o.random_state
        self.o_r = (spec.mkt_open, o.r_bar)
        self.o_mst = o.first_megashock_time
        self.o_msv = o.first_megashock_value
        self.o_jumps = [dict(j) for j in o.scheduled_jumps]
        self.global_rs = spec.global_rs

        # latency
        lat = spec.latency
        self.lat_model = lat.model
        self.lat_rs = lat.random_state
        self.lat = lat

    # ------------------------------------------------------------------ kernel
    def new_msg_id(self) -> int:
        mid = self.msg_id
        self.msg_id = mid + 1
        return mid

    def get_latency(self, sender: int, recipient: int) -> int:
        """``ScenarioLatencyModel.get_latency`` (or the line-distance fallback)."""
        lat = self.lat
        model = self.lat_model
        if model == LAT_MATRIX:
            return int(lat.matrix[sender, recipient])
        if sender == recipient:
            return 0
        if model == LAT_LOGNORMAL:
            value = self.lat_rs.lognormal(mean=lat.mu, sigma=lat.sigma)
        elif model == LAT_UNIFORM:
            value = self.lat_rs.uniform(lat.min_ns, lat.max_ns)
        elif model == LAT_PARETO:
            base = lat.min_ns if lat.min_ns > 0 else 1.0
            value = base * (1.0 + self.lat_rs.pareto(lat.alpha))
        else:
            value = lat.mean_ns
        return int(round(float(np.clip(value, lat.min_ns, lat.max_ns))))

    def send(self, sender: int, recipient: int, mid: int, kind: int, data: Any, delay: int = 0,
             order_id: int | None = None) -> None:
        """``Kernel.send_message`` + the ledger patch."""
        sent_time = self.current_time + self.comp_delay[sender] + delay
        deliver_at = sent_time + int(self.get_latency(sender, recipient))
        row = (mid, sender, recipient, sent_time, deliver_at, deliver_at - sent_time, kind,
               order_id, self.causal)
        heapq.heappush(self.heap, (deliver_at, sender, recipient, mid, kind, data, row))

    def set_wakeup(self, agent: int, requested_time: int) -> None:
        if requested_time < self.current_time:
            raise ValueError("set_wakeup() called with requested time not in future")
        mid = self.new_msg_id()
        heapq.heappush(self.heap, (requested_time, agent, agent, mid, K_WAKEUP, None, None))

    def run(self) -> Result:
        # Kernel.initialize: kernel_initializing (exchange wakes at close), then kernel_starting
        # (every agent, exchange first, asks for a wakeup at start_time).
        self.set_wakeup(0, self.mkt_close)
        for a in range(self.n):
            self.set_wakeup(a, self.spec.start_time)
        self.current_time = self.spec.start_time

        heap = self.heap
        agent_time = self.agent_time
        comp_delay = self.comp_delay
        ledger = self.res.ledger
        stop = self.stop_time
        while heap and self.current_time and self.current_time <= stop:
            entry = heapq.heappop(heap)
            t, sender, recipient, mid, kind, data, row = entry
            self.current_time = t
            if agent_time[recipient] > t:
                heapq.heappush(heap, (agent_time[recipient], sender, recipient, mid, kind, data, row))
                continue
            agent_time[recipient] = t
            if kind == K_WAKEUP:
                self.causal = mid
                ledger.append((mid, recipient, recipient, None, t, 0, K_WAKEUP, None, None))
                if recipient == 0:
                    self.exchange_wakeup(t)
                else:
                    self.trader_wakeup(recipient, t)
                agent_time[recipient] += comp_delay[recipient]
            else:
                agent_time[recipient] += comp_delay[recipient]
                self.causal = mid
                ledger.append(row)
                if recipient == 0:
                    self.exchange_receive(t, sender, kind, data)
                else:
                    self.trader_receive(recipient, t, kind, data)
        return self.res

    # ---------------------------------------------------------------- exchange
    def ex_send(self, recipient: int, kind: int, data: Any, order_id: int | None = None) -> None:
        """``ExchangeAgent.send_message``: pipeline delay on accept / execute / cancel reports."""
        mid = self.new_msg_id()
        delay = self.pipeline_delay if kind in (K_ACCEPTED, K_EXECUTED, K_CANCELLED) else 0
        self.send(0, recipient, mid, kind, data, delay, order_id)

    def exchange_wakeup(self, t: int) -> None:
        if t >= self.mkt_close:
            mid = self.new_msg_id()  # one MarketClosePriceMsg, delivered to every subscriber
            for agent in self.close_price_subs:
                self.send(0, agent, mid, K_CLOSE_PRICE, None)

    def exchange_receive(self, t: int, sender: int, kind: int, data: Any) -> None:
        self.comp_delay[0] = self.ex_computation_delay
        if t > self.mkt_close:
            if kind == K_QUERY_SPREAD:
                pass  # queries are still answered after the close
            else:
                self.ex_send(sender, K_MKT_CLOSED, None)
                return
        if kind == K_HOURS_REQ:
            self.comp_delay[0] = 0
            self.ex_send(sender, K_HOURS, None)
        elif kind == K_CLOSE_PRICE_REQ:
            self.close_price_subs.append(sender)
        elif kind == K_QUERY_SPREAD:
            bid = (self.bids[0][0], self.level_qty(self.bids[0])) if self.bids else None
            ask = (self.asks[0][0], self.level_qty(self.asks[0])) if self.asks else None
            self.ex_send(sender, K_SPREAD_RESP, (bid, ask, t > self.mkt_close))
        elif kind == K_LIMIT:
            self.handle_limit_order(list(data))
        elif kind == K_CANCEL:
            self.cancel_order(data, quiet=False)

    @staticmethod
    def level_qty(level: list) -> int:
        return sum(o[O_QTY] for o in level[1])

    def log_quotes(self) -> None:
        t = self.current_time
        if self.bids:
            lvl = self.bids[0]
            self.res.quotes.append((t, True, lvl[0], self.level_qty(lvl)))
        if self.asks:
            lvl = self.asks[0]
            self.res.quotes.append((t, False, lvl[0], self.level_qty(lvl)))

    def handle_limit_order(self, order: list, quiet: bool = False) -> None:
        """``OrderBook.handle_limit_order`` (with the opt-in STP patch)."""
        if order[O_QTY] <= 0 or order[O_PRICE] < 0:
            return  # discarded with a warning; no report, no quote log
        stp = self.stp
        while True:
            if stp:
                opp = self.asks if order[O_BID] else self.bids
                if opp and self.is_match(opp[0], order):
                    resting = opp[0][1][0]
                    if resting[O_AGENT] == order[O_AGENT]:
                        if stp == "cancel_oldest" and self.cancel_order(tuple(resting), quiet=quiet):
                            continue
                        if stp != "cancel_oldest":
                            if not quiet:
                                self.ex_send(order[O_AGENT], K_CANCELLED, tuple(order), order[O_ID])
                            break
            matched = self.execute_order(order)
            if matched:
                if order[O_QTY] <= 0:
                    break
            else:
                self.enter_order(list(order))
                if not quiet:
                    self.ex_send(order[O_AGENT], K_ACCEPTED, tuple(order), order[O_ID])
                break
        self.log_quotes()

    @staticmethod
    def is_match(level: list, order: list) -> bool:
        if order[O_BID]:
            return order[O_PRICE] >= level[0]
        return order[O_PRICE] <= level[0]

    def execute_order(self, order: list) -> bool:
        book = self.asks if order[O_BID] else self.bids
        if not book:
            return False
        level = book[0]
        if not self.is_match(level, order):
            return False
        orders = level[1]
        resting = orders[0]
        if order[O_QTY] >= resting[O_QTY]:
            orders.pop(0)
            matched = resting
            if not orders:
                del book[0]
        else:
            matched = list(resting)
            matched[O_QTY] = order[O_QTY]
            resting[O_QTY] -= matched[O_QTY]
        fill_qty = matched[O_QTY]
        fill_price = matched[O_PRICE]
        order[O_QTY] -= fill_qty
        # OrderExecutedMsg(matched) to the resting side, then OrderExecutedMsg(filled) to the aggressor.
        self.ex_send(matched[O_AGENT], K_EXECUTED,
                     (matched[O_ID], matched[O_AGENT], matched[O_BID], fill_price, fill_qty), matched[O_ID])
        self.ex_send(order[O_AGENT], K_EXECUTED,
                     (order[O_ID], order[O_AGENT], order[O_BID], fill_price, fill_qty), order[O_ID])
        return True

    def enter_order(self, order: list) -> None:
        book = self.bids if order[O_BID] else self.asks
        price = order[O_PRICE]
        if order[O_BID]:
            for i, level in enumerate(book):
                if price > level[0]:
                    book.insert(i, [price, [order]])
                    return
                if price == level[0]:
                    level[1].append(order)
                    return
        else:
            for i, level in enumerate(book):
                if price < level[0]:
                    book.insert(i, [price, [order]])
                    return
                if price == level[0]:
                    level[1].append(order)
                    return
        book.append([price, [order]])

    def cancel_order(self, order: tuple, quiet: bool = False) -> bool:
        """``OrderBook.cancel_order``: find the level by price, the order by id."""
        book = self.bids if order[O_BID] else self.asks
        if not book:
            return False
        price, oid = order[O_PRICE], order[O_ID]
        for i, level in enumerate(book):
            if level[0] != price:
                continue
            orders = level[1]
            for j, o in enumerate(orders):
                if o[O_ID] == oid:
                    cancelled = orders.pop(j)
                    if not orders:
                        del book[i]
                    if not quiet:
                        self.ex_send(order[O_AGENT], K_CANCELLED, tuple(cancelled), cancelled[O_ID])
                    return True
        return False

    # ----------------------------------------------------------------- traders
    def tr_send(self, agent: int, kind: int, data: Any, order_id: int | None = None) -> None:
        self.send(agent, 0, self.new_msg_id(), kind, data, 0, order_id)

    def trader_wakeup(self, a: int, t: int) -> None:
        # TradingAgent.wakeup
        if self.first_wake[a]:
            self.first_wake[a] = False
            self.tr_send(a, K_CLOSE_PRICE_REQ, None)
        if not self.mkt_hours_known[a]:
            self.tr_send(a, K_HOURS_REQ, None)
        # ScheduledAgent.wakeup
        if not self.mkt_hours_known[a] or self.mkt_closed[a]:
            return
        self.set_wakeup(a, t + self.agents[a].interval_ns)
        self.tr_send(a, K_QUERY_SPREAD, None)
        self.awaiting_spread[a] = True

    def trader_receive(self, a: int, t: int, kind: int, data: Any) -> None:
        had_hours = self.mkt_hours_known[a]
        rows = self.res.order_rows
        if kind == K_HOURS:
            self.mkt_hours_known[a] = True
        elif kind == K_MKT_CLOSED:
            self.mkt_closed[a] = True
        elif kind == K_EXECUTED:
            oid, agent, is_bid, price, qty = data
            rows.append((t, agent, T_EXEC, is_bid, price, qty, oid))
            oo = self.open_orders[a]
            o = oo.get(oid)
            if o is not None:
                if qty >= o[0]:
                    del oo[oid]
                else:
                    o[0] -= qty
        elif kind == K_ACCEPTED:
            oid, agent, is_bid, price, qty = data
            rows.append((t, agent, T_ACCEPTED, is_bid, price, qty, oid))
        elif kind == K_CANCELLED:
            oid, agent, is_bid, price, qty = data
            rows.append((t, agent, T_CANCELLED, is_bid, price, qty, oid))
            self.open_orders[a].pop(oid, None)
        elif kind == K_SPREAD_RESP:
            bid, ask, closed = data
            if closed:
                self.mkt_closed[a] = True
            self.known_bid[a] = bid
            self.known_ask[a] = ask
        if self.mkt_hours_known[a] and not had_hours:
            self.set_wakeup(a, self.mkt_open + 0)
        # ScheduledAgent.receive_message
        if self.awaiting_spread[a] and kind == K_SPREAD_RESP:
            if not self.mkt_closed[a]:
                self.act(a, t)
            self.awaiting_spread[a] = False

    def place_limit_order(self, a: int, t: int, qty: int, is_bid: bool, price: int) -> None:
        oid = self.order_id
        self.order_id = oid + 1
        self.open_orders[a][oid] = [qty, is_bid, price]
        self.tr_send(a, K_LIMIT, (oid, a, is_bid, price, qty), oid)
        self.res.order_rows.append((t, a, T_SUBMITTED, is_bid, price, qty, oid))

    def act(self, a: int, t: int) -> None:
        spec = self.agents[a]
        kb, ka = self.known_bid[a], self.known_ask[a]
        bid = kb[0] if kb else None
        ask = ka[0] if ka else None
        kind = spec.kind
        rs = spec.random_state
        if kind == NOISE:
            size = int(max(1, round(rs.normal(spec.order_size_mean, spec.order_size_std))))
            buy = bool(rs.randint(0, 2))
            offset = int(rs.randint(0, spec.price_offset_ticks + 1))
            if buy:
                anchor = int(ask) if ask else (int(bid) if bid else spec.reference_price)
                self.place_limit_order(a, t, size, True, anchor + offset)
            else:
                anchor = int(bid) if bid else (int(ask) if ask else spec.reference_price)
                self.place_limit_order(a, t, size, False, anchor - offset)
        elif kind == MM:
            mid = int((bid + ask) // 2) if (bid and ask) else spec.reference_price
            # cancel_all_orders: one CancelOrderMsg per known-open order, insertion order
            for oid, o in self.open_orders[a].items():
                self.tr_send(a, K_CANCEL, (oid, a, o[1], o[2], o[0]), oid)
            half = spec.spread_ticks // 2
            for lvl in range(spec.depth_levels):
                self.place_limit_order(a, t, spec.size_per_level, True, mid - half - lvl)
                self.place_limit_order(a, t, spec.size_per_level, False, mid + half + lvl)
        elif kind == VALUE:
            if bid and ask:
                mid = (int(bid) + int(ask)) / 2.0
            elif bid:
                mid = float(bid)
            elif ask:
                mid = float(ask)
            else:
                return
            fundamental = int(self.observe_price(t, rs, spec.sigma_n))
            size = spec.size
            if mid < fundamental - spec.threshold_ticks and ask:
                self.place_limit_order(a, t, size, True, int(ask))
            elif mid > fundamental + spec.threshold_ticks and bid:
                self.place_limit_order(a, t, size, False, int(bid))
        else:  # MOMENTUM
            if bid and ask:
                mid = (int(bid) + int(ask)) / 2.0
            elif bid:
                mid = float(bid)
            elif ask:
                mid = float(ask)
            else:
                return
            hist = self.mid_history[a]
            hist.append(mid)
            if len(hist) > spec.lookback + 1:
                hist.pop(0)
            if len(hist) <= spec.lookback:
                return
            past = hist[0]
            size = spec.size
            if mid > past + spec.threshold_ticks and ask:
                self.place_limit_order(a, t, size, True, int(ask))
            elif mid < past - spec.threshold_ticks and bid:
                self.place_limit_order(a, t, size, False, int(bid))

    # ------------------------------------------------------------------ oracle
    def compute_fundamental_at_timestamp(self, ts, v_adj, pt, pv) -> int:
        o = self.spec.oracle
        d = ts - pt
        mu = o.r_bar
        gamma = o.kappa
        theta = o.fund_vol
        v = self.o_rs.normal(
            loc=mu + (pv - mu) * (math.exp(-gamma * d)),
            scale=math.sqrt(((theta**2) / (2 * gamma)) * (1 - math.exp(-2 * gamma * d))),
        )
        v += v_adj
        v = max(0, v)
        v = int(round(v))
        for jump in self.o_jumps:
            if not jump.get("_consumed") and ts >= jump["time_ns"]:
                v = max(0, v + int(jump["magnitude"]))
                jump["_consumed"] = True
        self.o_r = (ts, v)
        return v

    def advance_fundamental_value_series(self, current_time: int) -> int:
        o = self.spec.oracle
        pt, pv = self.o_r
        if current_time <= pt:
            return pv
        mst, msv = self.o_mst, self.o_msv
        while mst < current_time:
            v = self.compute_fundamental_at_timestamp(mst, msv, pt, pv)
            pt, pv = mst, v
            mst = pt + int(self.global_rs.exponential(scale=1.0 / o.megashock_lambda_a))
            msv = self.o_rs.normal(loc=o.megashock_mean, scale=math.sqrt(o.megashock_var))
            msv = msv if self.o_rs.randint(2) == 0 else -msv
            self.o_mst, self.o_msv = mst, msv
        return self.compute_fundamental_at_timestamp(current_time, 0, pt, pv)

    def observe_price(self, current_time: int, rs: np.random.RandomState, sigma_n: float) -> int:
        if current_time >= self.mkt_close:
            r_t = self.advance_fundamental_value_series(self.mkt_close - 1)
        else:
            r_t = self.advance_fundamental_value_series(current_time)
        if sigma_n == 0:
            return r_t
        return int(round(rs.normal(loc=r_t, scale=math.sqrt(sigma_n))))
