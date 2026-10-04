"""Scenario → engine parameters, reproducing ``baselines/abides_fork/config.py`` exactly.

Everything that consumes randomness before the event loop starts is done here, with NumPy, in the
same order as the baseline, so the seeded streams the engine receives are bit-identical:

    np.random.seed(seed)                       # the global stream
    oracle RandomState      <- global draw 1
    np.random.exponential   <- global (oracle constructor: first megashock time)
    oracle.normal, oracle.randint(2)           # first megashock value / sign
    exchange RandomState    <- global draw
    agent RandomStates      <- one global draw each, agent_configs order x count
    latency RandomState     <- global draw (or the line-distance fallback model)
    kernel RandomState      <- global draw (unused when a latency model exists)

After this, the global stream is used only by the oracle's megashock arrivals, so it is handed to
the engine too (``global_rs``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DATE_NS = 1612483200000000000  # pd.to_datetime("20210205").value
OPEN_OFFSET_NS = 34200000000000  # str_to_ns("09:30:00")
ORACLE_CLOSE_OFFSET_NS = 57600000000000  # str_to_ns("16:00:00")
ONE_SECOND_NS = 1000000000
DEFAULT_COMPUTATION_DELAY = 50

AGENT_TYPES = ("NoiseTrader", "MarketMaker", "ValueTrader", "MomentumTrader")
NOISE, MM, VALUE, MOMENTUM = 0, 1, 2, 3

LAT_DETERMINISTIC, LAT_LOGNORMAL, LAT_UNIFORM, LAT_PARETO, LAT_MATRIX = 0, 1, 2, 3, 4


def _draw_random_state() -> np.random.RandomState:
    return np.random.RandomState(seed=np.random.randint(low=0, high=2**32, dtype="uint64"))


def _interval_ns(params: dict[str, Any]) -> int:
    if "rebalance_interval_ns" in params:
        return int(params["rebalance_interval_ns"])
    hz = float(params.get("arrival_rate_hz", 1.0))
    return int(1e9 / hz) if hz > 0 else int(1e9)


@dataclass
class AgentSpec:
    id: int
    kind: int
    random_state: np.random.RandomState
    interval_ns: int
    # NoiseTrader
    order_size_mean: float = 0.0
    order_size_std: float = 0.0
    price_offset_ticks: int = 0
    reference_price: int = 0
    # MarketMaker
    spread_ticks: int = 0
    depth_levels: int = 0
    size_per_level: int = 0
    # ValueTrader / MomentumTrader
    threshold_ticks: int = 0
    sigma_n: float = 1000.0
    lookback: int = 0
    size: int = 0  # int(max(1, round(order_size_mean))) for value / momentum


@dataclass
class OracleSpec:
    r_bar: int
    kappa: float
    fund_vol: float
    megashock_lambda_a: float
    megashock_mean: float
    megashock_var: int
    random_state: np.random.RandomState
    first_megashock_time: Any  # np.float64 (mkt_open + exponential draw)
    first_megashock_value: Any  # float (sign applied)
    scheduled_jumps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LatencySpec:
    model: int
    random_state: np.random.RandomState
    mean_ns: float = 0.0
    sigma: float = 0.0
    min_ns: float = 0.0
    max_ns: float = 1e12
    alpha: float = 1.5
    mu: float = 0.0
    matrix: Any = None  # fallback line-distance model: int64 [n, n]


@dataclass
class EngineSpec:
    scenario_id: str
    seed: int
    mkt_open: int
    mkt_close: int
    stop_time: int
    start_time: int
    stp_policy: str | None
    pipeline_delay: int
    exchange_computation_delay: int
    oracle: OracleSpec
    agents: list[AgentSpec]
    latency: LatencySpec
    global_rs: np.random.RandomState
    n_agents: int


def _line_distance_latency(n_agents: int) -> LatencySpec:
    """``abides_markets.utils.generate_latency_model(n)`` (deterministic line-distance model)."""
    rs = np.random.RandomState(seed=np.random.randint(low=0, high=2**32))
    x = rs.uniform(low=0.0, high=3866660, size=n_agents).reshape((n_agents, 1))
    # scipy squareform(pdist(x, "euclidean")) for 1-D points: sqrt((xi - xj)**2), zero diagonal.
    d = np.sqrt((x - x.T) ** 2)
    np.fill_diagonal(d, 0.0)
    lat = (d / 299792458e-9).astype(int)
    return LatencySpec(model=LAT_MATRIX, random_state=rs, matrix=lat.astype(np.int64))


def build_spec(scenario: dict[str, Any], seed: int | None = None) -> EngineSpec:
    seed = int(scenario["seed"] if seed is None else seed)
    np.random.seed(seed)

    exchange_cfg = scenario["exchange_config"]
    proto = bool(exchange_cfg.get("protocol_enforcement", False))
    stp_policy = str(exchange_cfg["stp_policy"]) if proto and exchange_cfg.get("stp_policy") else None
    ack_delay = int(exchange_cfg.get("ack_delay_ns", 0)) if proto else 0
    compute_delay = int(exchange_cfg.get("compute_delay_ns", 0)) if proto else 0
    oracle_params = scenario["oracle_config"].get("params", {})
    reference_price = int(oracle_params.get("initial_price", 100_000))
    horizon_ns = int(scenario["horizon_ns"])

    mkt_open = DATE_NS + OPEN_OFFSET_NS
    mkt_close = mkt_open + horizon_ns

    kappa_per_s = float(oracle_params.get("kappa", 0.0))
    kappa = kappa_per_s / 1e9 if kappa_per_s > 0 else 1.67e-16
    rate_per_s = float(oracle_params.get("jump_intensity", 0.0))
    lam = rate_per_s / 1e9 if rate_per_s > 0 else 2.77778e-18
    ms_mean = float(oracle_params.get("jump_sigma", 0.0)) or 1000.0
    ms_var = 50_000

    oracle_rs = _draw_random_state()
    jumps = (
        [
            {
                "time_ns": mkt_open + int(oracle_params["scheduled_jump"]["time_ns"]),
                "magnitude": int(oracle_params["scheduled_jump"]["magnitude"]),
                "_consumed": False,
            }
        ]
        if oracle_params.get("scheduled_jump")
        else []
    )
    # SparseMeanRevertingOracle.__init__
    ms_time_delta = np.random.exponential(scale=1.0 / lam)
    mst = mkt_open + ms_time_delta
    msv = oracle_rs.normal(loc=ms_mean, scale=math.sqrt(ms_var))
    msv = msv if oracle_rs.randint(2) == 0 else -msv
    oracle = OracleSpec(
        r_bar=reference_price,
        kappa=kappa,
        fund_vol=float(oracle_params.get("sigma", 5e-5)),
        megashock_lambda_a=lam,
        megashock_mean=ms_mean,
        megashock_var=ms_var,
        random_state=oracle_rs,
        first_megashock_time=mst,
        first_megashock_value=msv,
        scheduled_jumps=jumps,
    )

    _exchange_rs = _draw_random_state()  # the exchange never draws from it; consumed for order

    agents: list[AgentSpec] = []
    next_id = 1
    for agent_cfg in scenario["agent_configs"]:
        agent_type = str(agent_cfg["agent_type"])
        if agent_type not in AGENT_TYPES:
            raise KeyError(f"unsupported agent_type: {agent_type!r}")
        params = agent_cfg.get("params", {})
        interval = _interval_ns(params)
        for _ in range(int(agent_cfg["count"])):
            rs = _draw_random_state()
            spec = AgentSpec(id=next_id, kind=AGENT_TYPES.index(agent_type), random_state=rs,
                             interval_ns=int(max(1, interval)))
            if agent_type == "NoiseTrader":
                spec.order_size_mean = float(params.get("order_size_mean", 10))
                spec.order_size_std = float(params.get("order_size_std", 2))
                spec.price_offset_ticks = int(params.get("price_offset_ticks", 5))
                spec.reference_price = int(reference_price)
            elif agent_type == "MarketMaker":
                spec.spread_ticks = int(max(2, params.get("spread_ticks", 2)))
                spec.depth_levels = int(max(1, params.get("depth_levels", 3)))
                spec.size_per_level = int(max(1, params.get("size_per_level", 10)))
                spec.reference_price = int(reference_price)
            elif agent_type == "ValueTrader":
                spec.order_size_mean = float(params.get("order_size_mean", 25))
                spec.threshold_ticks = int(params.get("threshold_ticks", 2))
                spec.sigma_n = 1000.0
                spec.size = int(max(1, round(spec.order_size_mean)))
            else:  # MomentumTrader
                spec.order_size_mean = float(params.get("order_size_mean", 15))
                spec.threshold_ticks = int(params.get("threshold_ticks", 2))
                spec.lookback = int(max(1, params.get("lookback", 5)))
                spec.size = int(max(1, round(spec.order_size_mean)))
            agents.append(spec)
            next_id += 1

    n_agents = next_id
    latency_cfg = scenario.get("latency_config")
    if latency_cfg:
        lp = latency_cfg.get("params", {})
        model = str(latency_cfg.get("model", "deterministic"))
        mean_ns = float(lp.get("mean_ns", 0.0))
        latency = LatencySpec(
            model={"log_normal": LAT_LOGNORMAL, "uniform": LAT_UNIFORM, "pareto": LAT_PARETO}.get(
                model, LAT_DETERMINISTIC
            ),
            random_state=_draw_random_state(),
            mean_ns=mean_ns,
            sigma=float(lp.get("sigma", 0.0)),
            min_ns=float(lp.get("min_ns", 0.0)),
            max_ns=float(lp.get("max_ns", 1e12)),
            alpha=float(lp.get("alpha", 1.5)),
            mu=float(np.log(mean_ns)) if mean_ns > 0 else 0.0,
        )
    else:
        latency = _line_distance_latency(n_agents)
    _kernel_rs = _draw_random_state()  # consumed for order; abides.run never uses it

    return EngineSpec(
        scenario_id=str(scenario["scenario_id"]),
        seed=seed,
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        stop_time=mkt_close + ONE_SECOND_NS,
        start_time=DATE_NS,
        stp_policy=stp_policy,
        pipeline_delay=ack_delay,
        exchange_computation_delay=compute_delay,
        oracle=oracle,
        agents=agents,
        latency=latency,
        global_rs=np.random.mtrand._rand,
        n_agents=n_agents,
    )
