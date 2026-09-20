"""Time-constant random-parameter equilibrium Gagge benchmark generator.

The public pythermalcomfort API is used as the dependency reference.  The
equilibrium adapter below ports the state-rate and TSENS equations needed for
continuation beyond the package's fixed 60-minute call.  No random quantity is
drawn inside the integration loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import math
from typing import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr


PACKAGE = "pythermalcomfort"
PACKAGE_VERSION = importlib.metadata.version(PACKAGE)


@dataclass(frozen=True)
class ThermalConfig:
    parameter_variance_scale: float = 1.0
    parameter_variance_scales: dict[str, float] | None = None
    sigma_heat: float = 2.0
    sigma_sensation: float = 0.25
    tol_core: float = 1e-4
    tol_skin: float = 1e-4
    consecutive_steps: int = 20
    max_minutes: int = 2000
    body_surface_area: float = 1.8258
    wme: float = 0.0
    position: str = "standing"
    calculate_ce: bool = False
    max_skin_blood_flow: float = 90.0
    max_sweating: float = 500.0


ENVIRONMENT = (
    ("tdb", 25.0, 3.0, 18.0, 32.0),
    ("tr", 25.0, 3.0, 18.0, 32.0),
    ("v", 0.15, 0.06, 0.05, 0.8),
    ("rh", 50.0, 10.0, 20.0, 80.0),
    ("p_atm", 101325.0, 2000.0, 95000.0, 105000.0),
)
PERSON = (
    ("met", 1.2, 0.20, 0.8, 2.0),
    ("clo", 0.5, 0.15, 0.1, 1.2),
    ("max_skin_blood_flow", 90.0, 15.0, 30.0, 150.0),
    ("max_sweating", 500.0, 75.0, 250.0, 750.0),
)
ROOT_NAMES = tuple(item[0] for item in ENVIRONMENT + PERSON)
OBSERVED_NAMES = ROOT_NAMES + ("T_skin", "T_sens")


def _bounded(z: np.ndarray, nominal: float, sd: float,
             lower: float, upper: float) -> np.ndarray:
    """Smooth bounded Gaussian transform calibrated at the nominal point."""
    q = np.clip((nominal - lower) / (upper - lower), 1e-6, 1 - 1e-6)
    logit = math.log(q / (1.0 - q))
    slope = sd / max((upper - lower) * q * (1.0 - q), 1e-12)
    return lower + (upper - lower) / (1.0 + np.exp(-(logit + slope * z)))


def _parameters(z: np.ndarray, variance_scale: float = 1.0,
                variance_scales: dict[str, float] | None = None
                ) -> dict[str, float]:
    if variance_scale <= 0:
        raise ValueError("parameter variance scale must be positive")
    values = {}
    for value, spec in zip(z, ENVIRONMENT + PERSON):
        name, nominal, sd, lower, upper = spec
        scale = (variance_scales.get(name, variance_scale)
                 if variance_scales is not None else variance_scale)
        if scale <= 0:
            raise ValueError(f"variance scale for {name} must be positive")
        values[name] = float(_bounded(
            np.asarray([value]), nominal,
            sd * scale,
            lower, upper)[0])
    return values


def _hsic(x: np.ndarray, y: np.ndarray) -> float:
    """Small biased RBF-HSIC diagnostic, not used by the discovery methods."""
    if len(x) > 500:
        x, y = x[:500], y[:500]
    def kernel(value):
        distance = (value[:, None] - value[None, :]) ** 2
        positive = distance[distance > 0]
        bandwidth = math.sqrt(float(np.median(positive))) if positive.size else 1.0
        return np.exp(-distance / max(2.0 * bandwidth**2, 1e-12))
    n = len(x)
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(kernel(x) @ H @ kernel(y) @ H) / max((n - 1) ** 2, 1))


def _sigmoid_piecewise_sensation(t_body: float, met: float, wme: float,
                                 v: float, clo: float) -> float:
    met_factor = 58.15
    rm = (met - wme) * 58.2
    speed = max(v, 0.1)
    w_max = (0.38 * speed ** -0.29 if clo <= 0
             else 0.59 * speed ** -0.08)
    low = (0.194 / met_factor) * rm + 36.301
    high = (0.347 / met_factor) * rm + 36.669
    if t_body < low:
        return 0.4685 * (t_body - low)
    if t_body < high:
        return w_max * 4.7 * (t_body - low) / (high - low)
    return w_max * 4.7 + 0.4685 * (t_body - high)


def _rates(core: float, skin: float, p: dict[str, float],
           heat_offset: float, config: ThermalConfig) -> dict[str, float]:
    """One deterministic Gagge state update, including persistent heat input."""
    ta, tr, v, rh, patm = (p["tdb"], p["tr"], p["v"], p["rh"], p["p_atm"])
    met, clo = p["met"], p["clo"]
    pressure = patm / 101325.0
    vapor = rh * math.exp(18.6686 - 4030.183 / (ta + 235.0)) / 100.0
    speed = max(v, 0.1)
    met_factor = 58.2
    rm = (met - config.wme) * met_factor
    m = met * met_factor
    r_clo = 0.155 * clo
    f_a_cl = 1.0 + 0.15 * clo
    lr = 2.2 / pressure
    i_cl = 1.0 if clo <= 0 else 0.45
    w_max = (0.38 * speed ** -0.29 if clo <= 0
             else 0.59 * speed ** -0.08)
    h_cc = max(3.0 * pressure ** 0.53,
               8.600001 * (speed * pressure) ** 0.53)
    if not config.calculate_ce and met > 0.85:
        h_cc = max(h_cc, 5.66 * (met - 0.85) ** 0.39)
    h_r = 4.7
    t_cl = (h_r * tr + h_cc * ta) / (h_r + h_cc)
    for _ in range(30):
        h_r = 4.0 * 0.95 * 5.6697e-8 * ((t_cl + tr) / 2.0 + 273.15) ** 3 * 0.73
        h_t = h_r + h_cc
        r_a = 1.0 / (f_a_cl * h_t)
        t_op = (h_r * tr + h_cc * ta) / h_t
        new_t_cl = (r_a * skin + r_clo * t_op) / (r_a + r_clo)
        if abs(new_t_cl - t_cl) <= 0.01:
            t_cl = new_t_cl
            break
        t_cl = new_t_cl
    q_sensible = (skin - t_op) / (r_a + r_clo)
    skin_signal = skin - 33.7
    cold_skin = max(-skin_signal, 0.0)
    core_signal = core - 36.8
    warm_core = max(core_signal, 0.0)
    m_bl = (6.3 + 120.0 * warm_core) / (1.0 + 0.5 * cold_skin)
    m_bl = min(max(m_bl, 0.5), p["max_skin_blood_flow"])
    alpha = 0.0417737 + 0.7451833 / (m_bl + 0.585417)
    t_body = alpha * skin + (1.0 - alpha) * core
    warm_body = max(t_body - (0.1 * 33.7 + 0.9 * 36.8), 0.0)
    m_rsw = min(170.0 * warm_body * math.exp(max(skin_signal, 0.0) / 10.7),
                p["max_sweating"])
    e_rsw = 0.68 * m_rsw
    r_ea = 1.0 / (lr * f_a_cl * h_cc)
    r_ecl = r_clo / (lr * i_cl)
    e_max = (math.exp(18.6686 - 4030.183 / (skin + 235.0)) - vapor) / (r_ea + r_ecl)
    e_max = 1e-6 if abs(e_max) < 1e-6 else e_max
    p_rsw = e_rsw / e_max
    wetted = 0.06 + 0.94 * p_rsw
    e_diff = wetted * e_max - e_rsw
    if wetted > w_max:
        wetted = w_max
        p_rsw = w_max / 0.94
        e_rsw = p_rsw * e_max
        e_diff = 0.06 * (1.0 - p_rsw) * e_max
    if e_max < 0:
        e_diff, e_rsw, wetted = 0.0, 0.0, w_max
    e_skin = e_rsw + e_diff
    shivering = 19.4 * cold_skin * max(36.8 - core, 0.0)
    metabolic = rm + shivering
    heat_core_skin = (core - skin) * (5.28 + 1.163 * m_bl)
    q_res = 0.0023 * m * (44.0 - vapor)
    c_res = 0.0014 * m * (34.0 - ta)
    core_storage = metabolic - heat_core_skin - q_res - c_res - config.wme
    skin_storage = heat_core_skin - q_sensible - e_skin + heat_offset
    d_core = core_storage * config.body_surface_area / (.97 * (1.0 - alpha) * 70.0 * 60.0)
    d_skin = skin_storage * config.body_surface_area / (.97 * alpha * 70.0 * 60.0)
    return {"d_core": d_core, "d_skin": d_skin, "t_body": t_body,
            "t_sens": _sigmoid_piecewise_sensation(
                t_body, met, config.wme, v, clo),
            "m_bl": m_bl, "w_max": w_max, "e_max": e_max}


def equilibrate(p: dict[str, float], u_t: float, config: ThermalConfig,
                *, initial=(36.8, 33.7), trace: list | None = None) -> dict:
    core, skin = map(float, initial)
    stable = 0
    for minute in range(1, config.max_minutes + 1):
        state = _rates(core, skin, p, config.sigma_heat * u_t, config)
        if trace is not None:
            trace.append({"minute": minute, "E": tuple(p[n] for n in ROOT_NAMES[:5]),
                          "P": tuple(p[n] for n in ROOT_NAMES[5:]), "U_T": float(u_t)})
        core += state["d_core"]
        skin += state["d_skin"]
        if abs(state["d_core"]) < config.tol_core and abs(state["d_skin"]) < config.tol_skin:
            stable += 1
            if stable >= config.consecutive_steps:
                final = _rates(core, skin, p, config.sigma_heat * u_t, config)
                return {"core": core, "skin": skin, "residual_core": abs(final["d_core"]),
                        "residual_skin": abs(final["d_skin"]), "minutes": minute,
                        "t_body": final["t_body"], "t_sens": final["t_sens"]}
        else:
            stable = 0
    raise RuntimeError(f"Gagge equilibrium did not converge for parameters {p}")


def _core_at_clamped_skin(p: dict[str, float], skin: float,
                          config: ThermalConfig) -> float:
    lo, hi = 25.0, 45.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        rate = _rates(mid, skin, p, 0.0, config)["d_core"]
        if rate > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clamped_sensation(p: dict[str, float], skin: float,
                      u_y: float, config: ThermalConfig) -> float:
    core = _core_at_clamped_skin(p, skin, config)
    state = _rates(core, skin, p, 0.0, config)
    return float(state["t_sens"] + config.sigma_sensation * u_y)


def _oracle_edges(config: ThermalConfig) -> tuple[np.ndarray, dict]:
    """Structural support from finite-difference dependence checks."""
    d = len(OBSERVED_NAMES)
    truth = np.zeros((d, d), dtype=np.uint8)
    skin_index = len(ROOT_NAMES)
    sensation_index = skin_index + 1
    base = _parameters(
        np.zeros(len(ROOT_NAMES)), config.parameter_variance_scale,
        config.parameter_variance_scales)
    central = equilibrate(base, 0.0, config)
    audit = {}
    for root_index, root_name in enumerate(ROOT_NAMES):
        skin_effects, direct_effects = [], []
        for shift in (-1.0, 1.0):
            latent = np.zeros(len(ROOT_NAMES))
            latent[root_index] = shift
            perturbed = _parameters(
                latent, config.parameter_variance_scale,
                config.parameter_variance_scales)
            outcome = equilibrate(perturbed, 0.0, config)
            skin_effects.append(abs(outcome["skin"] - central["skin"]))
            direct_effects.append(abs(
                clamped_sensation(perturbed, central["skin"], 0.0, config)
                - clamped_sensation(base, central["skin"], 0.0, config)))
        audit[root_name] = {
            "skin_effect_max": float(max(skin_effects)),
            "controlled_sensation_effect_max": float(max(direct_effects)),
        }
        if max(skin_effects) > 1e-7:
            truth[root_index, skin_index] = 1
        if max(direct_effects) > 1e-7:
            truth[root_index, sensation_index] = 1
    skin_effects = [abs(
        clamped_sensation(base, central["skin"] + delta, 0.0, config)
        - clamped_sensation(base, central["skin"], 0.0, config))
                    for delta in (-0.5, 0.5)]
    audit["T_skin"] = {
        "controlled_sensation_effect_max": float(max(skin_effects))}
    if max(skin_effects) <= 1e-7:
        raise RuntimeError("T_skin does not affect controlled sensation")
    truth[skin_index, sensation_index] = 1
    return truth, audit


def generate(seed: int, n: int, *, config: ThermalConfig | None = None,
             diagnostics: bool = False):
    config = config or ThermalConfig()
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, len(ROOT_NAMES) + 2))
    params = [_parameters(
        row[:len(ROOT_NAMES)], config.parameter_variance_scale,
        config.parameter_variance_scales) for row in latent]
    rows = np.empty((n, len(OBSERVED_NAMES)), dtype=np.float64)
    traces = []
    convergence = []
    for i, p in enumerate(params):
        trace = [] if diagnostics else None
        result = equilibrate(p, float(latent[i, -2]), config, trace=trace)
        rows[i, :len(ROOT_NAMES)] = [p[name] for name in ROOT_NAMES]
        rows[i, -2] = result["skin"]
        rows[i, -1] = result["t_sens"] + config.sigma_sensation * latent[i, -1]
        convergence.append({"minutes": result["minutes"],
                            "residual_core": result["residual_core"],
                            "residual_skin": result["residual_skin"]})
        if diagnostics:
            traces.append(trace)
    truth, dependency_audit = _oracle_edges(config)
    pairs = [(i, j) for i in range(5) for j in range(5, 9)]
    dependence = []
    for left, right in pairs:
        x, y = rows[:, left], rows[:, right]
        dependence.append({
            "left": OBSERVED_NAMES[left], "right": OBSERVED_NAMES[right],
            "pearson": float(pearsonr(x, y).statistic),
            "spearman": float(spearmanr(x, y).statistic),
            "hsic": _hsic(x, y),
        })
    metadata = {"data_model": "nonlinear_piecewise_equilibrium_scm",
                "discovery_score": "method_default",
                "score_misspecification": True,
                "thermal_package": PACKAGE,
                "thermal_package_version": PACKAGE_VERSION,
                "config": asdict(config), "column_names": OBSERVED_NAMES,
                "root_names": ROOT_NAMES, "oracle_pairs": pairs,
                "dependency_audit": dependency_audit,
                "cross_block_dependence": dependence,
                "convergence": convergence,
                "max_equilibrium_residual": max(
                    max(item["residual_core"], item["residual_skin"])
                    for item in convergence),
                "activation_frequency": {
                    "max_skin_blood_flow": float(np.mean(
                        rows[:, 7] < config.max_skin_blood_flow - 1e-8)),
                    "max_sweating": float(np.mean(
                        rows[:, 8] < config.max_sweating - 1e-8)),
                }}
    if diagnostics:
        metadata["exogenous_latents"] = latent.tolist()
        metadata["integration_traces"] = traces
    return rows, truth, pairs, metadata
