"""One registry for continuous L1 policies."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LambdaResolution:
    policy: str
    value: float
    n: int
    d: int
    requested: float | None
    multiplier: float


def resolve_lambda1(
    policy,
    *,
    n,
    d,
    fixed=0.03,
    multiplier=1.0,
    reference=0.03,
    reference_n=1000,
    reference_d=50,
):
    if n <= 0 or d <= 1:
        raise ValueError("n must be positive and d must exceed one")
    if policy in {"fixed", "fixed_0.03"}:
        value, canonical, requested = float(fixed), "fixed", float(fixed)
    elif policy in {"sqrt_log_d_over_n", "sqrt_logd_over_n"}:
        scale = math.sqrt(
            (math.log(d) / n) / (math.log(reference_d) / reference_n))
        value = float(multiplier * reference * scale)
        canonical, requested = "sqrt_log_d_over_n", None
    elif str(policy).startswith("sqrt_c"):
        coefficient = float(str(policy).removeprefix("sqrt_c"))
        value = coefficient * math.sqrt(math.log(d) / n)
        canonical, requested, multiplier = (
            "sqrt_log_d_over_n", None, coefficient)
    else:
        raise ValueError(f"unknown lambda policy: {policy}")
    return LambdaResolution(
        canonical, value, int(n), int(d), requested, float(multiplier))
