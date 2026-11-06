"""Calibration-derived one-way timing intervals (stable-clock assumption).

A signed point estimate is not an exact duration.  Admission uses the upper
endpoint of the nonnegative feasible interval; raw estimates remain available
for diagnostics.  A wholly negative interval is inconsistent and must fail.
"""


def duration_bounds(raw_ns: int, uncertainty_ns: int) -> dict:
    for name, value in (("raw_ns", raw_ns), ("uncertainty_ns", uncertainty_ns)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(name + " must be an integer")
    if uncertainty_ns < 0:
        raise ValueError("uncertainty_ns must be nonnegative")
    lower = raw_ns - uncertainty_ns
    upper = raw_ns + uncertainty_ns
    if upper < 0:
        raise ValueError("one-way timing interval is entirely negative")
    return {
        "signed_estimate_ns": raw_ns,
        "uncertainty_ns": uncertainty_ns,
        "lower_bound_ns": max(0, lower),
        "upper_bound_ns": upper,
        "point_estimate_negative": raw_ns < 0,
        "scheduler_seconds": upper / 1_000_000_000,
        "policy": "calibration_upper_bound_v1",
    }
