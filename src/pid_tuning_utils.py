"""Shared simulation-based scoring and local refinement utilities."""
import math
import random

from pid_controller import simulate_pid
from performance_metrics import calculate_metrics


def response_metrics(motor, gains, duration=6.0, dt=0.005):
    kp, ki, kd = gains
    params = {key: motor[key] for key in ("J", "b", "Kt", "Ke", "R", "L")}
    response = simulate_pid(
        target_rpm=float(motor["target_rpm"]),
        duration=float(duration),
        dt=float(dt),
        kp=max(0.0, float(kp)),
        ki=max(0.0, float(ki)),
        kd=max(0.0, float(kd)),
        max_voltage=float(motor["max_voltage"]),
        params=params,
        load_torque=float(motor["load_torque"]),
    )
    return calculate_metrics(response)


def score_metrics(metrics, duration=6.0):
    settling = metrics["settling_time_s"]
    if not math.isfinite(settling):
        settling = duration * 2.0
    return (
        10.0 * min(metrics["steady_state_error_percent"], 100.0)
        + 3.0 * min(metrics["overshoot_percent"], 100.0)
        + 0.5 * settling
        + 0.01 * min(metrics["IAE_rpm_s"], 100000.0)
    )


def refine_gains(motor, initial_gains, seed=42, duration=6.0, dt=0.005, n_candidates=80):
    """Search nearby and broad candidate gains around an ML recommendation.

    The model's gains are the starting point; simulation selects the final
    candidate. This makes the system a hybrid ML + simulation tuner.
    """
    rng = random.Random(seed)
    kp0, ki0, kd0 = [max(0.0, float(x)) for x in initial_gains]
    candidates = [
        (kp0, ki0, kd0),
        (0.01, 0.005, 0.0005),
        (0.03, 0.003, 0.001),
        (0.05, 0.005, 0.0),
        (0.07, 0.008, 0.001),
    ]

    factors = (0.5, 0.75, 1.0, 1.25, 1.5)
    for fk in factors:
        for fi in factors:
            for fd in (0.5, 1.0, 1.5):
                candidates.append((
                    min(0.15, max(0.001, kp0 * fk)),
                    min(0.03, max(0.0, ki0 * fi)),
                    min(0.01, max(0.0, kd0 * fd)),
                ))

    # Include broad candidates so zero or tiny predicted gains can escape a
    # narrow multiplicative neighborhood. These ranges match data generation.
    for _ in range(n_candidates):
        candidates.append((
            rng.uniform(0.005, 0.10),
            rng.uniform(0.0, 0.02),
            rng.uniform(0.0, 0.005),
        ))

    best = None
    best_metrics = None
    best_score = float("inf")
    seen = set()
    for gains in candidates:
        gains = tuple(round(float(x), 8) for x in gains)
        if gains in seen:
            continue
        seen.add(gains)
        try:
            metrics = response_metrics(motor, gains, duration=duration, dt=dt)
            candidate_score = score_metrics(metrics, duration=duration)
        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            continue
        if math.isfinite(candidate_score) and candidate_score < best_score:
            best, best_metrics, best_score = gains, metrics, candidate_score

    if best is None:
        raise RuntimeError("No valid gain candidate could be simulated.")
    return best, best_metrics, best_score
