import numpy as np


def calculate_metrics(result, tolerance=0.02):
    """Calculate speed-control metrics from simulate_pid() output.

    Settling time is the earliest sample after which all remaining samples
    stay within +/- tolerance of the constant target. NaN means the response
    never settles within the simulation horizon.
    """
    required = {"time_s", "target_rpm", "speed_rpm", "error_rpm"}
    if not required.issubset(result.columns):
        raise ValueError(f"Missing columns: {required - set(result.columns)}")
    if result.empty or not 0 < tolerance < 1:
        raise ValueError("Data must be non-empty and tolerance must be between 0 and 1.")

    time = result["time_s"].to_numpy(dtype=float)
    speed = result["speed_rpm"].to_numpy(dtype=float)
    target = result["target_rpm"].to_numpy(dtype=float)
    if not (np.isfinite(time).all() and np.isfinite(speed).all() and np.isfinite(target).all()):
        raise ValueError("Input data contains non-finite values.")
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("Time must be strictly increasing.")

    target_value = float(target[-1])
    if target_value <= 0 or not np.allclose(target, target_value):
        raise ValueError("Target speed must be constant and positive.")

    error = target_value - speed
    band = tolerance * abs(target_value)
    overshoot = max(0.0, (float(np.max(speed)) - target_value) / target_value * 100.0)

    t10 = np.flatnonzero(speed >= 0.10 * target_value)
    t90 = np.flatnonzero(speed >= 0.90 * target_value)
    rise_time = (
        float(time[t90[0]] - time[t10[0]])
        if len(t10) and len(t90) else float("nan")
    )

    # suffix_inside[i] is true only if sample i and every later sample
    # are inside the settling band. This avoids mistaking a temporary
    # crossing of the band for a settled response.
    inside = np.abs(error) <= band
    suffix_inside = np.logical_and.accumulate(inside[::-1])[::-1]
    settled_indices = np.flatnonzero(suffix_inside)
    settling_time = float(time[settled_indices[0]]) if len(settled_indices) else float("nan")

    return {
        "target_rpm": target_value,
        "final_speed_rpm": float(speed[-1]),
        "final_error_rpm": float(error[-1]),
        "steady_state_error_percent": float(abs(error[-1]) / target_value * 100.0),
        "overshoot_percent": float(overshoot),
        "rise_time_s": rise_time,
        "settling_time_s": settling_time,
        "IAE_rpm_s": float(np.trapezoid(np.abs(error), time)),
    }


if __name__ == "__main__":
    from pid_controller import simulate_pid

    result = simulate_pid(target_rpm=100.0, duration=10.0, kp=0.03, ki=0.006, kd=0.0005)
    for name, value in calculate_metrics(result).items():
        if isinstance(value, float) and np.isnan(value):
            print(f"{name}: Not reached within simulation")
        else:
            print(f"{name}: {value:.3f}")
