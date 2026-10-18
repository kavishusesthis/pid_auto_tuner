
import numpy as np


def calculate_metrics(result, tolerance=0.02):
    """
    Calculate performance metrics from simulate_pid() output.

    Settling time is the first time after which the response
    remains within the specified tolerance band until the end.
    """
    required = {"time_s", "target_rpm", "speed_rpm", "error_rpm"}

    if not required.issubset(result.columns):
        raise ValueError(f"Missing columns: {required - set(result.columns)}")

    if result.empty or not 0 < tolerance < 1:
        raise ValueError("Data must be non-empty and tolerance must be between 0 and 1.")

    time = result["time_s"].to_numpy(dtype=float)
    speed = result["speed_rpm"].to_numpy(dtype=float)
    target = result["target_rpm"].to_numpy(dtype=float)

    if not (
        np.isfinite(time).all()
        and np.isfinite(speed).all()
        and np.isfinite(target).all()
    ):
        raise ValueError("Input data contains non-finite values.")

    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("Time must be strictly increasing.")

    target_value = float(target[-1])

    if target_value <= 0 or not np.allclose(target, target_value):
        raise ValueError("Target speed must be constant and positive.")

    error = target_value - speed
    band = tolerance * target_value

    overshoot = max(
        0.0,
        (float(np.max(speed)) - target_value) / target_value * 100.0,
    )

    # Rise time: first crossing of 90% minus first crossing of 10%.
    t10_indices = np.flatnonzero(speed >= 0.10 * target_value)
    t90_indices = np.flatnonzero(speed >= 0.90 * target_value)

    rise_time = (
        float(time[t90_indices[0]] - time[t10_indices[0]])
        if len(t10_indices) and len(t90_indices)
        else float("nan")
    )

    # Settling time: last time outside the tolerance band.
    outside = np.flatnonzero(np.abs(error) > band)

    if len(outside) == 0:
        settling_time = float(time[0])
    elif outside[-1] == len(time) - 1:
        settling_time = float("nan")
    else:
        settling_time = float(time[outside[-1] + 1])

    return {
        "target_rpm": target_value,
        "final_speed_rpm": float(speed[-1]),
        "final_error_rpm": float(error[-1]),
        "steady_state_error_percent": float(
            abs(error[-1]) / target_value * 100.0
        ),
        "overshoot_percent": overshoot,
        "rise_time_s": rise_time,
        "settling_time_s": settling_time,
        "IAE_rpm_s": float(np.trapezoid(np.abs(error), time)),
    }


if __name__ == "__main__":
    from pid_controller import simulate_pid

    result = simulate_pid(
        target_rpm=100.0,
        duration=10.0,
        kp=0.03,
        ki=0.006,
        kd=0.0005,
    )

    metrics = calculate_metrics(result)

    print("\nPID Performance Metrics")
    print("-" * 30)

    for name, value in metrics.items():
        if isinstance(value, float) and np.isnan(value):
            print(f"{name}: Not reached within simulation")
        else:
            print(f"{name}: {value:.3f}")
