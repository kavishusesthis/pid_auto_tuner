
import numpy as np
import pandas as pd

from simulate_motor import DEFAULT_PARAMS


def simulate_pid(
    target_rpm=100.0,
    duration=10.0,
    dt=0.001,
    kp=0.01,
    ki=0.005,
    kd=0.0005,
    max_voltage=12.0,
    params=None,
    load_torque=0.0,
):
    """Simulate a DC motor with a saturated, anti-windup PID controller."""

    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive.")
    if target_rpm < 0 or max_voltage <= 0 or load_torque < 0:
        raise ValueError("Invalid target speed, voltage limit, or load.")
    if min(kp, ki, kd) < 0:
        raise ValueError("PID gains must be non-negative.")

    motor = DEFAULT_PARAMS.copy()
    if params is not None:
        unknown = set(params) - set(motor)
        if unknown:
            raise ValueError(f"Unknown motor parameters: {unknown}")
        motor.update(params)

    J, b = motor["J"], motor["b"]
    Kt, Ke = motor["Kt"], motor["Ke"]
    R, L = motor["R"], motor["L"]

    if min(J, Kt, Ke, R, L) <= 0 or b < 0:
        raise ValueError("Invalid motor parameters.")

    steps = int(round(duration / dt))
    time = np.arange(steps + 1) * dt

    current = 0.0
    omega = 0.0
    integral = 0.0
    previous_speed = 0.0
    rows = []

    for k, t in enumerate(time):
        speed_rpm = omega * 60.0 / (2.0 * np.pi)
        error = target_rpm - speed_rpm

        # Derivative on measured speed avoids setpoint derivative kick.
        d_speed = (speed_rpm - previous_speed) / dt if k else 0.0

        proportional = kp * error
        derivative = -kd * d_speed

        # Compute output before updating the integral.
        raw_voltage = proportional + ki * integral + derivative
        voltage = float(np.clip(raw_voltage, 0.0, max_voltage))

        # Integrate only when unsaturated or when error helps
        # bring the output back out of saturation.
        if (
            0.0 < raw_voltage < max_voltage
            or (raw_voltage >= max_voltage and error < 0)
            or (raw_voltage <= 0.0 and error > 0)
        ):
            integral += error * dt

        rows.append({
            "time_s": t,
            "target_rpm": target_rpm,
            "speed_rpm": speed_rpm,
            "error_rpm": error,
            "voltage_V": voltage,
            "current_A": current,
        })

        if k == steps:
            break

        di_dt = (voltage - R * current - Ke * omega) / L
        domega_dt = (Kt * current - b * omega - load_torque) / J

        current += di_dt * dt
        omega += domega_dt * dt
        previous_speed = speed_rpm

    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = simulate_pid()

    print("Closed-loop PID simulation")
    print(f"Target: {result['target_rpm'].iloc[0]:.1f} RPM")
    print(f"Final speed: {result['speed_rpm'].iloc[-1]:.2f} RPM")
    print(f"Final error: {result['error_rpm'].iloc[-1]:.2f} RPM")
    print(f"Maximum voltage: {result['voltage_V'].max():.2f} V")
