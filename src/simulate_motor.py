import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "J": 0.01,       # Rotor inertia (kg m^2)
    "b": 0.001,      # Viscous friction coefficient (N m s/rad)
    "Kt": 0.01,      # Motor torque constant (N m/A)
    "Ke": 0.01,      # Back-EMF constant (V s/rad)
    "R": 1.0,        # Armature resistance (ohms)
    "L": 0.5,        # Armature inductance (H)
}


def simulate_dc_motor(
    voltage=6.0,
    duration=2.0,
    dt=0.001,
    params=None,
    load_torque=0.0,
):
    """
    Simulate a DC motor using its electrical and mechanical equations.

    Parameters
    ----------
    voltage : float
        Constant applied armature voltage in volts.
    duration : float
        Simulation duration in seconds.
    dt : float
        Integration time step in seconds.
    params : dict or None
        Optional motor parameters; unspecified values use defaults.
    load_torque : float
        Constant opposing load torque in N m.

    Returns
    -------
    pandas.DataFrame
        Time, current, angular velocity, speed in RPM, and applied voltage.
    """
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive.")

    if load_torque < 0:
        raise ValueError("load_torque must be non-negative.")

    motor = DEFAULT_PARAMS.copy()

    if params is not None:
        unknown = set(params) - set(motor)
        if unknown:
            raise ValueError(f"Unknown motor parameters: {sorted(unknown)}")
        motor.update(params)

    if any(motor[key] <= 0 for key in ("J", "R", "L", "Kt", "Ke")):
        raise ValueError("J, R, L, Kt, and Ke must be positive.")

    if motor["b"] < 0:
        raise ValueError("b must be non-negative.")

    J = motor["J"]
    b = motor["b"]
    Kt = motor["Kt"]
    Ke = motor["Ke"]
    R = motor["R"]
    L = motor["L"]

    steps = int(round(duration / dt))
    time = np.arange(steps + 1) * dt

    current = 0.0
    omega = 0.0

    currents = np.zeros(steps + 1)
    speeds = np.zeros(steps + 1)
    voltages = np.full(steps + 1, float(voltage))

    # Forward Euler integration of the motor equations.
    for k in range(steps):
        di_dt = (voltage - R * current - Ke * omega) / L
        domega_dt = (Kt * current - b * omega - load_torque) / J

        current += di_dt * dt
        omega += domega_dt * dt

        currents[k + 1] = current
        speeds[k + 1] = omega

    return pd.DataFrame({
        "time_s": time,
        "voltage_V": voltages,
        "current_A": currents,
        "speed_rad_s": speeds,
        "speed_rpm": speeds * 60.0 / (2.0 * np.pi),
    })


if __name__ == "__main__":
    result = simulate_dc_motor(voltage=6.0, duration=2.0)

    print("DC Motor Simulation")
    print("-------------------")
    print(f"Simulation duration: {result['time_s'].iloc[-1]:.2f} seconds")
    print(f"Final current: {result['current_A'].iloc[-1]:.3f} A")
    print(f"Final speed: {result['speed_rpm'].iloc[-1]:.2f} RPM")
    print()
    print(result.tail())