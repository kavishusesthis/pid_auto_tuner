"""Generate motor-aware PID experiments using the existing simulator.

This script searches a random set of gain candidates for each randomly sampled
simulated motor. The best candidate is the best FOUND by this search, not a
mathematically guaranteed global optimum.
"""
import csv
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pid_controller import simulate_pid
from performance_metrics import calculate_metrics

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_MOTORS = 200
N_RANDOM_CANDIDATES = 60
DT = 0.005
DURATION = 6.0
MAX_VOLTAGE = 12.0

MOTOR_PARAMETER_RANGES = {
    "J": (0.005, 0.025),
    "b": (0.0005, 0.003),
    "Kt": (0.007, 0.018),
    "Ke": (0.007, 0.018),
    "R": (0.7, 1.5),
    "L": (0.25, 0.8),
}
FEATURES = ["J", "b", "Kt", "Ke", "R", "L", "load_torque", "target_rpm", "max_voltage"]


def make_motor(rng):
    motor = {key: rng.uniform(*bounds) for key, bounds in MOTOR_PARAMETER_RANGES.items()}
    motor["load_torque"] = rng.uniform(0.0, 0.03)
    motor["target_rpm"] = rng.choice([60.0, 80.0, 100.0, 120.0])
    motor["max_voltage"] = MAX_VOLTAGE
    return motor


def candidates(rng):
    values = [
        (0.01, 0.005, 0.0005),
        (0.05, 0.005, 0.0),
        (0.03, 0.003, 0.001),
        (0.07, 0.008, 0.001),
    ]
    for _ in range(N_RANDOM_CANDIDATES):
        values.append((
            rng.uniform(0.005, 0.10),
            rng.uniform(0.0, 0.02),
            rng.uniform(0.0, 0.005),
        ))
    # Keep order stable while removing duplicate gain triples.
    return list(dict.fromkeys(values))


def score(metrics, duration):
    """Heuristic objective: lower is better; unsettled responses are penalized."""
    error_pct = metrics["steady_state_error_percent"]
    overshoot = metrics["overshoot_percent"]
    settling = metrics["settling_time_s"]
    iae = metrics["IAE_rpm_s"]
    if not math.isfinite(settling):
        settling = duration * 2.0
    return (
        10.0 * min(error_pct, 100.0)
        + 3.0 * min(overshoot, 100.0)
        + 0.5 * settling
        + 0.01 * min(iae, 100000.0)
    )


def main():
    rng = random.Random(SEED)
    experiment_rows = []
    optimal_rows = []

    for motor_index in range(N_MOTORS):
        motor_id = f"M{motor_index:04d}"
        motor = make_motor(rng)
        params = {key: motor[key] for key in MOTOR_PARAMETER_RANGES}
        best_row = None
        best_score = float("inf")

        for kp, ki, kd in candidates(rng):
            try:
                response = simulate_pid(
                    target_rpm=motor["target_rpm"], duration=DURATION, dt=DT,
                    kp=kp, ki=ki, kd=kd, max_voltage=motor["max_voltage"],
                    params=params, load_torque=motor["load_torque"],
                )
                metrics = calculate_metrics(response)
                candidate_score = score(metrics, DURATION)
                if not math.isfinite(candidate_score):
                    continue
            except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
                continue

            row = {
                "motor_id": motor_id,
                **motor,
                "duration_s": DURATION,
                "dt": DT,
                "Kp": float(kp), "Ki": float(ki), "Kd": float(kd),
                **metrics,
                "score": float(candidate_score),
            }
            experiment_rows.append(row)
            if candidate_score < best_score:
                best_score = candidate_score
                best_row = row

        if best_row is not None:
            optimal_rows.append(best_row)
            print(f"{motor_id}: best score {best_score:.3f}")
        else:
            print(f"{motor_id}: no valid candidates")

    if len(optimal_rows) < 10:
        raise RuntimeError(
            f"Only {len(optimal_rows)} motors produced valid results. "
            "Check the simulator and parameter ranges."
        )

    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    all_path = DATA_DIR / "motor_training_data.csv"
    best_path = DATA_DIR / "optimal_pid_dataset.csv"
    write_csv(all_path, experiment_rows)
    write_csv(best_path, optimal_rows)

    print("\nDataset generation complete.")
    print(f"Candidate experiments: {len(experiment_rows)} -> {all_path}")
    print(f"Motor profiles with a selected candidate: {len(optimal_rows)} -> {best_path}")
    print("Note: selected gains are best among the candidates tested, not guaranteed global optima.")


if __name__ == "__main__":
    main()
