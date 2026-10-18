
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from pid_controller import simulate_pid
from performance_metrics import calculate_metrics


OUTPUT = Path("data/pid_training_dataset.csv")
TARGET_RPM = 100.0
DURATION = 10.0

# Explore a range of PID gains.
kp_values = np.linspace(0.01, 0.05, 6)
ki_values = np.linspace(0.001, 0.012, 6)
kd_values = np.linspace(0.0, 0.001, 5)

rows = []
total = len(kp_values) * len(ki_values) * len(kd_values)

print(f"Generating {total} PID simulations...")

for index, (kp, ki, kd) in enumerate(
    product(kp_values, ki_values, kd_values), start=1
):
    try:
        response = simulate_pid(
            target_rpm=TARGET_RPM,
            duration=DURATION,
            kp=float(kp),
            ki=float(ki),
            kd=float(kd),
        )

        metrics = calculate_metrics(response)

        rows.append({
            "target_rpm": TARGET_RPM,
            "duration_s": DURATION,
            "Kp": kp,
            "Ki": ki,
            "Kd": kd,
            **metrics,
        })

    except (ValueError, FloatingPointError, OverflowError) as exc:
        print(f"Skipped configuration {index}: {exc}")

    if index % 25 == 0 or index == total:
        print(f"Progress: {index}/{total}")

if not rows:
    raise RuntimeError("No simulations completed successfully.")

dataset = pd.DataFrame(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT, index=False)

print("\nDataset generation complete.")
print(f"Successful simulations: {len(dataset)}/{total}")
print(f"Dataset saved to: {OUTPUT.resolve()}")
print("\nFirst five rows:")
print(dataset.head().to_string(index=False))

print("\nMetric summary:")
print(
    dataset[
        [
            "steady_state_error_percent",
            "overshoot_percent",
            "rise_time_s",
            "settling_time_s",
            "IAE_rpm_s",
        ]
    ].describe().round(3)
)
