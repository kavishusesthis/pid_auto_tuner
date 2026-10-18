
from pathlib import Path

import pandas as pd

DATASET = Path("data/pid_training_dataset.csv")

df = pd.read_csv(DATASET)

# A controller is considered acceptable if it has low final error
# and limited overshoot. Settling time is also reported separately.
df["acceptable"] = (
    (df["steady_state_error_percent"] <= 2.0)
    & (df["overshoot_percent"] <= 5.0)
)

print(f"Total configurations: {len(df)}")
print(f"Configurations meeting error/overshoot limits: {df['acceptable'].sum()}")
print(f"Configurations that settled: {df['settling_time_s'].notna().sum()}")

print("\nTop 10 by IAE (lowest first):")

columns = [
    "Kp",
    "Ki",
    "Kd",
    "final_speed_rpm",
    "steady_state_error_percent",
    "overshoot_percent",
    "rise_time_s",
    "settling_time_s",
    "IAE_rpm_s",
    "acceptable",
]

print(
    df.sort_values("IAE_rpm_s")[columns]
    .head(10)
    .to_string(index=False, float_format=lambda x: f"{x:.3f}")
)

print("\nBest configurations meeting the error/overshoot limits:")

good = df[df["acceptable"]].sort_values("IAE_rpm_s")

if good.empty:
    print("No configurations meet both limits. We need to expand the gain search.")
else:
    print(
        good[columns]
        .head(10)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

Path("outputs").mkdir(parents=True, exist_ok=True)
df.sort_values("IAE_rpm_s").to_csv(
    "outputs/pid_ranked_dataset.csv", index=False
)

print("\nSaved ranked dataset to outputs/pid_ranked_dataset.csv")
