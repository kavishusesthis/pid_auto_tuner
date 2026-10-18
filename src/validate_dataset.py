
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/pid_training_dataset.csv")
REPORT_PATH = Path("outputs/dataset_validation_report.txt")

df = pd.read_csv(DATA_PATH)

required_columns = [
    "target_rpm",
    "duration_s",
    "Kp",
    "Ki",
    "Kd",
    "final_speed_rpm",
    "final_error_rpm",
    "steady_state_error_percent",
    "overshoot_percent",
    "rise_time_s",
    "settling_time_s",
    "IAE_rpm_s",
]

missing_columns = sorted(set(required_columns) - set(df.columns))

if missing_columns:
    raise ValueError(f"Dataset is missing columns: {missing_columns}")

numeric = df[required_columns].apply(pd.to_numeric, errors="coerce")
finite_values = np.isfinite(numeric.to_numpy())

# Missing rise/settling times are allowed when a response
# does not reach the required threshold during the simulation.
allowed_missing = {"rise_time_s", "settling_time_s"}
required_for_finite_check = [
    c for c in required_columns if c not in allowed_missing
]

non_numeric_or_missing = {
    c: int(numeric[c].isna().sum())
    for c in required_columns
    if c not in allowed_missing and numeric[c].isna().any()
}

infinite_counts = {
    c: int(np.isinf(numeric[c].to_numpy(dtype=float)).sum())
    for c in required_columns
    if np.isinf(numeric[c].to_numpy(dtype=float)).any()
}

duplicate_rows = int(df.duplicated().sum())
duplicate_gain_sets = int(
    df.duplicated(subset=["Kp", "Ki", "Kd"]).sum()
)

invalid_gain_rows = int(
    (
        (numeric["Kp"] < 0)
        | (numeric["Ki"] < 0)
        | (numeric["Kd"] < 0)
    ).fillna(True).sum()
)

invalid_metric_rows = int(
    (
        (numeric["target_rpm"] <= 0)
        | (numeric["duration_s"] <= 0)
        | (numeric["overshoot_percent"] < 0)
        | (numeric["IAE_rpm_s"] < 0)
    ).fillna(True).sum()
)

missing_rise = int(numeric["rise_time_s"].isna().sum())
missing_settling = int(numeric["settling_time_s"].isna().sum())

acceptable = (
    (numeric["steady_state_error_percent"] <= 2.0)
    & (numeric["overshoot_percent"] <= 5.0)
).fillna(False)

report = [
    "PID DATASET VALIDATION REPORT",
    "=" * 32,
    f"Dataset: {DATA_PATH.resolve()}",
    f"Rows: {len(df)}",
    f"Columns: {len(df.columns)}",
    f"Exact duplicate rows: {duplicate_rows}",
    f"Repeated Kp/Ki/Kd combinations: {duplicate_gain_sets}",
    f"Rows with invalid PID gains: {invalid_gain_rows}",
    f"Rows with invalid target/duration/metrics: {invalid_metric_rows}",
    f"Missing required values: {non_numeric_or_missing}",
    f"Infinite values by column: {infinite_counts}",
    f"Missing rise times: {missing_rise}",
    f"Missing settling times: {missing_settling}",
    f"Configurations meeting error <= 2% and overshoot <= 5%: {int(acceptable.sum())}",
    "",
    "Missing timing metrics are not automatically errors:",
    "they may indicate that the response did not reach the required",
    "threshold within the simulation duration.",
]

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

print("\n".join(report))
print(f"\nReport saved to: {REPORT_PATH.resolve()}")
