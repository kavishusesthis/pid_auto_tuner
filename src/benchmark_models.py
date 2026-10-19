
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from pid_tuning_utils import response_metrics


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "optimal_pid_dataset.csv"
MODEL_PATH = ROOT / "outputs" / "decision_tree.json"
OUTPUT_PATH = ROOT / "outputs" / "ml_model_benchmark.csv"

FEATURES = [
    "J", "b", "Kt", "Ke", "R", "L",
    "load_torque", "target_rpm", "max_voltage",
]
TARGETS = ["Kp", "Ki", "Kd"]
MOTOR_PARAMETERS = [
    "J", "b", "Kt", "Ke", "R", "L",
    "load_torque", "target_rpm", "max_voltage",
]


def main():
    df = pd.read_csv(DATA_PATH)
    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    test_ids = payload.get("test_motor_ids")

    if not test_ids:
        raise ValueError(
            "test_motor_ids not found in outputs/decision_tree.json"
        )

    required = FEATURES + TARGETS + [
        "motor_id", "duration_s", "dt"
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    test_mask = df["motor_id"].isin(test_ids)
    train = df.loc[~test_mask].copy()
    test = df.loc[test_mask].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test split is empty.")

    X_train = train[FEATURES]
    y_train = train[TARGETS]
    X_test = test[FEATURES]
    y_test = test[TARGETS]

    models = {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=6,
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Unseen test motors: {test['motor_id'].nunique()}")

    all_results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"\n{name}")
        print(f"Gain MAE:  {mae:.6f}")
        print(f"Gain RMSE: {rmse:.6f}")

        for i, target in enumerate(TARGETS):
            target_mae = mean_absolute_error(
                y_test[target], predictions[:, i]
            )
            target_rmse = np.sqrt(
                mean_squared_error(y_test[target], predictions[:, i])
            )
            print(
                f"{target}: MAE={target_mae:.6f}, "
                f"RMSE={target_rmse:.6f}"
            )

        passed = 0
        iae_values = []

        for position, (_, row) in enumerate(test.iterrows()):
            motor = {
                key: float(row[key])
                for key in MOTOR_PARAMETERS
            }

            # response_metrics expects one gains tuple: (Kp, Ki, Kd).
            gains = tuple(float(value) for value in predictions[position])

            metrics = response_metrics(
                motor,
                gains,
                duration=float(row["duration_s"]),
                dt=float(row["dt"]),
            )

            error = float(metrics["steady_state_error_percent"])
            overshoot = float(metrics["overshoot_percent"])
            iae = float(metrics["IAE_rpm_s"])

            meets_limits = error <= 2.0 and overshoot <= 5.0
            passed += int(meets_limits)
            iae_values.append(iae)

            actual_gains = y_test.iloc[position].to_numpy(dtype=float)

            all_results.append({
                "model": name,
                "motor_id": row["motor_id"],
                "actual_Kp": actual_gains[0],
                "actual_Ki": actual_gains[1],
                "actual_Kd": actual_gains[2],
                "predicted_Kp": gains[0],
                "predicted_Ki": gains[1],
                "predicted_Kd": gains[2],
                "steady_state_error_percent": error,
                "overshoot_percent": overshoot,
                "IAE_rpm_s": iae,
                "meets_limits": meets_limits,
            })

        print(
            "Motors meeting error <= 2% and overshoot <= 5%: "
            f"{passed}/{len(test)}"
        )
        print(f"Mean IAE: {np.mean(iae_values):.3f}")

    pd.DataFrame(all_results).to_csv(OUTPUT_PATH, index=False)
    print(f"\nPer-motor results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
