"""Evaluate predicted gains on held-out motors by running the actual simulator."""
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decision_tree import DecisionTreeRegressor
from pid_controller import simulate_pid
from performance_metrics import calculate_metrics
from pid_tuning_utils import response_metrics, refine_gains, score_metrics

MODEL_PATH = ROOT / "outputs" / "decision_tree.json"
DATA_PATH = ROOT / "data" / "optimal_pid_dataset.csv"
RESULT_PATH = ROOT / "outputs" / "unseen_motor_results.csv"


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def rmse(actual, predicted):
    if not actual:
        return float("nan")
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def main():
    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        raise FileNotFoundError("Run dataset generation and model training first.")

    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    model = DecisionTreeRegressor.from_dict(payload)
    features = payload["feature_names"]
    targets = payload["target_names"]
    test_ids = set(payload["test_motor_ids"])
    rows = [row for row in read_csv(DATA_PATH) if row["motor_id"] in test_ids]
    if not rows:
        raise ValueError("No held-out motors found in the optimal dataset.")

    results = []
    target_actual = {name: [] for name in targets}
    target_pred = {name: [] for name in targets}

    for row in rows:
        x = [float(row[name]) for name in features]
        predicted = model.predict_one(x)
        for name, actual, pred in zip(targets, [float(row[n]) for n in targets], predicted):
            target_actual[name].append(actual)
            target_pred[name].append(pred)

        motor = {name: float(row[name]) for name in features}
        duration = float(row["duration_s"])
        dt = float(row["dt"])
        raw_metrics = response_metrics(motor, predicted, duration=duration, dt=dt)
        refined_gains, refined_metrics, refined_score = refine_gains(
            motor, predicted, seed=sum(ord(ch) for ch in row["motor_id"]),
            duration=duration, dt=dt, n_candidates=40,
        )
        results.append({
            "motor_id": row["motor_id"],
            **{name: float(row[name]) for name in features},
            "actual_best_found_Kp": float(row["Kp"]),
            "actual_best_found_Ki": float(row["Ki"]),
            "actual_best_found_Kd": float(row["Kd"]),
            "predicted_Kp": max(0.0, predicted[0]),
            "predicted_Ki": max(0.0, predicted[1]),
            "predicted_Kd": max(0.0, predicted[2]),
            **{f"raw_{key}": value for key, value in raw_metrics.items()},
            "refined_Kp": refined_gains[0],
            "refined_Ki": refined_gains[1],
            "refined_Kd": refined_gains[2],
            **{f"refined_{key}": value for key, value in refined_metrics.items()},
            "refined_score": refined_score,
            "best_found_score_for_reference": float(row["score"]),
        })

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("Unseen-motor evaluation")
    print(f"Held-out motors evaluated: {len(results)}")
    print("Gain prediction RMSE (lower is better; gains have different scales):")
    for name in targets:
        print(f"  {name}: {rmse(target_actual[name], target_pred[name]):.6f}")

    raw_good = sum(
        r["raw_steady_state_error_percent"] <= 2.0 and r["raw_overshoot_percent"] <= 5.0
        for r in results
    )
    refined_good = sum(
        r["refined_steady_state_error_percent"] <= 2.0
        and r["refined_overshoot_percent"] <= 5.0
        for r in results
    )
    print(f"Raw ML-gain responses meeting error <= 2% and overshoot <= 5%: {raw_good}/{len(results)}")
    print(f"After simulation-based refinement: {refined_good}/{len(results)}")
    failed_refined = [
        r for r in results
        if not (
            r["refined_steady_state_error_percent"] <= 2.0
            and r["refined_overshoot_percent"] <= 5.0
        )
    ]
    if failed_refined:
        print("Motors still outside one or both limits after refinement:")
        for r in failed_refined:
            print(
                f"  {r['motor_id']}: error="
                f"{r['refined_steady_state_error_percent']:.3f}%, "
                f"overshoot={r['refined_overshoot_percent']:.3f}%"
            )
    else:
        print("All held-out motors met both limits after refinement.")
    print(f"Detailed results saved: {RESULT_PATH}")
    print("Interpretation: this is a small simulation prototype; metrics do not prove real-motor safety.")


if __name__ == "__main__":
    main()
