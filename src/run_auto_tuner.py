"""Predict PID gains, refine them by simulation, and report the final response."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decision_tree import DecisionTreeRegressor
from pid_tuning_utils import response_metrics, refine_gains

MODEL_PATH = ROOT / "outputs" / "decision_tree.json"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run train_gain_model.py first.")

    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    model = DecisionTreeRegressor.from_dict(payload)
    feature_names = payload["feature_names"]

    # Example motor profile; edit these values to test a different simulated motor.
    motor = {
        "J": 0.012,
        "b": 0.0014,
        "Kt": 0.012,
        "Ke": 0.011,
        "R": 1.1,
        "L": 0.45,
        "load_torque": 0.012,
        "target_rpm": 100.0,
        "max_voltage": 12.0,
    }
    duration, dt = 6.0, 0.005
    features = [float(motor[name]) for name in feature_names]
    predicted = [max(0.0, value) for value in model.predict_one(features)]
    raw_metrics = response_metrics(motor, predicted, duration=duration, dt=dt)
    refined, refined_metrics, score = refine_gains(
        motor, predicted, seed=2026, duration=duration, dt=dt, n_candidates=80
    )

    print("ML-Based PID Auto-Tuner (Decision Tree implemented from scratch)")
    print("\nPredicted gains from ML model:")
    print(f"  Kp = {predicted[0]:.6f}\n  Ki = {predicted[1]:.6f}\n  Kd = {predicted[2]:.6f}")
    print("\nMetrics before refinement:")
    for name, value in raw_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("\nSelected gains after simulation-based refinement:")
    print(f"  Kp = {refined[0]:.6f}\n  Ki = {refined[1]:.6f}\n  Kd = {refined[2]:.6f}")
    print(f"  Search score = {score:.4f}")
    print("\nMetrics after refinement:")
    for name, value in refined_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("\nThe ML model supplies the initial guess; the simulator selects the final gains.")
    print("Verify constraints before applying any gains to physical hardware.")


if __name__ == "__main__":
    main()
