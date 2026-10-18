"""Train a from-scratch Decision Tree and reserve whole motors for testing."""
import csv
import json
import random
from pathlib import Path

from decision_tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "optimal_pid_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "decision_tree.json"

FEATURES = ["J", "b", "Kt", "Ke", "R", "L", "load_torque", "target_rpm", "max_voltage"]
TARGETS = ["Kp", "Ki", "Kd"]
TEST_FRACTION = 0.20
SEED = 42


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}. Generate the dataset first.")
    return rows


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run src/generate_motor_dataset.py first.")
    rows = read_rows(DATA_PATH)
    motor_ids = sorted({row["motor_id"] for row in rows})
    if len(motor_ids) < 10:
        raise ValueError("Need at least 10 distinct motor profiles for a meaningful holdout split.")

    rng = random.Random(SEED)
    shuffled = motor_ids[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * TEST_FRACTION))
    test_ids = sorted(shuffled[:n_test])
    train_ids = sorted(shuffled[n_test:])

    train_rows = [r for r in rows if r["motor_id"] in set(train_ids)]
    X = [[float(r[name]) for name in FEATURES] for r in train_rows]
    y = [[float(r[name]) for name in TARGETS] for r in train_rows]

    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=3).fit(X, y)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        **model.to_dict(),
        "feature_names": FEATURES,
        "target_names": TARGETS,
        "training_motor_ids": train_ids,
        "test_motor_ids": test_ids,
        "training_rows": len(train_rows),
        "test_motor_count": len(test_ids),
        "seed": SEED,
        "note": "Test motor IDs were excluded from training. Gains are best-found search labels.",
    }
    MODEL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = OUTPUT_DIR / "training_report.txt"
    report.write_text(
        "From-scratch Decision Tree training report\n"
        f"Total motor profiles: {len(motor_ids)}\n"
        f"Training motor profiles: {len(train_ids)}\n"
        f"Held-out motor profiles: {len(test_ids)}\n"
        f"Training rows: {len(train_rows)}\n"
        f"Features: {', '.join(FEATURES)}\n"
        f"Targets: {', '.join(TARGETS)}\n"
        f"Maximum tree depth: {model.max_depth}\n"
        f"Minimum samples per leaf: {model.min_samples_leaf}\n"
        "The held-out motor IDs are saved in decision_tree.json for reproducible evaluation.\n",
        encoding="utf-8",
    )
    print(f"Trained on {len(train_ids)} motor profiles ({len(train_rows)} rows).")
    print(f"Reserved {len(test_ids)} unseen motor profiles for evaluation.")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Report saved: {report}")


if __name__ == "__main__":
    main()
