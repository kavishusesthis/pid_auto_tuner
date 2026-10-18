
import pandas as pd

from pid_controller import simulate_pid
from performance_metrics import calculate_metrics


experiments = [
    {
        "name": "Baseline",
        "kp": 0.03,
        "ki": 0.003,
        "kd": 0.0005,
    },
    {
        "name": "Higher integral gain",
        "kp": 0.03,
        "ki": 0.006,
        "kd": 0.0005,
    },
]

results = []

for experiment in experiments:
    response = simulate_pid(
        target_rpm=100.0,
        duration=10.0,
        kp=experiment["kp"],
        ki=experiment["ki"],
        kd=experiment["kd"],
    )

    metrics = calculate_metrics(response)

    results.append({
        "controller": experiment["name"],
        "Kp": experiment["kp"],
        "Ki": experiment["ki"],
        "Kd": experiment["kd"],
        **metrics,
    })

comparison = pd.DataFrame(results)

print("\nPID Controller Comparison")
print("=" * 90)

columns = [
    "controller",
    "Kp",
    "Ki",
    "Kd",
    "final_speed_rpm",
    "steady_state_error_percent",
    "overshoot_percent",
    "rise_time_s",
    "settling_time_s",
    "IAE_rpm_s",
]

print(comparison[columns].to_string(index=False))

comparison.to_csv("outputs/pid_comparison.csv", index=False)
print("\nSaved comparison to outputs/pid_comparison.csv")
