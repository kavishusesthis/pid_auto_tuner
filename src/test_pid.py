
import matplotlib.pyplot as plt
from pid_controller import simulate_pid

result = simulate_pid(
    target_rpm=100.0,
    duration=10.0,
    kp=0.03,
    ki=0.006,
    kd=0.0005,
)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(
    result["time_s"], result["target_rpm"],
    "--", label="Target speed"
)
axes[0].plot(
    result["time_s"], result["speed_rpm"],
    label="Actual speed"
)
axes[0].set_ylabel("Speed (RPM)")
axes[0].set_title("DC Motor: Closed-Loop PID Response")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(result["time_s"], result["voltage_V"])
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Voltage (V)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("outputs/pid_response.png", dpi=150)
plt.show()

print(f"Final speed: {result['speed_rpm'].iloc[-1]:.2f} RPM")
print("Graph saved to outputs/pid_response.png")
