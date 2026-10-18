import matplotlib.pyplot as plt
from simulate_motor import simulate_dc_motor

result = simulate_dc_motor(
    voltage=6.0,
    duration=2.0,
)

plt.figure(figsize=(9, 5))
plt.plot(result["time_s"], result["speed_rpm"])
plt.xlabel("Time (seconds)")
plt.ylabel("Motor Speed (RPM)")
plt.title("DC Motor Speed Response to 6 V")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/motor_response.png", dpi=150)
plt.show()

print("Graph saved to outputs/motor_response.png")