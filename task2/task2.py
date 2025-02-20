import numpy as np
import matplotlib.pyplot as plt

def langevin_equation(gamma, D, T, N, v0=0, seed=42, num_paths=100):
    np.random.seed(seed)
    dt = T / N
    v = np.zeros((num_paths, N+1))
    v[:, 0] = v0
    W = np.random.randn(num_paths, N) * np.sqrt(dt)
    
    for j in range(N):
        v[:, j+1] = v[:, j] - gamma * v[:, j] * dt + np.sqrt(2 * D) * W[:, j]
    
    return v

# Parameters
gamma = 0.5  # Friction coefficient
D = 1.0  # Noise strength
T = 10.0  # Total time
N = 1000  # Number of time steps
num_paths = 100  # Number of simulated paths

# Generate and compute statistics
v = langevin_equation(gamma, D, T, N, num_paths=num_paths)
mean_v = np.mean(v, axis=0)
var_v = np.var(v, axis=0)

t = np.linspace(0, T, N+1)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, mean_v, label='Mean Velocity')
plt.xlabel('Time')
plt.ylabel('Mean v')
plt.title('Mean Velocity over Time')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, var_v, label='Velocity Variance')
plt.xlabel('Time')
plt.ylabel('Variance')
plt.title('Variance of Velocity over Time')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("plot_task2.png")

# Print statistics
print(f"Final Mean Velocity: {mean_v[-1]}")
print(f"Final Velocity Variance: {var_v[-1]}")
