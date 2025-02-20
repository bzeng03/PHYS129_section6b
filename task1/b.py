import numpy as np
import matplotlib.pyplot as plt

def geometric_brownian_motion(mu, sigma, T, N, num_paths=10):
    dt = T / N  # Time step
    t = np.linspace(0, T, N+1)
    X = np.zeros((num_paths, N+1))
    X[:, 0] = 1  # Initial condition X_0 = 1
    
    for i in range(num_paths):
        W = np.random.randn(N) * np.sqrt(dt)  # Wiener increments
        for j in range(N):
            X[i, j+1] = X[i, j] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * W[j])
    
    return t, X

# Parameters
mu = 0.1   # Drift coefficient
sigma = 0.2  # Diffusion coefficient
T = 1.0     # Total time
N = 100     # Number of time steps
num_paths = 5  # Number of simulated paths

# Generate and plot
t, X = geometric_brownian_motion(mu, sigma, T, N, num_paths)
plt.figure(figsize=(10, 5))
for i in range(num_paths):
    plt.plot(t, X[i], label=f'Path {i+1}')
plt.xlabel("Time")
plt.ylabel("X_t")
plt.title("Geometric Brownian Motion Simulation")
plt.legend()
plt.show()
plt.savefig("plot_b.png")