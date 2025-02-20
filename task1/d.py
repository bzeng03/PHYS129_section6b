import numpy as np
import matplotlib.pyplot as plt

def stratonovich_stochastic_integrator(mu, sigma, T, N, seed=114514):
    np.random.seed(seed)  # Set seed for reproducibility
    dt = T / N  # Time step
    t = np.linspace(0, T, N+1)
    X = np.zeros(N+1)
    X[0] = 1  # Initial condition X_0 = 1
    W = np.random.randn(N) * np.sqrt(dt)  # Wiener increments
    
    for j in range(N):
        X_mid = X[j] + 0.5 * sigma * X[j] * W[j]  # Midpoint correction
        X[j+1] = X[j] + mu * X_mid * dt + sigma * X_mid * W[j]
    
    return t, X

# Parameters
mu = 0.1   # Drift coefficient
sigma = 0.2  # Diffusion coefficient
T = 10.0     # Total time
N = 100      # Number of time steps

# Generate and plot
t, X = stratonovich_stochastic_integrator(mu, sigma, T, N)
plt.figure(figsize=(10, 5))
plt.plot(t, X, label="Stratonovich Integral Trajectory")
plt.xlabel("Time")
plt.ylabel("X_t")
plt.title("Stratonovich Stochastic Integrator Simulation")
plt.legend()
plt.show()
plt.savefig("plot_d.png")