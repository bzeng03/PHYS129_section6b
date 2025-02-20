import numpy as np
import matplotlib.pyplot as plt

def ito_stochastic_integrator(mu, sigma, T, N, seed=114514):
    np.random.seed(seed)  # Set seed for reproducibility
    dt = T / N  # Time step
    t = np.linspace(0, T, N+1)
    X = np.zeros(N+1)
    X[0] = 1  # Initial condition X_0 = 1
    W = np.random.randn(N) * np.sqrt(dt)  # Wiener increments
    
    for j in range(N):
        X[j+1] = X[j] + mu * X[j] * dt + sigma * X[j] * W[j]
    
    return t, X

# Parameters
mu = 0.1   # Drift coefficient
sigma = 0.2  # Diffusion coefficient
T = 10.0     # Total time
N = 100      # Number of time steps

# Generate and plot
t, X = ito_stochastic_integrator(mu, sigma, T, N)
plt.figure(figsize=(10, 5))
plt.plot(t, X, label="Ito Integral Trajectory")
plt.xlabel("Time")
plt.ylabel("X_t")
plt.title("Ito Stochastic Integrator Simulation")
plt.legend()
plt.show()
plt.savefig("plot_c.png")