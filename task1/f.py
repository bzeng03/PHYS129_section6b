import numpy as np
import matplotlib.pyplot as plt

def ito_integrator(mu, sigma, T, N, num_paths=100, seed=42):
    np.random.seed(seed)
    dt = T / N
    X = np.ones((num_paths, N+1))
    W = np.random.randn(num_paths, N) * np.sqrt(dt)
    
    for j in range(N):
        X[:, j+1] = X[:, j] + mu * X[:, j] * dt + sigma * X[:, j] * W[:, j]
    
    return X

def stratonovich_integrator(mu, sigma, T, N, num_paths=100, seed=42):
    np.random.seed(seed)
    dt = T / N
    X = np.ones((num_paths, N+1))
    W = np.random.randn(num_paths, N) * np.sqrt(dt)
    
    for j in range(N):
        X_mid = X[:, j] + 0.5 * sigma * X[:, j] * W[:, j]
        X[:, j+1] = X[:, j] + mu * X_mid * dt + sigma * X_mid * W[:, j]
    
    return X

def stopping_function_dynamics(X):
    return X**2

# Parameters
mu = 0.1
sigma = 0.2
T = 10.0
N = 100
num_paths = 100

X_ito = ito_integrator(mu, sigma, T, N, num_paths)
X_strat = stratonovich_integrator(mu, sigma, T, N, num_paths)

F_I = stopping_function_dynamics(X_ito)
F_S = stopping_function_dynamics(X_strat)

# Compute mean and variance
mean_F_I = np.mean(F_I[:, -1])
var_F_I = np.var(F_I[:, -1])
mean_F_S = np.mean(F_S[:, -1])
var_F_S = np.var(F_S[:, -1])

N_values = np.logspace(1, 4, num=10, dtype=int)
mean_ito, var_ito = [], []
mean_strat, var_strat = [], []

for N in N_values:
    X_ito = ito_integrator(mu, sigma, T, N, num_paths=num_paths)
    X_strat = stratonovich_integrator(mu, sigma, T, N, num_paths=num_paths)
    
    F_I = stopping_function_dynamics(X_ito)
    F_S = stopping_function_dynamics(X_strat)
    
    mean_ito.append(np.mean(F_I[:, -1]))
    var_ito.append(np.var(F_I[:, -1]))
    mean_strat.append(np.mean(F_S[:, -1]))
    var_strat.append(np.var(F_S[:, -1]))

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(N_values, mean_ito, marker='o', label='Ito Mean')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Mean')
plt.title('Ito Stopping Function Mean')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(N_values, var_ito, marker='o', label='Ito Variance')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Variance')
plt.title('Ito Stopping Function Variance')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(N_values, mean_strat, marker='o', label='Stratonovich Mean')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Mean')
plt.title('Stratonovich Stopping Function Mean')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(N_values, var_strat, marker='o', label='Stratonovich Variance')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Variance')
plt.title('Stratonovich Stopping Function Variance')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("plot_f.png")

# Print Statistics
print(f"Ito Stopping Function - Mean: {mean_F_I}, Variance: {var_F_I}")
print(f"Stratonovich Stopping Function - Mean: {mean_F_S}, Variance: {var_F_S}")