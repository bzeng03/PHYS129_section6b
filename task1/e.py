import numpy as np
import matplotlib.pyplot as plt

def ito_stochastic_integrator(mu, sigma, T, N, seed=42, num_paths=100):
    np.random.seed(seed)
    dt = T / N
    X = np.ones((num_paths, N+1))
    
    for i in range(num_paths):
        W = np.random.randn(N) * np.sqrt(dt)
        for j in range(N):
            X[i, j+1] = X[i, j] + mu * X[i, j] * dt + sigma * X[i, j] * W[j]
    
    return X

def stratonovich_stochastic_integrator(mu, sigma, T, N, seed=42, num_paths=100):
    np.random.seed(seed)
    dt = T / N
    X = np.ones((num_paths, N+1))
    
    for i in range(num_paths):
        W = np.random.randn(N) * np.sqrt(dt)
        for j in range(N):
            X_mid = X[i, j] + 0.5 * sigma * X[i, j] * W[j]
            X[i, j+1] = X[i, j] + mu * X_mid * dt + sigma * X_mid * W[j]
    
    return X

# Parameters
mu = 0.1
sigma = 0.2
T = 10.0
N_values = np.logspace(1, 4, num=10, dtype=int)
num_paths = 100

mean_ito, var_ito = [], []
mean_strat, var_strat = [], []

for N in N_values:
    X_ito = ito_stochastic_integrator(mu, sigma, T, N, num_paths=num_paths)
    X_strat = stratonovich_stochastic_integrator(mu, sigma, T, N, num_paths=num_paths)
    
    mean_ito.append(np.mean(X_ito[:, -1]))
    var_ito.append(np.var(X_ito[:, -1]))
    mean_strat.append(np.mean(X_strat[:, -1]))
    var_strat.append(np.var(X_strat[:, -1]))

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(N_values, mean_ito, marker='o', label='Ito Mean')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Mean')
plt.title('Ito Integral Mean')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(N_values, var_ito, marker='o', label='Ito Variance')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Variance')
plt.title('Ito Integral Variance')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(N_values, mean_strat, marker='o', label='Stratonovich Mean')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Mean')
plt.title('Stratonovich Integral Mean')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(N_values, var_strat, marker='o', label='Stratonovich Variance')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Variance')
plt.title('Stratonovich Integral Variance')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("plot_e.png")