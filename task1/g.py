import numpy as np
import matplotlib.pyplot as plt

def autocorrelation(F, max_lag):
    mean_F = np.mean(F)
    F_fluctuations = F - mean_F
    autocorr = np.correlate(F_fluctuations, F_fluctuations, mode='full')
    autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
    return autocorr[:max_lag]

def stochastic_integrator(mu, sigma, T, N, num_paths=100, seed=42):
    np.random.seed(seed)
    dt = T / N
    X = np.ones((num_paths, N+1))
    W = np.random.randn(num_paths, N) * np.sqrt(dt)
    
    for j in range(N):
        X[:, j+1] = X[:, j] + mu * X[:, j] * dt + sigma * X[:, j] * W[:, j]
    
    return X

def stopping_function(X):
    return X**2

# Parameters
mu = 0.1
sigma = 0.2
T = 30.0
N = 300
num_paths = 100
max_lag = 50

X = stochastic_integrator(mu, sigma, T, N, num_paths)
F = stopping_function(X)

# Compute autocorrelation for different stopping times
time_indices = [5, 10, 20, 30]
plt.figure(figsize=(10, 6))

for t in time_indices:
    index = int((t / T) * N)
    autocorr = autocorrelation(F[:, index], max_lag)
    plt.plot(range(max_lag), autocorr, label=f'Stopping Time t={t}')

plt.xlabel('Lag (τ)')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of Stopping Function at Different Times')
plt.legend()
plt.show()
plt.savefig("plot_g.png")