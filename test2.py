import numpy as np
import matplotlib.pyplot as plt

# Parameters for the log-uniform distribution
min_val = 1  # Minimum value (inclusive)
max_val = 100  # Maximum value (exclusive)
size = 10000  # Number of samples

# Generate log-uniform distribution
log_min = np.log(min_val)
log_max = np.log(max_val)
uniform_samples = np.random.uniform(log_min, log_max, size)
log_uniform_samples = np.exp(uniform_samples)

# Plot the histogram
plt.hist(log_uniform_samples, bins=100, density=True, alpha=0.6, color='g')
plt.title('Log-Uniform Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()


def generate_log_uniform(min_val, max_val, size):
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    uniform_samples = np.random.uniform(log_min, log_max, size)
    return np.exp(uniform_samples)

def generate_log_uniform_sample(min_val, max_val):
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    uniform_sample = np.random.uniform(log_min, log_max)
    return np.exp(uniform_sample)

# Example usage:
# sample = generate_log_uniform(1, 100)

# Example usage:
# samples = generate_log_uniform(1, 100, 10000)
