import numpy as np
import matplotlib.pyplot as plt

# Step 1: Set actual parameters
actual_mean = 50
actual_variance = 25      # Variance = 25
actual_std = np.sqrt(actual_variance)

# Step 2: Generate synthetic data
np.random.seed(42)        # For reproducibility
data = np.random.normal(actual_mean, actual_std, 1000)

# Step 3: Maximum Likelihood Estimation (MLE)
estimated_mean = np.mean(data)
estimated_variance = np.mean((data - estimated_mean) ** 2)

# Step 4: Display results
print("Actual Mean:", actual_mean)
print("Estimated Mean (MLE):", round(estimated_mean, 2))

print("\nActual Variance:", actual_variance)
print("Estimated Variance (MLE):", round(estimated_variance, 2))

# Step 5: Plot Histogram
plt.figure(figsize=(8,5))
plt.hist(data, bins=30, density=True, color='skyblue', edgecolor='black')
plt.axvline(actual_mean, color='red', linestyle='--', linewidth=2, label='Actual Mean')
plt.axvline(estimated_mean, color='green', linestyle='-', linewidth=2, label='Estimated Mean')

plt.title("Maximum Likelihood Estimation (MLE)")
plt.xlabel("Data Values")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
