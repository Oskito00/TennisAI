import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data from your table
train_sizes = np.array([25, 50, 75, 100])  # percentages
samples = np.array([19, 38, 57, 77])
train_mse = np.array([0.000005, 0.000016, 0.000015, 0.000014])
test_mse = np.array([0.129145, 0.089949, 0.081761, 0.076280])

# Define power law function for curve fitting (more stable than exponential)
def power_law(x, a, b, c):
    return a * (x ** -b) + c

# Fit curve to test MSE data
popt, _ = curve_fit(power_law, samples, test_mse, p0=[1, 0.5, 0.07], maxfev=10000)

# Generate points for smooth curve
x_smooth = np.linspace(19, 150, 100)  # Start from minimum samples
y_smooth = power_law(x_smooth, *popt)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(samples, test_mse, 'bo', label='Test MSE (Actual)', markersize=8)
plt.plot(x_smooth, y_smooth, 'b-', label='Test MSE (Projected)', alpha=0.7)
plt.plot(samples, train_mse, 'ro', label='Train MSE', markersize=8)

plt.xlabel('Number of Training Samples')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Training Size with Projected Performance')
plt.legend()
plt.grid(True, alpha=0.3)

# Add text annotation
plt.text(0.02, 0.98, 
         'The curve suggests that gathering more\ndata will only marginally improve performance',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')

# Set y-axis to log scale for better visualization
plt.yscale('log')

plt.savefig('mse_curve.png', dpi=300, bbox_inches='tight')
plt.close()