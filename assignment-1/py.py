import numpy as np
import matplotlib.pyplot as plt


# Define the covariance matrix
covariance_matrix = np.array([[4, 2, 1],
                              [2, 3, 1.5],
                              [1, 1.5, 2]])

# Compute the SVD
u, s, vh = np.linalg.svd(covariance_matrix)

# Select the first two columns of the matrix of principal components
principal_components_subspace = u[:, :2]

# Simulate a random vector (replace this with your simulation code)
simulated_vector = np.random.randn(3)

# Project the simulated vector onto the subspace
projection = np.dot(simulated_vector, principal_components_subspace)

# Print the results
print("Simulated Vector:", simulated_vector)
print("\nProjection onto the Subspace:", projection)

plt.imshow(covariance_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Covariance')
plt.title('Covariance Matrix Heatmap')
plt.show()