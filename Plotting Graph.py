import numpy as np
import matplotlib.pyplot as plt

# Generate elliptic boundary coordinates
theta = np.linspace(0, 2*np.pi, 100)
a = 2.0  # Major axis length
b = 1.0  # Minor axis length
x = a * np.cos(theta)
y = b * np.sin(theta)

# Calculate normals
dx_dt = -a * np.sin(theta)
dy_dt = b * np.cos(theta)
normals = np.stack((-dy_dt, dx_dt), axis=1)
normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize normals

# Find envelope of normals
envelope_x = x - a * normals[:, 0]
envelope_y = y - b * normals[:, 1]

# Plot the elliptic boundary and its envelope of normals
plt.plot(x, y, label='Elliptic Boundary')
plt.plot(envelope_x, envelope_y, label='Envelope of Normals')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')
plt.show()