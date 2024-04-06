# This code was created by karim with some editing by luke
import numpy as np

extendable_class = np.load("/content/extendableMappings.npy")
nonextendable_class = np.load("/content/nonExtendableMappings.npy")
goodclassifier_class = np.load("/content/trueClassifiersGood.npy")
farthest_class = np.load("/content/farthestBsMORE.npy")
trueClassifiersGood = np.load("/content/trueClassifiersGood.npy")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Assign colors and labels to each set of vectors
colors = ['r', 'g', 'b', 'y']
labels = ['Good Classifier', 'Extends', "Doesn't Extend", 'Farthest B']

# Create a new figure and axis for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# PLOT CLASSIFIER
soa = np.array([[0, 0, 0, -1, -5, 1], [0, 0, 0, -1, -2, 1],
                [0, 0, 0, 1, 2, -1], [0, 0, 0, -2, -8, 2]])  # Vectors representing a "Good Classifier"
# Unpack the arrays for plotting
X, Y, Z, U, V, W = zip(*soa)
# Plot each set of vectors using quiver plot
ax.quiver(X, Y, Z, U, V, W, color=colors[0], label=labels[0], linewidth=0.5)


# PLOT EXTENDABLE MAPPINGS
for i in range(5):
  soa1 = extendable_class[i]
  soa1 = soa1[:, :-1] # delete the last column bc we don't use it to graph
  padded = np.hstack((np.zeros((soa1.shape[0], 3)), soa1))
  X, Y, Z, U, V, W = zip(*padded)
  ax.quiver(X, Y, Z, U, V, W, color=colors[1], label=labels[1], linewidth=0.5, alpha=0.5)


# PLOT NONEXTENDABLE MAPPINGS
for i in range(5):
  soa2 = nonextendable_class[i]
  soa2 = soa2[:, :-1] # delete the last column bc we don't use it to graph
  padded = np.hstack((np.zeros((soa2.shape[0], 3)), soa2))
  X, Y, Z, U, V, W = zip(*padded)
  ax.quiver(X, Y, Z, U, V, W, color=colors[2], label=labels[2], linewidth=0.5, alpha=0.5)


# Set limits for the axes
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([-20, 20])

# Set labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Add legend to the plot
# ax.legend()

# Set title for the plot
plt.title('3D Vector Plot')

# Show the plot
plt.show()
