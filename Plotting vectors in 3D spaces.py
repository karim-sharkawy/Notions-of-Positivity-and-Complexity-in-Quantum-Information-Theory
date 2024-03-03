import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

soa = np.array([[0, 0, 0, -1, -5, 1], [0, 0, 0, -1, -2, 1],
                [0, 0, 0, 1, 2, -1], [0, 0, 0, -2, -8, 2]]) # good classifier

soa1 = np.array([[1, 1, 1, -2, 6, 2], [1, 1, 1, 7, 1, 3],
                [1, 1, 1, 0, 0, 0], [1, 1, 1, 5, 7, 5]]) # extends

soa2 = np.array([[4, 4, 4, 8, 4, 7], [4, 4, 4, 2, 5, 8],
                [4, 4, 4, 4, 2, 7], [4, 4, 4, 6, 7, 8]]) # doesn't extend

soa3 = np.array([[9, 9, 9, 6, 2, 9], [9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9], [9, 9, 9, 6, 2, 9]])  # farthest B         

colors = ['r', 'g', 'b', 'y']
labels = ['Good Classifier', 'Extends', "Doesn't Extend", 'Farthest B']

X, Y, Z, U, V, W = zip(*soa)
X1, Y1, Z1, U1, V1, W1 = zip(*soa1)
X2, Y2, Z2, U2, V2, W2 = zip(*soa2)
X3, Y3, Z3, U3, V3, W3 = zip(*soa3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(X, Y, Z, U, V, W, color=colors[0], label=labels[0])
ax.quiver(X1, Y1, Z1, U1, V1, W1, color=colors[1], label=labels[1])
ax.quiver(X2, Y2, Z2, U2, V2, W2, color=colors[2], label=labels[2])
ax.quiver(X3, Y3, Z3, U3, V3, W3, color=colors[3], label=labels[3])

'''
# Add labels to all vectors
for i in range(len(soa)):
    ax.text(X[i] + U[i], Y[i] + V[i], Z[i] + W[i], f'({U[i]}, {V[i]}, {W[i]})', color='black')

for i in range(len(soa1)):
    ax.text(X1[i] + U1[i], Y1[i] + V1[i], Z1[i] + W1[i], f'({U1[i]}, {V1[i]}, {W1[i]})', color='black')

for i in range(len(soa2)):
    ax.text(X2[i] + U2[i], Y2[i] + V2[i], Z2[i] + W2[i], f'({U2[i]}, {V2[i]}, {W2[i]})', color='black')

for i in range(len(soa3)):
    ax.text(X3[i] + U3[i], Y3[i] + V3[i], Z3[i] + W3[i], f'({U3[i]}, {V3[i]}, {W3[i]})', color='black')
'''

ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([-20, 20])

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()

plt.title('3D Vector Plot')

plt.show()