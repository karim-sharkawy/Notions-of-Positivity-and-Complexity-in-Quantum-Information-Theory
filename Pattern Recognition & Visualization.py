import numpy as np

Extendables = np.load('/content/Extendables.npy')
Nonextendables = np.load('/content/Nonextendables.npy')
Classifiers = np.load('/content/Classifiers.npy')

tau = np.array([[0.5, 0, 0, 0.5],
                [0, 0.5, 0.5, 0],
                [0.5, 0, 0.5, 0],
                [0, 0.5, 0, 0.5]])

INVtau = np.array([[0.75, -0.25, 0.75, -0.25],
                   [-0.25, 0.75, -0.25, 0.75],
                   [-0.25, 0.75, 0.75, -0.25],
                   [0.75, -0.25, -0.25, 0.75]])

print("Extendable count: ", len(Extendables))
print("Nonextendable count: ", len(Nonextendables))
print("Classifier count: ", len(Classifiers))

extendable_class = Extendables
nonextendable_class = Nonextendables
goodclassifier_class = Classifiers

### Visualization
    # 1.   Cone Visualization
    # 2.   Distances between mappings

# Cone Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assign colors and labels to each set of vectors
colors = ['r', 'g', 'b', 'y']
labels = ['Classifier', 'Extendable', "Nonextendable", 'Farthest B']

# Create a new figure and axis for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot classifiers
for i in range(25):
    soa0 = goodclassifier_class[i]
    soa0 = soa0[:, :-1]  # delete the last column bc we don't use it to graph
    ax.scatter(soa0[:, 0], soa0[:, 1], soa0[:, 2], color=colors[0], alpha=0.2)

# Plot extendable mappings
for i in range(25):
    soa1 = extendable_class[i]
    soa1 = soa1[:, :-1]  # delete the last column bc we don't use it to graph
    ax.scatter(soa1[:, 0], soa1[:, 1], soa1[:, 2], color=colors[1], alpha=0.2)

# Plot nonextendable mappings
for i in range(25):
    soa2 = nonextendable_class[i]
    soa2 = soa2[:, :-1]  # delete the last column bc we don't use it to graph
    ax.scatter(soa2[:, 0], soa2[:, 1], soa2[:, 2], color=colors[2], alpha=0.2)

# Set limits for the axes
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([-20, 20])

# Set labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Set title for the plot
plt.title('3D Vector Plot')

scatter1_proxy = plt.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
scatter2_proxy = plt.Line2D([0],[0], linestyle="none", c='g', marker = 'o')
scatter3_proxy = plt.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
ax.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy], labels)

# Show the plot
plt.show()

# 1) Average Distances

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to calculate average distances and plot results between any two datasets
def calculate_average_distances_and_plot(data1, data2, plot_title='3D Visualization of Average Distances', xlabel='Dataset 1 Index', ylabel='Dataset 2 Index'):
    average_distances = []
    positions = []  # Positions for plotting
    colors = []  # Colors for differentiation

    num_data1 = len(data1)
    num_data2 = len(data2)

    for i in range(num_data1):
        for j in range(num_data2):
            distances = []
            for row_d2 in data2[j]:
                for row_d1 in data1[i]:
                    dot_product = np.dot(row_d2, row_d1)
                    distances.append(dot_product)
            average_distance = np.sqrt(np.mean(np.square(distances)))
            average_distances.append(average_distance)
            positions.append((i, j, average_distance))  # Append positions for plotting
            colors.append('blue') if i < num_data1 / 2 else colors.append('red')

    positions = np.array(positions)
    colors = np.array(colors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot average distances as markers
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, marker='o')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Average Distance')
    ax.set_title(plot_title)
    plt.show()

    return average_distances

# Now you can use the function for different comparisons:
# Compare Extendables and Nonextendables
average_distances = calculate_average_distances_and_plot(Extendables[:500], Nonextendables[:500],
                                                         plot_title='Distance between Extendables and Nonextendables',
                                                         xlabel='Extendable Matrix Index',
                                                         ylabel='Nonextendable Matrix Index')

print(f"Mean average distance: {np.mean(average_distances)}")
print(f"Max average distance: {np.max(average_distances)}")
print(f"Min average distance: {np.min(average_distances)}\n")

# plotting all three

# Generalized function for calculating average distances and plotting them
def calculate_average_distances_and_plot_multiple(data_sets, plot_title='3D Visualization of Average Distances'):
    # Initialize lists for average distances and positions
    average_distances = []
    positions = []  # Positions for plotting
    colors = []  # Colors for differentiation
    labels = []  # To label each dataset for differentiation

    # Loop through all combinations of data_sets
    num_sets = len(data_sets)
    for i in range(num_sets):
        for j in range(i+1, num_sets):  # Only plot pairwise comparisons
            data1 = data_sets[i]
            data2 = data_sets[j]

            num_data1 = len(data1)
            num_data2 = len(data2)

            # Compare the data
            for k in range(num_data1):
                for l in range(num_data2):
                    distances = []
                    for row_d2 in data2[l]:
                        for row_d1 in data1[k]:
                            dot_product = np.dot(row_d2, row_d1)
                            distances.append(dot_product)
                    average_distance = np.sqrt(np.mean(np.square(distances)))
                    average_distances.append(average_distance)
                    positions.append((k, l, average_distance))  # Append positions for plotting
                    colors.append('blue') if k < num_data1 / 2 else colors.append('red')
                    labels.append(f'{i}-{j}')  # Label for the pair (i, j)

    # Convert lists to numpy arrays for plotting
    positions = np.array(positions)
    colors = np.array(colors)

    # Plot results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot average distances as markers
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, marker='o')

    # Set labels and title
    ax.set_xlabel('Matrix Index from Dataset 1')
    ax.set_ylabel('Matrix Index from Dataset 2')
    ax.set_zlabel('Average Distance')
    ax.set_title(plot_title)

    # Optionally, show a legend for differentiation (if you want)
    # ax.legend(handles=scatter.legend_elements()[0], labels=labels)

    # Show plot
    plt.show()

    # Return distances for further analysis if needed
    return average_distances

# Example usage:
Extendables = np.load('/content/Extendables.npy')
Nonextendables = np.load('/content/Nonextendables.npy')
Classifiers = np.load('/content/Classifiers.npy')

# Now you can use the function to plot distances between all three datasets in one plot
data_sets = [Extendables[:500], Nonextendables[:500], Classifiers[:500]]

# Call function to plot and calculate average distances
average_distances = calculate_average_distances_and_plot_multiple(data_sets,
                                                                 plot_title='Distances between Extendables, Nonextendables, and Classifiers')

print("Average distances between the datasets:\n")
print(f"Mean average distance: {np.mean(average_distances)}")
print(f"Max average distance: {np.max(average_distances)}")
print(f"Min average distance: {np.min(average_distances)}\n")

# 2) Cosine Distance

import numpy as np
from scipy.spatial.distance import cosine

# Calculate cosine distances (squared cosine similarity) and angles
angle_scores_e = []  # Initialize lists for angle scores for extendables
angle_scores_n = []  # Initialize lists for angle scores for nonextendables
angles_e = []        # Initialize lists for actual angles for extendables
angles_n = []        # Initialize lists for actual angles for nonextendables

def calculate_cosine_distances_and_angles(matrices_e, matrices_n, angle_scores_e, angle_scores_n, angles_e, angles_n):
    num_e = len(matrices_e)
    num_n = len(matrices_n)

    for i in range(num_e):
        for j in range(num_n):
            cosine_similarity = 1 - cosine(matrices_e[i].flatten(), matrices_n[j].flatten())
            angle_scores_e.append(cosine_similarity ** 2)  # Append squared cosine similarity for extendables
            angle_scores_n.append(cosine_similarity ** 2)  # Append squared cosine similarity for nonextendables

            # Calculate actual angle in degrees
            angle_radians = np.arccos(cosine_similarity)
            angle_degrees = np.degrees(angle_radians)
            if not np.isnan(angle_degrees):  # Check for NaN values
                angles_e.append(angle_degrees)
                angles_n.append(angle_degrees)

# Calculate angles and cosine distances
calculate_cosine_distances_and_angles(extendable_class[:1000], nonextendable_class[:1000], angle_scores_e, angle_scores_n, angles_e, angles_n)

# Print results
print("Cosine distances (squared cosine similarity) and angles between extendables and nonextendables:\n")
print(f"Mean angle score: {np.mean(angle_scores_e)}")
print(f"Max angle score: {np.max(angle_scores_e)}")
print(f"Min angle score: {np.min(angle_scores_e)}\n")
print(f"Mean angle (degrees): {np.mean(angles_e)}")
print(f"Max angle (degrees): {np.max(angles_e)}")
print(f"Min angle (degrees): {np.min(angles_e)}")

# Function to calculate cosine similarity and angle in degrees
def calculate_cosine_similarity_and_angle(matrix1, matrix2):
    vec1 = matrix1.flatten()
    vec2 = matrix2.flatten()

    # Calculate cosine similarity (squared cosine similarity)
    cosine_similarity = 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle_score = cosine_similarity ** 2

    # Calculate angle in degrees
    angle_radians = np.arccos(cosine_similarity)
    angle_degrees = np.degrees(angle_radians)

    return angle_score, angle_degrees

# Prepare data for plotting
positions_e = []  # Positions for extendables
positions_n = []  # Positions for nonextendables
colors = []  # Colors for differentiation (optional)

# Iterate through matrices and calculate cosine similarities and angles
for matrix_e in extendable_class[:100]:
    for matrix_n in nonextendable_class[:100]:
        angle_score, angle_degrees = calculate_cosine_similarity_and_angle(matrix_e, matrix_n)

        # Append positions and colors
        positions_e.append((angle_score, angle_degrees, 0))  # Position for extendable
        positions_n.append((angle_score, angle_degrees, 1))  # Position for nonextendable
        colors.append('blue')  # Color for extendable
        colors.append('red')   # Color for nonextendable

# Convert lists to numpy arrays
positions_e = np.array(positions_e)
positions_n = np.array(positions_n)

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot extendable matrices as blue markers
ax.scatter(positions_e[:, 0], positions_e[:, 1], positions_e[:, 2], c='blue', marker='o', label='Extendable')

# Plot nonextendable matrices as red markers
ax.scatter(positions_n[:, 0], positions_n[:, 1], positions_n[:, 2], c='red', marker='^', label='Nonextendable')

# Set labels and title
ax.set_xlabel('Cosine Similarity (Squared)')
ax.set_ylabel('Angle (Degrees)')
ax.set_zlabel('Class')
ax.set_title('3D Visualization of Matrices with Cosine Similarities and Angles')

# Add legend
ax.legend()

plt.show()

"""
when is $f \cdot \tau^{-1}$ extendable when $f \in V^+_{small}$?
"""

from scipy.optimize import linprog
import numpy as np

# Function to check if all values in a matrix are positive
positive = []
def all_positive_nested(matrix):
    for row in matrix:
        for value in row:
            if value < 0:
                return False
    return True

for idx, matrix in enumerate(extendable_class):
    if all_positive_nested(matrix):
        positive.append(matrix)
print(len(positive))

tau_inv_right = []
for idx, matrix in enumerate(positive):
  after_transform = np.dot(matrix, INVtau.T)
  tau_inv_right.append(after_transform)

extendable = []
nonExtendable = []
extendableoriginals = []
nonExtendableoriginals= []

for idx, matrix in enumerate(tau_inv_right):
  mat = [1, 1, 1, 1]
  for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],]
  solve_for = [[min(matrix[0][2], matrix[0][3])],
             [min(matrix[1][2], matrix[1][3])],
             [min(matrix[2][2], matrix[2][3])],
             [min(matrix[3][2], matrix[3][3])],
             [-max(-matrix[0][0], -matrix[0][1])],
             [-max(-matrix[1][0], -matrix[1][1])],
             [-max(-matrix[2][0], -matrix[2][1])],
             [-max(-matrix[3][0], -matrix[3][1])]]
  for_A_eq = [[1,1,-1,-1]]
  for_b_eq = [0]

  result = linprog(c=mat, A_ub = for_A_ub, b_ub = solve_for, A_eq = for_A_eq, b_eq = for_b_eq, bounds = None)

  if (result.success):
    extendable.append(matrix)
    extendableoriginals.append(positive[idx])
  else:
    nonExtendable.append(matrix)
    nonExtendableoriginals.append(positive[idx])

print(len(extendable))

for map in extendable:
  print(map, '\n')

for map in nonExtendable:
  print(map, '\n')

"""adding $t_1 + t_3 \geq 0$ and the others as well"""

import numpy as np
from scipy.optimize import linprog

mutations = []
extendablemutations = []
nonextendablemutations = []

for matrix in extendable_class:
  matrix = np.dot(INVtau, matrix)
  mutations.append(matrix)

for matrix in mutations:
  mat = [1, 1, 1, 1]
  for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],
            [-1, 0, -1, 0],
            [-1, 0, 0, -1],
            [0, -1, -1, 0],
            [0, -1, 0, -1]]
  solve_for = [[min(matrix[0][2], matrix[0][3])],
             [min(matrix[1][2], matrix[1][3])],
             [min(matrix[2][2], matrix[2][3])],
             [min(matrix[3][2], matrix[3][3])],
             [-max(-matrix[0][0], -matrix[0][1])],
             [-max(-matrix[1][0], -matrix[1][1])],
             [-max(-matrix[2][0], -matrix[2][1])],
             [-max(-matrix[3][0], -matrix[3][1])],
             [0],
             [0],
             [0],
             [0]]
  for_A_eq = [[1,1,-1,-1]]
  for_b_eq = [0]

  result = linprog(c=mat, A_ub = for_A_ub, b_ub = solve_for, A_eq = for_A_eq, b_eq = for_b_eq, bounds = None)

  if (result.success):
    extendablemutations.append(matrix)

  else:
    nonextendablemutations.append(matrix)

print(len(extendablemutations), "extend")
print(len(nonextendablemutations), "don't extend")

"""# Algebraic Properties"""

from numpy.linalg import matrix_rank
from scipy.linalg import lu, inv

def matrix_properties(matrix):
    # Rank of the matrix
    rank = matrix_rank(matrix)

    # Determinant of the matrix
    determinant = np.linalg.det(matrix)

    # eigenvalues and eigenvectors of the product matrix.T @ matrix
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Null space (kernel) of the matrix
    _, s, V = np.linalg.svd(matrix)
    tolerance = max(matrix.shape) * np.spacing(np.max(s))
    null_space = V.T[:, s < tolerance]

    # Column space (range) of the matrix
    Q, _ = np.linalg.qr(matrix)
    column_space = Q[:, :rank]  # Use `rank` from matrix_rank

    P, L, U = lu(matrix) # Compute LU decomposition
    L_inv = inv(L)
    RREF = np.dot(L_inv, np.dot(P, matrix))

    return {
        'rank': rank,
        'determinant': determinant,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'null_space': null_space,
        'column_space': column_space,
        'rref': RREF
    }

for index, matrix in enumerate(extendable_class[0:2]):
    properties = matrix_properties(matrix)
    print(f"Matrix {index+1}:\n{matrix}")
    print(f"rank: {properties['rank']} and determinant: {properties['determinant']}")
    print(f"eigenvalues: {properties['eigenvalues']}")
    print(f"eigenvectors: \n{properties['eigenvectors']}")
    print(f"RREF: \n{properties['rref']}")
    print(f"null space: \n{properties['null_space']}")
    print(f"column space: \n{properties['column_space']}\n")

"""**coplanar/colinear**"""

from numpy.linalg import matrix_rank
coplanerCountEX = 0
coplanerCountNEX = 0
coplanerCountCLA = 0

# Check coplanarity for every extendable matrix
for matrix in extendable_class:
    if matrix_rank(matrix) < 3:
      coplanerCountEX += 1

for matrix in nonextendable_class: #nonextendable
    if matrix_rank(matrix) < 3:
      coplanerCountNEX += 1

for matrix in goodclassifier_class: #classifier
    if matrix_rank(matrix) < 3:
      coplanerCountCLA += 1

print(f"Percentage of coplanar extendables: {coplanerCountEX}/{len(extendable_class)} = {coplanerCountEX/len(extendable_class)*100}%")
print(f"Percentage of coplanar nonextendables: {coplanerCountNEX}/{len(nonextendable_class)} = {coplanerCountNEX/len(nonextendable_class)*100}%")
print(f"Percentage of coplanar classifiers: {coplanerCountCLA}/{len(goodclassifier_class)} = {coplanerCountCLA/len(goodclassifier_class)*100}%")

print("\nCoplanar matrices imply the rows/columns are colinear!")

"""**colinearity scores**"""

import numpy as np

equalszero = 0
nozero = 0

def row_collinearity_score(matrix):
    min_abs_det = np.inf

    for i in range(4):
        submatrix = np.delete(matrix, i, axis=0)  # Exclude the current row
        for j in range(4):
            if i != j:
                submatrix_temp = np.delete(submatrix, j, axis=1)  # Exclude the corresponding column
                if submatrix_temp.shape[0] == submatrix_temp.shape[1]:  # Check if it's a square matrix
                    det = np.abs(np.linalg.det(submatrix_temp))
                    min_abs_det = min(min_abs_det, det)

    return min_abs_det

for matrix in goodclassifier_class:
    score = row_collinearity_score(matrix)
    if score == 0:
        equalszero += 1 #at least one set of rows in the matrix are linearly dependent
    else:
        nozero += 1

print(equalszero, nozero)

"""# **Classifier Verification**"""

passed_matrices = []
failed_matrices = []

# Iterate over each matrix
for matrix_index in range(len(goodclassifier_class)):
    matrix = goodclassifier_class[matrix_index]  # matrix at current index
    passes_all_conditions = True  # Flag

    # Checking every row
    for i in range(matrix.shape[0]):
        row = matrix[i]
        x, y, z, w = row[0], row[1], row[2], row[3]

        # Checking conditions
        condition1 = x + y - z - w >= 0
        condition2 = x + y + z + w <= 0
        condition3 = 0*x + y - 0*z - w >= 0 #testing
        condition4 = x + y - z - w < 0 #testing

        if not (condition3):
            passes_all_conditions = False
            break

    # Record results
    if passes_all_conditions:
        passed_matrices.append(matrix)
    else:
        failed_matrices.append(matrix)

print(f"{len(passed_matrices)} Classifiers passed the condition")
print(f"{len(failed_matrices)} Classifiers failed the condition")

passed_matrices = np.array(passed_matrices)# Convert to NumPy array
failed_matrices = np.array(failed_matrices)

passedfirst = 0
failedfirst = []

for matrix_index in range(len(goodclassifier_class)):  # Iterate over matrices
    matrix = -goodclassifier_class[matrix_index]  # Get the matrix at the current index
    passes_all_conditions = True  # Flag to track if the matrix passes all conditions

    for i in range(4):  # Iterate over rows of the matrix
        row = matrix[i]

        if not(row[0] + row[1] + -row[2] + -row[3] >= 0) and not(row[0] + row[1] + row[2] + row[3] <= 0): #condition
            passes_all_conditions = False
            break

    if passes_all_conditions:
        passedfirst += 1

    if not passes_all_conditions:
       failedfirst.append(matrix)

print(f"{passedfirst} Classifiers passed the condition")
print(f"{len(failedfirst)} Classifiers failed the condition")

failedfirst = np.array(failedfirst)

import numpy as np

def is_matrix_truly_extendable(matrix):
    # Iterate over each row of the matrix
    for row in matrix:
        a, b, c, d = row[0], row[1], row[2], row[3]

        l = np.max([-a, -b])
        r = np.min([c, d])

        # Check if there exists a value t in the range [l, r] in the current row
        if not any(l <= value <= r for value in row):
            print("POOKA")
            return False

        # Check if -min(a,b) <= min(c,d)
        if not (l <= np.min([c, d])):
            print("MIKA")
            return False

        # Check if -a <= t <= c and -b <= t <= d for all t in the row
        for t in row:
            if not (-a <= t <= c and -b <= t <= d):
                print("YOKA")
                return False

    return True

# Example usage
matrix = np.array([
    [1, 21, 21, 1],
    [1, 21, 21, 1],
    [-1, 23, 19, 3],
    [-1, 23, 19, 3]
])


print("Is the matrix truly extendable?", is_matrix_truly_extendable(matrix))



"""# Assumption Bin

positivity/negativity: nonextendables and most extendables must have at least one negative number
"""

positive = 0
nonnegative = 0
positive1 = 0
nonnegative1 = 0
positive2 = 0
nonnegative2 = 0

# Function to check if all values in a matrix are positive
def all_positive_nested(matrix):
    for row in matrix:
        for value in row:
            if value <= 0:
                return False
    return True

def nonNeg_nested(matrix):
    for row in matrix:
        for value in row:
            if value < 0:
                return False
    return True

# Check each extendable
for idx, matrix in enumerate(extendable_class):
    if all_positive_nested(matrix):
        positive += 1
    if nonNeg_nested(matrix):
      nonnegative += 1

# Check each nonextendable
for idx, matrix in enumerate(nonextendable_class):
    if all_positive_nested(matrix):
        positive1 += 1
    if nonNeg_nested(matrix):
      nonnegative1 += 1

# Check each classifier
for idx, matrix in enumerate(goodclassifier_class):
    if all_positive_nested(matrix):
        positive2 += 1
    if nonNeg_nested(matrix):
      nonnegative2 += 1

print(f"Number of positive extendable matrices: {positive}")
print(f"Number of nonnegative extendable matrices: {nonnegative}\n")

print(f"Number of nonnegative nonextendable matrices: {nonnegative1}")
print(f"Number of nonnegative classifiers: {nonnegative2}")

"""testing stuff idk"""

def count_matrices_with_condition(matrices):
    count = 0
    for matrix in matrices:
        all_rows_satisfy_condition = True
        for row in matrix:
            x, y, z, w = row[0], row[1], row[2], row[3]
            if not((-y + w < 0) or (z - x < 0)):
                all_rows_satisfy_condition = False
                break
        if all_rows_satisfy_condition:
            count += 1
    return count

count = count_matrices_with_condition(goodclassifier_class)
print(f"# of matrices where w < 0: {count}")

def count_matrices_with_condition(matrices):
    count = 0
    for matrix in matrices:
        for row in matrix:
            x, y, z, w = row[0], row[1], row[2], row[3]
            if ((-y + w < 0) or (z - x < 0)):
                count += 1
    return count

count = count_matrices_with_condition(goodclassifier_class)
print(f"# of rows where y < 0: {count}")

"""# positivity inheritance"""

from scipy.optimize import linprog

positive = []
nonpositive = []
mutes = []

# Function to check if all values in a matrix are positive
def all_positive_nested(matrix):
    for row in matrix:
        for value in row:
            if value < 0:
                return False
    return True

for matrix in extendable_class:
  bruh = np.dot(np.dot(INVtau, matrix), tau.T)
  mutes.append(bruh)

# Check each extendable
for idx, matrix in enumerate(mutes):
    if all_positive_nested(matrix):
        positive.append(matrix)
    else:
      nonpositive.append(matrix)

print(len(positive))
#print(nonpositive[0])
#print(positive[0])

from scipy.optimize import linprog

extendablemutations = 0
emutations = []
nmutations = []
failed = []

''
for matrix in extendable_class:
  matrix = np.dot(np.dot(INVtau, matrix), INVtau)
  emutations.append(matrix)
'''
for matrix in nonextendable_class:
  matrix = np.dot(tau, matrix)
  nmutations.append(matrix)
'''
for matrix in nmutations:
  mat = [1, 1, 1, 1]
  for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],]
  solve_for = [[min(matrix[0][2], matrix[0][3])],
             [min(matrix[1][2], matrix[1][3])],
             [min(matrix[2][2], matrix[2][3])],
             [min(matrix[3][2], matrix[3][3])],
             [-max(-matrix[0][0], -matrix[0][1])],
             [-max(-matrix[1][0], -matrix[1][1])],
             [-max(-matrix[2][0], -matrix[2][1])],
             [-max(-matrix[3][0], -matrix[3][1])]]
  for_A_eq = [[1,1,-1,-1]]
  for_b_eq = [0]

  result = linprog(c=mat, A_ub = for_A_ub, b_ub = solve_for, A_eq = for_A_eq, b_eq = for_b_eq, bounds = None)

  if (result.success):
    extendablemutations += 1

  else:
    failed.append(matrix)

print(extendablemutations, "extend")
print(len(failed), "don't extend")

Have_A_t = 0

for matrix in failed:
    all_rows_satisfy_condition = True

    for row in matrix:
        a, b, c, d = row[0], row[1], row[2], row[3]

        l = np.max([-a, -b])
        r = np.min([c, d])

        # Check if there exists a value t in the range [l, r] in the current row
        if not any(l <= value <= r for value in row):
            all_rows_satisfy_condition = False
            break  # No need to check further rows if this one doesn't satisfy

            # Check if -min(a,b) <= t <= max(c,d) holds for all t in the matrix
        if not (l <= min(c, d)):
            all_rows_satisfy_condition = False
            break

        # Check if -a, -b <= t <= c, d
        for t in row:
            if not (-a <= t <= c and -b <= t <= d):
                all_rows_satisfy_condition = False
                break

    if all_rows_satisfy_condition:
        Have_A_t += 1

print(Have_A_t, "of the (initially) failed matrices passed every test for extendability")

import numpy as np

def check_conditions(matrix):
    # Initialize conditions as True
    cond1 = cond2 = cond3 = True

    # Variables to store the sum of first two rows and the sum of last two rows
    sum_first_two_rows = np.sum(matrix[0]) + np.sum(matrix[1])
    sum_last_two_rows = np.sum(matrix[2]) + np.sum(matrix[3])

    for i in range(4):
        row = matrix[i]

        # Check condition 1
        if row[0] + row[1] != row[2] + row[3]:
            cond1 = False

        # Check condition 3
        if not (row[0] + row[2] > 0 and row[0] + row[3] > 0 and
                row[1] + row[2] > 0 and row[1] + row[3] > 0):
            cond3 = False

    # Check condition 2
    cond2 = (sum_first_two_rows == sum_last_two_rows)

    return cond1, cond2, cond3

# Initialize counters
passed_all_conditions = 0
failed_any_condition = 0

# Check each matrix in the list
for matrix in failed:
    result = check_conditions(matrix)
    if all(result):
        passed_all_conditions += 1
    else:
        failed_any_condition += 1

# Print results
print(f"Number of matrices that passed all conditions: {passed_all_conditions}")
print(f"Number of matrices that failed any condition: {failed_any_condition}")

"""POSITIVITY AFTER APPLYING INVERSE TAU TO EXTENDABLES"""

from scipy.optimize import linprog

# Load the provided .npy file
matrices = np.load("/content/ExtendableClass.npy")

# Define the tau_inverse function
def tau_inverse(a, b, c, d):
    return np.array([2*c-b, 2*d-a, 2*b-c, 2*a-d])

# Define the is_extendable function
def is_extendable(matrix):
    mat = [1, 1, 1, 1]
    for_A_ub = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ]
    solve_for = [
        min(matrix[0][2], matrix[0][3]),
        min(matrix[1][2], matrix[1][3]),
        min(matrix[2][2], matrix[2][3]),
        min(matrix[3][2], matrix[3][3]),
        -max(-matrix[0][0], -matrix[0][1]),
        -max(-matrix[1][0], -matrix[1][1]),
        -max(-matrix[2][0], -matrix[2][1]),
        -max(-matrix[3][0], -matrix[3][1])
    ]
    for_A_eq = [[1, 1, -1, -1]]
    for_b_eq = [0]

    result = linprog(c=mat, A_ub=for_A_ub, b_ub=solve_for, A_eq=for_A_eq, b_eq=for_b_eq, bounds=None)

    return result.success

# Function to process matrices
def process_matrices(matrices):
    newly_extendable = 0
    still_non_extendable = 0
    afterinversetau_extends = []

    for matrix in matrices:
        transformed_matrix = np.array([
            tau_inverse(matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]),
            tau_inverse(matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]),
            tau_inverse(matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]),
            tau_inverse(matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
        ])

        if is_extendable(transformed_matrix):
            newly_extendable += 1
            afterinversetau_extends.append(transformed_matrix)
        else:
            still_non_extendable += 1

    return newly_extendable, still_non_extendable, np.array(afterinversetau_extends)

# Process the matrices
newly_extendable, still_non_extendable, afterinversetau_extends = process_matrices(matrices)

# Print the results
print("Number of newly extendable matrices after Tau: ", newly_extendable)
print("Number of still non-extendable matrices after Tau: ", still_non_extendable)

"""the following is to check the degree of extendability"""

newly_extendable_second_pass, still_non_extendable_second_pass, afterinversetau_extends_second_pass = process_matrices(afterinversetau_extends)

# Print the results of the second pass
print("Number of newly extendable matrices after second pass: ", newly_extendable_second_pass)
print("Number of still non-extendable matrices after second pass: ", still_non_extendable_second_pass)

newly_extendable_third_pass, still_non_extendable_third_pass, afterinversetau_extends_third_pass = process_matrices(afterinversetau_extends_second_pass)

# Print the results of the second pass
print("Number of newly extendable matrices after third pass: ", newly_extendable_third_pass)
print("Number of still non-extendable matrices after third pass: ", still_non_extendable_third_pass)

newly_extendable_fourth_pass, still_non_extendable_fourth_pass, afterinversetau_extends_fourth_pass = process_matrices(afterinversetau_extends_third_pass)

# Print the results of the second pass
print("Number of newly extendable matrices after fourth pass: ", newly_extendable_fourth_pass)
print("Number of still non-extendable matrices after fourth pass: ", still_non_extendable_fourth_pass)

newly_extendable_fifth_pass, still_non_extendable_fifth_pass, afterinversetau_extends_fifth_pass = process_matrices(afterinversetau_extends_fourth_pass)

# Print the results of the second pass
print("Number of newly extendable matrices after fifth pass: ", newly_extendable_fifth_pass)
print("Number of still non-extendable matrices after fifth pass: ", still_non_extendable_fifth_pass)
np.save('afterinversetau_extends_fifth_pass.npy', afterinversetau_extends_fifth_pass)

newly_extendable_sixth_pass, still_non_extendable_sixth_pass, afterinversetau_extends_sixth_pass = process_matrices(afterinversetau_extends_fifth_pass)

# Print the results of the second pass
print("Number of newly extendable matrices after sixth pass: ", newly_extendable_sixth_pass)
print("Number of still non-extendable matrices after sixth pass: ", still_non_extendable_sixth_pass)
print(afterinversetau_extends_sixth_pass)

from scipy.optimize import linprog

# Load matrices
matrices = np.load("/content/ExtendableClass.npy")

# Define the tau_inverse function
def tau_inverse(a, b, c, d):
    return np.array([2*c-b, 2*d-a, 2*b-c, 2*a-d])

# Define the is_extendable function
def is_extendable(matrix):
    mat = [1, 1, 1, 1]
    for_A_ub = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ]
    solve_for = [
        [min(matrix[0][2], matrix[0][3])],
        [min(matrix[1][2], matrix[1][3])],
        [min(matrix[2][2], matrix[2][3])],
        [min(matrix[3][2], matrix[3][3])],
        [-max(-matrix[0][0], -matrix[0][1])],
        [-max(-matrix[1][0], -matrix[1][1])],
        [-max(-matrix[2][0], -matrix[2][1])],
        [-max(-matrix[3][0], -matrix[3][1])]
    ]
    for_A_eq = [[1, 1, -1, -1]]
    for_b_eq = [0]

    result = linprog(c=mat, A_ub=for_A_ub, b_ub=solve_for, A_eq=for_A_eq, b_eq=for_b_eq, bounds=None)

    return result.success

# Define the is_valid function
def is_valid(matrix):
    rows, cols = matrix.shape

    # Check row condition (check in E(2,2))
    for row in range(rows):
        if sum(matrix[row][:2]) != sum(matrix[row][2:]):
            return False

    # Check column condition (linearity check)
    for col in range(cols):
        if matrix[0][col] + matrix[1][col] != matrix[2][col] + matrix[3][col]:
            return False

    # Positivity check
    for row in range(rows):
        for i in range(4):
            for j in range(i+1, 4):
                if matrix[row][i] + matrix[row][j] < 0:
                    return False

    return True

# Function to process matrices
def process_matrices(matrices):
    newly_extendable = 0
    still_non_extendable = 0
    non_extendable_matrices = []

    for matrix in matrices:
        transformed_matrix = np.array([
            tau_inverse(matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]),
            tau_inverse(matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]),
            tau_inverse(matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]),
            tau_inverse(matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
        ])

        if is_extendable(transformed_matrix):
            newly_extendable += 1
        else:
            still_non_extendable += 1
            non_extendable_matrices.append(matrix)

    valid_non_extendable_matrices = [matrix for matrix in non_extendable_matrices if is_valid(matrix)]

    return newly_extendable, still_non_extendable, len(valid_non_extendable_matrices)

# Process the matrices
newly_extendable, still_non_extendable, valid_non_extendable = process_matrices(matrices)

# Print the results
print("Number of newly extendable matrices after Tau: ", newly_extendable)
print("Number of still non-extendable matrices after Tau: ", still_non_extendable)
print("Number of valid non-extendable matrices: ", valid_non_extendable)

import numpy as np
from scipy.optimize import linprog

# Load matrices
matrices = np.load("/content/ExtendableClass.npy")

# Define the new tau inverse matrix
INVtau = np.array([[2, -1, -1, 2],
                   [0, 1, 0, 1],
                   [1, 0, 1, 0],
                   [-1, 2, 2, -1]])

# Function to apply the tau inverse transformation
def apply_tau_inverse(matrix):
    return np.dot(INVtau, matrix)

# Define the is_extendable function
def is_extendable(matrix):
    mat = [1, 1, 1, 1]
    for_A_ub = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ]
    solve_for = [
        [min(matrix[0][2], matrix[0][3])],
        [min(matrix[1][2], matrix[1][3])],
        [min(matrix[2][2], matrix[2][3])],
        [min(matrix[3][2], matrix[3][3])],
        [-max(-matrix[0][0], -matrix[0][1])],
        [-max(-matrix[1][0], -matrix[1][1])],
        [-max(-matrix[2][0], -matrix[2][1])],
        [-max(-matrix[3][0], -matrix[3][1])]
    ]
    for_A_eq = [[1, 1, -1, -1]]
    for_b_eq = [0]

    result = linprog(c=mat, A_ub=for_A_ub, b_ub=solve_for, A_eq=for_A_eq, b_eq=for_b_eq, bounds=None)

    return result.success

# Define the is_valid function
def is_valid(matrix):
    rows, cols = matrix.shape

    # Check row condition (check in E(2,2))
    for row in range(rows):
        if sum(matrix[row][:2]) != sum(matrix[row][2:]):
            return False

    # Check column condition (linearity check)
    for col in range(cols):
        if matrix[0][col] + matrix[1][col] != matrix[2][col] + matrix[3][col]:
            return False

    # Positivity check
    for row in range(rows):
        for i in range(4):
            for j in range(i+1, 4):
                if matrix[row][i] + matrix[row][j] < 0:
                    return False

    return True

# Function to process matrices
def process_matrices(matrices):
    still_extendable = 0
    newly_non_extendable = 0
    failed_tests = []

    # Apply tau inverse transformation
    transformed_matrices = np.array([apply_tau_inverse(matrix) for matrix in matrices])

    # Check extendability
    for matrix in transformed_matrices:
        if is_extendable(matrix):
            still_extendable += 1
        else:
            newly_non_extendable += 1
            failed_tests.append(matrix)

    # Check validity for non-extendable matrices
    failed_tests = np.array(failed_tests)
    valid_non_extendable_matrices = [matrix for matrix in failed_tests if is_valid(matrix)]

    return still_extendable, newly_non_extendable, len(valid_non_extendable_matrices)

# Process the matrices
still_extendable, newly_non_extendable, valid_non_extendable = process_matrices(matrices)

# Print the results
print("Number of still extendable matrices after Tau: ", still_extendable)
print("Number of newly non-extendable matrices after Tau: ", newly_non_extendable)
print("Number of valid non-extendable matrices: ", valid_non_extendable)

import numpy as np
from scipy.optimize import linprog

# Load matrices
matrices = np.load("/content/ExtendableClass.npy")

# Define the new tau inverse matrix
INVtau = np.array([[2, -1, -1, 2],
                   [0, 1, 0, 1],
                   [1, 0, 1, 0],
                   [-1, 2, 2, -1]])

# Function to apply the tau inverse transformation
def apply_tau_inverse(matrix):
    return np.dot(matrix, INVtau)

# Define the is_extendable function
def is_extendable(matrix):
    mat = [1, 1, 1, 1]
    for_A_ub = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ]
    solve_for = [
        [min(matrix[0][2], matrix[0][3])],
        [min(matrix[1][2], matrix[1][3])],
        [min(matrix[2][2], matrix[2][3])],
        [min(matrix[3][2], matrix[3][3])],
        [-max(-matrix[0][0], -matrix[0][1])],
        [-max(-matrix[1][0], -matrix[1][1])],
        [-max(-matrix[2][0], -matrix[2][1])],
        [-max(-matrix[3][0], -matrix[3][1])]
    ]
    for_A_eq = [[1, 1, -1, -1]]
    for_b_eq = [0]

    result = linprog(c=mat, A_ub=for_A_ub, b_ub=solve_for, A_eq=for_A_eq, b_eq=for_b_eq, bounds=None)

    return result.success

# Define the is_valid function
def is_valid(matrix):
    rows, cols = matrix.shape

    # Check row condition (check in E(2,2))
    for row in range(rows):
        if sum(matrix[row][:2]) != sum(matrix[row][2:]):
            return False

    # Check column condition (linearity check)
    for col in range(cols):
        if matrix[0][col] + matrix[1][col] != matrix[2][col] + matrix[3][col]:
            return False

    # Positivity check
    for row in range(rows):
        for i in range(4):
            for j in range(i+1, 4):
                if matrix[row][i] + matrix[row][j] < 0:
                    return False

    return True

# Function to process matrices
def process_matrices(matrices):
    still_extendable = 0
    newly_non_extendable = 0
    failed_tests = []

    # Apply tau inverse transformation
    transformed_matrices = np.array([apply_tau_inverse(matrix) for matrix in matrices])

    # Check extendability
    for matrix in transformed_matrices:
        if is_extendable(matrix):
            still_extendable += 1
        else:
            newly_non_extendable += 1
            failed_tests.append(matrix)

    # Check validity for non-extendable matrices
    failed_tests = np.array(failed_tests)
    valid_non_extendable_matrices = [matrix for matrix in failed_tests if is_valid(matrix)]

    return still_extendable, newly_non_extendable, len(valid_non_extendable_matrices)

# Process the matrices
still_extendable, newly_non_extendable, valid_non_extendable = process_matrices(matrices)

# Print the results
print("Number of still extendable matrices after Tau: ", still_extendable)
print("Number of newly non-extendable matrices after Tau: ", newly_non_extendable)
print("Number of valid non-extendable matrices: ", valid_non_extendable)

"""# mutations"""

import numpy as np
from scipy.optimize import linprog

mutations = []
extendablemutations = []
nonextendablemutations = []

for matrix in extendable_class:
  matrix = np.dot(tau, matrix)
  mutations.append(matrix)

for matrix in mutations:
  mat = [1, 1, 1, 1]
  for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],]
  solve_for = [[min(matrix[0][2], matrix[0][3])],
             [min(matrix[1][2], matrix[1][3])],
             [min(matrix[2][2], matrix[2][3])],
             [min(matrix[3][2], matrix[3][3])],
             [-max(-matrix[0][0], -matrix[0][1])],
             [-max(-matrix[1][0], -matrix[1][1])],
             [-max(-matrix[2][0], -matrix[2][1])],
             [-max(-matrix[3][0], -matrix[3][1])]]
  for_A_eq = [[1,1,-1,-1]]
  for_b_eq = [0]

  result = linprog(c=mat, A_ub = for_A_ub, b_ub = solve_for, A_eq = for_A_eq, b_eq = for_b_eq, bounds = None)

  if (result.success):
    extendablemutations.append(matrix)

  else:
    nonextendablemutations.append(matrix)

print(len(extendablemutations), "extend")
print(len(nonextendablemutations), "don't extend")

Have_A_t = 0

for matrix in extendablemutations:
    all_rows_satisfy_condition = True

    for row in matrix:
        a, b, c, d = row[0], row[1], row[2], row[3]

        l = np.max([-a, -b])
        r = np.min([c, d])

        # Check if there exists a value t in the range [l, r] in the current row
        if not any(l <= value <= r for value in row):
            all_rows_satisfy_condition = False
            break  # No need to check further rows if this one doesn't satisfy

            # Check if -min(a,b) <= t <= max(c,d) holds for all t in the matrix
        if not (l <= min(c, d)):
            all_rows_satisfy_condition = False
            break

        # Check if -a, -b <= t <= c, d
        for t in row:
            if not (-a <= t <= c and -b <= t <= d):
                all_rows_satisfy_condition = False
                break

    if all_rows_satisfy_condition:
        Have_A_t += 1

print(Have_A_t, "of the (initially) failed matrices passed every test for extendability")

"""**limits with all taus on right side**"""

matrix = extendable_class[0]

onetransform = np.dot(matrix, tau)
twotransform = np.dot(np.dot(matrix, tau), tau)
threetransform = np.dot(np.dot(np.dot(matrix, tau),tau),tau)
fourtransform = np.dot(np.dot(np.dot(np.dot(matrix, tau),tau),tau), tau)
fivetransform = np.dot(np.dot(np.dot(np.dot(np.dot(matrix, tau),tau),tau), tau), tau)

print(matrix)
print(onetransform)
print(twotransform)
print(threetransform)
print(fourtransform)
print(fivetransform)

"""**tau inverse**"""

matrix = extendable_class[0]

onetransform = np.dot(matrix, INVtau)
twotransform = np.dot(np.dot(matrix, INVtau), INVtau)
threetransform = np.dot(np.dot(np.dot(matrix, INVtau), INVtau), INVtau)
fourtransform = np.dot(np.dot(np.dot(np.dot(matrix, INVtau), INVtau), INVtau), INVtau)
fivetransform = np.dot(np.dot(np.dot(np.dot(np.dot(matrix, INVtau), INVtau), INVtau), INVtau), INVtau)

print(matrix)
print(onetransform)
print(twotransform)
print(threetransform)
print(fourtransform)
print(fivetransform)

"""**limits with all taus on the left side**"""

matrix = nonextendable_class[0]

onetransform = np.dot(tau, matrix)
twotransform = np.dot(tau, np.dot(tau, matrix))
threetransform = np.dot(tau, np.dot(tau, np.dot(tau, matrix)))
fourtransform = np.dot(tau, np.dot(tau, np.dot(tau, np.dot(tau, matrix))))
fivetransform = np.dot(tau, np.dot(tau, np.dot(tau, np.dot(tau, np.dot(tau, matrix)))))

print(matrix)
print(onetransform)
print(twotransform)
print(threetransform)
print(fourtransform)
print(fivetransform)

"""**tau inverse**"""

matrix = extendable_class[0]

onetransform = np.dot(INVtau, matrix)
twotransform = np.dot(INVtau, np.dot(INVtau, matrix))
threetransform = np.dot(INVtau, np.dot(INVtau, np.dot(INVtau, matrix)))
fourtransform = np.dot(INVtau, np.dot(INVtau, np.dot(INVtau, np.dot(INVtau, matrix))))
fivetransform = np.dot(INVtau, np.dot(INVtau, np.dot(INVtau, np.dot(INVtau, np.dot(INVtau, matrix)))))

print(matrix)
print(onetransform)
print(twotransform)
print(threetransform)
print(fourtransform)
print(fivetransform)

"""# Conjecture test"""

import numpy as np
from scipy.optimize import linprog

def tau(a, b, c, d):
    return np.array([(a+d)/2, (b+c)/2, (a+c)/2, (b+d)/2])

def psi(a, b, c, d): # = 0.5*idc
  return np.array([(a+b)/2, (b+a)/2, (d+c)/2, (c+d)/2])

def tau_inverse(a, b, c, d):
    return np.array([2*c - b, 2*d - a, 2*b - c, 2*a - d])

# List to store transformed matrices
tauNonextends = []
failedtests = []
newextends = []
newnunextends = []

# Apply tau to each row in nonextendable_clss and append to tauNonextends
for row in nonextendable_class:
    transformed_row = tau(*row)
    tauNonextends.append(transformed_row)

# Convert tauNonextends to a numpy array if needed
tauNonextends = np.array(tauNonextends)

newlyExtendable = 0
stillnonExtendable = 0

for matrix in tauNonextends:
  mat = [1, 1, 1, 1]
  for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],]
  solve_for = [[min(matrix[0][2], matrix[0][3])],
             [min(matrix[1][2], matrix[1][3])],
             [min(matrix[2][2], matrix[2][3])],
             [min(matrix[3][2], matrix[3][3])],
             [-max(-matrix[0][0], -matrix[0][1])],
             [-max(-matrix[1][0], -matrix[1][1])],
             [-max(-matrix[2][0], -matrix[2][1])],
             [-max(-matrix[3][0], -matrix[3][1])]]
  for_A_eq = [[1,1,-1,-1]]
  for_b_eq = [0]

  result = linprog(c=mat, A_ub = for_A_ub, b_ub = solve_for, A_eq = for_A_eq, b_eq = for_b_eq, bounds = None)

  if (result.success):
    newlyExtendable += 1
    newextends.append(matrix)

  else:
    stillnonExtendable += 1
    failedtests.append(matrix)

failedtests = np.array(failedtests)
newextends = np.array(newextends)

Have_A_t = 0

for matrix in failedtests:
    all_rows_satisfy_condition = True

    for row in matrix:
        a, b, c, d = row[0], row[1], row[2], row[3]

        l = np.max([-a, -b])
        r = np.min([c, d])

        # Check if there exists a value t in the range [l, r] in the current row
        if not any(l <= value <= r for value in row):
            all_rows_satisfy_condition = False
            break  # No need to check further rows if this one doesn't satisfy

            # Check if -min(a,b) <= t <= max(c,d) holds for all t in the matrix
        if not (l <= max(c, d)):
            all_rows_satisfy_condition = False
            break

        # Check if -a, -b <= t <= c, d
        for t in row:
            if not (-a <= t <= c and -b <= t <= d):
                all_rows_satisfy_condition = False
                break

    if all_rows_satisfy_condition:
        Have_A_t += 1

print(Have_A_t, "of the (initially) failed matrices passed every test for extendability")


print("newly extendable", newlyExtendable)
print("still not extendable", stillnonExtendable)
print("Number of failed tests: ",len(failedtests));

for matrix in failedtests:
  print(matrix)

# print most occuring rows in failed tests

import numpy as np
from collections import Counter


# Flatten all rows from all matrices in failedtests into a single list
all_rows = []
for matrix in failedtests:
    for row in matrix:
        all_rows.append(tuple(row))  # Convert row to a tuple to make it hashable

# Count the occurrences of each row
row_counter = Counter(all_rows)

# Get the 20 most common rows
most_common_rows = row_counter.most_common(20)

# Print the 20 most common rows and their counts
for row, count in most_common_rows:
    print(f"Row: {row}, Count: {count}")


#Note to self:
#1. Its evident that these rows atleast have one non-positive numbers which we had already discovered previously as
#   common characteristic of all non-extends
#2. What makes the the matrices extendable or non extendable after applying tau depends on the biggest positve
#   and negative numbers. Tau essentially finds average of 2 numbers. And if x > y, and |x| > |y| the tau always passes, otherwise it fails

# checking identical element in each row of orignal extendable lists
count_similar_rows_ori_extendable = 0

for matrix in extendable_class:
    for row in matrix:
        if np.all(row == row[0]):
            count_similar_rows_ori_extendable += 1
            break
print(len(extendable_class), "total number of extendable matrices")
print(count_similar_rows_ori_extendable, "matrices have at least one row with all identical elements")

# checking identical element in each row of orignal extendable lists
count_similar_rows_ori_nonextendable = 0

for matrix in nonextendable_class:
    for row in matrix:
        if np.all(row == row[0]):
            count_similar_rows_ori_nonextendable += 1
            break
print(len(nonextendable_class), "total number of extendable matrices")
print(count_similar_rows_ori_nonextendable, "matrices have at least one row with all identical elements")

"""darsh check this"""

newly_extendable = 0
still_non_extendable = 0

def is_extendable(matrix):
    first_row = matrix[0]
    a, b, c, d = first_row[0], first_row[1], first_row[2], first_row[3]
    l = -min(a, b)
    r = max(c, d)

    # Check if -min(a,b) <= t <= max(c,d) holds for all t in the matrix
    if not (l <= max(c, d)):
        return False

    # Check if -a, -b <= t <= c, d
    for t in first_row:
        if not (-a <= t <= c and -b <= t <= d):
            return False

    return True

failedextendtest = []

for matrix in failedtests:
    if is_extendable(matrix):
        newly_extendable += 1
        failedextendtest.append(matrix)
    else:
        still_non_extendable += 1

failedextendtest = np.array(failedextendtest)
print("Failed test is now extendable: ", newly_extendable)
print("Failed test is still non extendable: ", still_non_extendable)
print("Extendable check matrices that passes \n", failedextendtest)


#We see that there is one row with all similar numbers

"""applying tau first and then the mappings"""

import numpy as np
from scipy.optimize import linprog

tau = np.array([[1, 0, 0.5, 0.5],
                [0, 1, 0.5, 0.5],
                [0.5, 0.5, 1, 0],
                [0.5, 0.5, 0, 1]])

#psi = np.array()

# Perform matrix multiplication
multiplied_matrices = [np.dot(tau, matrix) for matrix in nonextendable_class]
multiplied_matrices = np.array(multiplied_matrices)

newlyExtendable = 0
stillnonExtendable = 0
failedTAU = []

for matrix in multiplied_matrices:
  mat = [1, 1, 1, 1]
  for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],]
  solve_for = [[min(matrix[0][2], matrix[0][3])],
             [min(matrix[1][2], matrix[1][3])],
             [min(matrix[2][2], matrix[2][3])],
             [min(matrix[3][2], matrix[3][3])],
             [-max(-matrix[0][0], -matrix[0][1])],
             [-max(-matrix[1][0], -matrix[1][1])],
             [-max(-matrix[2][0], -matrix[2][1])],
             [-max(-matrix[3][0], -matrix[3][1])]]
  for_A_eq = [[1,1,-1,-1]]
  for_b_eq = [0]

  result = linprog(c=mat, A_ub = for_A_ub, b_ub = solve_for, A_eq = for_A_eq, b_eq = for_b_eq, bounds = None)

  if (result.success):
    newlyExtendable += 1
  else:
    stillnonExtendable += 1
    failedTAU.append(matrix)

print("newly extendable", newlyExtendable)
print("still not extendable", stillnonExtendable)
failedTAU = np.array(failedTAU)

#COMBINE ALL MATRICES

import numpy as np
extendable_matrices = np.load("/content/extendableMappings.npy")
nonextendable_matrices = np.load("/content/nonExtendableMappings.npy")

# Combine the matrices into a single list
combined_matrices = np.concatenate((extendable_class, nonextendable_class), axis=0)

# Save the combined list of matrices to a new .npy file
np.save('/content/combinedMappings.npy', combined_matrices)

# Print the number of matrices in each category and in total
print("Number of extendable matrices: ", len(extendable_matrices))
print("Number of non-extendable matrices: ", len(nonextendable_matrices))
print("Total number of combined matrices: ", len(combined_matrices))

"""APPLYING TAU TO MATRIX"""

#MATRIX TIMES TAU

import numpy as np
from scipy.optimize import linprog

def tau(a, b, c, d):
    return np.array([(a+d)/2, (b+c)/2, (a+c)/2, (b+d)/2])

matrices = np.load("/content/combinedMappings.npy")

def is_extendable(matrix):
    mat = [1, 1, 1, 1]
    for_A_ub = [[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1],
                [-1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,-1]]
    solve_for = [[min(matrix[0][2], matrix[0][3])],
                 [min(matrix[1][2], matrix[1][3])],
                 [min(matrix[2][2], matrix[2][3])],
                 [min(matrix[3][2], matrix[3][3])],
                 [-max(-matrix[0][0], -matrix[0][1])],
                 [-max(-matrix[1][0], -matrix[1][1])],
                 [-max(-matrix[2][0], -matrix[2][1])],
                 [-max(-matrix[3][0], -matrix[3][1])]]
    for_A_eq = [[1,1,-1,-1]]
    for_b_eq = [0]

    result = linprog(c=mat, A_ub=for_A_ub, b_ub=solve_for, A_eq=for_A_eq, b_eq=for_b_eq, bounds=None)

    return result.success

def process_matrices(matrices):
    newly_extendable = 0
    still_non_extendable = 0
    extendabletau_matrices = []
    non_extendabletau_matrices = []

    for matrix in matrices:
        transformed_matrix = np.array([
            tau(matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]),
            tau(matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]),
            tau(matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]),
            tau(matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
        ])
        if is_extendable(transformed_matrix):
            newly_extendable += 1
            extendabletau_matrices.append(transformed_matrix)
        else:
            still_non_extendable += 1
            non_extendabletau_matrices.append(transformed_matrix)

    return extendabletau_matrices, non_extendabletau_matrices, newly_extendable, still_non_extendable

extendabletau_matrices, non_extendabletau_matrices, newly_extendable, still_non_extendable = process_matrices(matrices)

np.save('ExtendableMatricesTau.npy', extendabletau_matrices)
np.save('NonExtendableMatricesTau.npy', non_extendabletau_matrices)

print("Number of newly extendable matrices after Tau: ", newly_extendable)
print("Number of still non-extendable matrices after Tau: ", still_non_extendable)

#TAU TIMES MATRIX

import numpy as np
from scipy.optimize import linprog

# Define tau as a 4x4 matrix

tau = np.array([[0.5, 0, 0.5, 0],
                [0, 0.5, 0, 0.5],
                [0, 0.5, 0.5, 0],
                [0.5, 0, 0, 0.5]])


# Function to check extendability
def is_extendable(matrix):
    mat = [1, 1, 1, 1]
    for_A_ub = [[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1],
                [-1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,-1]]
    solve_for = [[min(matrix[0][2], matrix[0][3])],
                 [min(matrix[1][2], matrix[1][3])],
                 [min(matrix[2][2], matrix[2][3])],
                 [min(matrix[3][2], matrix[3][3])],
                 [-max(-matrix[0][0], -matrix[0][1])],
                 [-max(-matrix[1][0], -matrix[1][1])],
                 [-max(-matrix[2][0], -matrix[2][1])],
                 [-max(-matrix[3][0], -matrix[3][1])]]
    for_A_eq = [[1,1,-1,-1]]
    for_b_eq = [0]

    result = linprog(c=mat, A_ub=for_A_ub, b_ub=solve_for, A_eq=for_A_eq, b_eq=for_b_eq, bounds=None)

    return result.success

# Function to process matrices by multiplying with tau matrix
def process_matrices_with_tau_multiplication(tau, matrix):
    newly_extendable = 0
    still_non_extendable = 0
    extendabletau_matrices = []
    non_extendabletau_matrices = []

    for matrix in matrices:
        transformed_matrix = np.dot(tau, matrix)

        if is_extendable(transformed_matrix):
            newly_extendable += 1
            extendabletau_matrices.append(transformed_matrix)
        else:
            still_non_extendable += 1
            non_extendabletau_matrices.append(transformed_matrix)

    return extendabletau_matrices, non_extendabletau_matrices, newly_extendable, still_non_extendable

# Load the matrix data
matrices = np.load("/content/combinedMappings.npy")
extendabletau_matrices, non_extendabletau_matrices, newly_extendable, still_non_extendable = process_matrices_with_tau_multiplication(tau, matrices)

# Print results
print("Number of newly extendable matrices after multiplying with Tau: ", newly_extendable)
print("Number of still non-extendable matrices after multiplying with Tau: ", still_non_extendable)

# CHECK IF THE TAU TIMES MATRIX HAS POSITIVE MAPPINGS

def is_valid(matrix, rows, cols, row, col, num):
    original_value = matrix[row][col]
    matrix[row][col] = num

    # Check row condition (sum of the first two equals the sum of the last two in each row)
    if col == 3:
        if sum(matrix[row][:2]) != matrix[row][2] + num:
            matrix[row][col] = original_value
            return False

    # Check column condition (sum of the 1st and 2nd rows equal the 3rd and 4th)
    if row == 3:
        if matrix[0][col] + matrix[1][col] != matrix[2][col] + num:
            matrix[row][col] = original_value
            return False

    # Positivity check (sum of any element added to any other element is positive)
    if col == 3:
        matrix[row][col] = num
        if matrix[row][0] + matrix[row][2] < 0:
            matrix[row][col] = original_value
            return False
        if matrix[row][0] + matrix[row][3] < 0:
            matrix[row][col] = original_value
            return False
        if matrix[row][1] + matrix[row][2] < 0:
            matrix[row][col] = original_value
            return False
        if matrix[row][1] + matrix[row][3] < 0:
            matrix[row][col] = original_value
            return False

    matrix[row][col] = original_value
    return True

# Checking the positivity test for each extendable matrix after inverse Tau
valid_matrices = []
invalid_matrices = []

for matrix in non_extendabletau_matrices:
    valid = True
    rows, cols = matrix.shape
    # Iterate over each element in the matrix and test validity
    for row in range(rows):
        for col in range(cols):
            if not is_valid(matrix, rows, cols, row, col, matrix[row][col]):
                valid = False
                break
        if not valid:
            break

    if valid:
        valid_matrices.append(matrix)
    else:
        invalid_matrices.append(matrix)

print(f"Number of matrices passing the positivity test: {len(valid_matrices)}")
print(f"Number of matrices failing the positivity test: {len(invalid_matrices)}")

print(non_extendabletau_matrices[0])

# CHECK THE ORIGINAL MATRIX

import numpy as np

# Define the tau inverse matrix
tau_inverse_matrix = np.array([
    [3/4, -1/4, 3/4, -1/4],
    [-1/4, 3/4, -1/4, 3/4],
    [-1/4, 3/4, 3/4, -1/4],
    [3/4, -1/4, -1/4, 3/4]
])

matrix_to_inverse = non_extendabletau_matrices[0]
print("Original Matrix (Non-Extendable):\n", matrix_to_inverse)

inversed_matrix = np.dot(tau_inverse_matrix, matrix_to_inverse)
print("Inversed Matrix:\n", inversed_matrix)

import numpy as np
from scipy.optimize import linprog

# inverse tau
tau_inverse_matrix = np.array([
    [3/4, -1/4, 3/4, -1/4],
    [-1/4, 3/4, -1/4, 3/4],
    [-1/4, 3/4, 3/4, -1/4],
    [3/4, -1/4, -1/4, 3/4]
])

def apply_inverse_tau(matrix):
    transformed_matrix = np.zeros_like(matrix)
    for i in range(4):
        transformed_matrix[i] = np.dot(matrix[i], tau_inverse_matrix)
    return transformed_matrix

def is_extendable(matrix):
    mat = [1, 1, 1, 1]
    for_A_ub = [[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1],
                [-1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,-1]]
    solve_for = [[min(matrix[0][2], matrix[0][3])],
                 [min(matrix[1][2], matrix[1][3])],
                 [min(matrix[2][2], matrix[2][3])],
                 [min(matrix[3][2], matrix[3][3])],
                 [-max(-matrix[0][0], -matrix[0][1])],
                 [-max(-matrix[1][0], -matrix[1][1])],
                 [-max(-matrix[2][0], -matrix[2][1])],
                 [-max(-matrix[3][0], -matrix[3][1])]]
    for_A_eq = [[1,1,-1,-1]]
    for_b_eq = [0]

    result = linprog(c=mat, A_ub=for_A_ub, b_ub=solve_for, A_eq=for_A_eq, b_eq=for_b_eq, bounds=None)

    return result.success

inverse_transformed_matrices = []
extendable_matrices_after_inverse_tau = []
non_extendable_matrices_after_inverse_tau = []

newly_extendable_after_inverse_tau = 0
still_non_extendable_after_inverse_tau = 0

for matrix in matrices:
    inverse_transformed_matrix = apply_inverse_tau(matrix)
    inverse_transformed_matrices.append(inverse_transformed_matrix)

    if is_extendable(inverse_transformed_matrix):
        newly_extendable_after_inverse_tau += 1
        extendable_matrices_after_inverse_tau.append(inverse_transformed_matrix)
    else:
        still_non_extendable_after_inverse_tau += 1
        non_extendable_matrices_after_inverse_tau.append(inverse_transformed_matrix)

print(f"Total: {len(inverse_transformed_matrices)}")
print(f"Number of extendable matrices after inverse Tau: {newly_extendable_after_inverse_tau}")
print(f"Number of non-extendable matrices after inverse Tau: {still_non_extendable_after_inverse_tau}")

import numpy as np
from scipy.optimize import linprog

# inverse tau
tau_inverse_matrix = np.array([
    [3/4, -1/4, 3/4, -1/4],
    [3/4, -1/4, -1/4, 3/4],
    [-1/4, 3/4, 3/4, -1/4],
    [-1/4, 3/4, -1/4, 3/4]
])

def apply_inverse_tau(matrix):
    transformed_matrix = np.zeros_like(matrix)
    for i in range(4):
        transformed_matrix[i] = np.dot(tau_inverse_matrix, matrix[i])
    return transformed_matrix

def is_extendable(matrix):
    mat = [1, 1, 1, 1]
    for_A_ub = [[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1],
                [-1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,-1]]
    solve_for = [[min(matrix[0][2], matrix[0][3])],
                 [min(matrix[1][2], matrix[1][3])],
                 [min(matrix[2][2], matrix[2][3])],
                 [min(matrix[3][2], matrix[3][3])],
                 [-max(-matrix[0][0], -matrix[0][1])],
                 [-max(-matrix[1][0], -matrix[1][1])],
                 [-max(-matrix[2][0], -matrix[2][1])],
                 [-max(-matrix[3][0], -matrix[3][1])]]
    for_A_eq = [[1,1,-1,-1]]
    for_b_eq = [0]

    result = linprog(c=mat, A_ub=for_A_ub, b_ub=solve_for, A_eq=for_A_eq, b_eq=for_b_eq, bounds=None)

    return result.success

inverse_transformed_matrices = []
extendable_matrices_after_inverse_tau = []
non_extendable_matrices_after_inverse_tau = []

newly_extendable_after_inverse_tau = 0
still_non_extendable_after_inverse_tau = 0

for matrix in matrices:
    inverse_transformed_matrix = apply_inverse_tau(matrix)
    inverse_transformed_matrices.append(inverse_transformed_matrix)

    if is_extendable(inverse_transformed_matrix):
        newly_extendable_after_inverse_tau += 1
        extendable_matrices_after_inverse_tau.append(inverse_transformed_matrix)
    else:
        still_non_extendable_after_inverse_tau += 1
        non_extendable_matrices_after_inverse_tau.append(inverse_transformed_matrix)

print(f"Number of newly extendable matrices processed with inverse Tau: {len(inverse_transformed_matrices)}")
print(f"Number of extendable matrices after inverse Tau: {newly_extendable_after_inverse_tau}")
print(f"Number of non-extendable matrices after inverse Tau: {still_non_extendable_after_inverse_tau}")

np.save('ExtendableMatricesAfterInverseTau.npy', extendable_matrices_after_inverse_tau)
np.save('NonExtendableMatricesAfterInverseTau.npy', non_extendable_matrices_after_inverse_tau)

import numpy as np
from scipy.optimize import linprog

# inverse tau
tau_inverse_matrix = np.array([
    [3/4, -1/4, 3/4, -1/4],
    [-1/4, 3/4, -1/4, 3/4],
    [-1/4, 3/4, 3/4, -1/4],
    [3/4, -1/4, -1/4, 3/4]
])

matrices = np.load("/content/combinedMappings.npy")

def apply_inverse_tau(matrix):
    transformed_matrix = np.zeros_like(matrix)
    for i in range(4):
        transformed_matrix[i] = np.dot(matrix[i], tau_inverse_matrix.T)
    return transformed_matrix

def is_extendable(matrix):
    mat = [1, 1, 1, 1]
    for_A_ub = [[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1],
                [-1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,-1]]
    solve_for = [[min(matrix[0][2], matrix[0][3])],
                 [min(matrix[1][2], matrix[1][3])],
                 [min(matrix[2][2], matrix[2][3])],
                 [min(matrix[3][2], matrix[3][3])],
                 [-max(-matrix[0][0], -matrix[0][1])],
                 [-max(-matrix[1][0], -matrix[1][1])],
                 [-max(-matrix[2][0], -matrix[2][1])],
                 [-max(-matrix[3][0], -matrix[3][1])]]
    for_A_eq = [[1,1,-1,-1]]
    for_b_eq = [0]

    result = linprog(c=mat, A_ub=for_A_ub, b_ub=solve_for, A_eq=for_A_eq, b_eq=for_b_eq, bounds=None)

    return result.success

inverse_transformed_matrices = []
extendable_matrices_after_inverse_tau = []
non_extendable_matrices_after_inverse_tau = []

newly_extendable_after_inverse_tau = 0
still_non_extendable_after_inverse_tau = 0

for matrix in matrices:
    inverse_transformed_matrix = apply_inverse_tau(matrix)
    inverse_transformed_matrices.append(inverse_transformed_matrix)

print(f"All matrices after inverse: {len(inverse_transformed_matrices)}")
np.save('InverseofBigCone.npy', inverse_transformed_matrices)

def is_valid(matrix, rows, cols, row, col, num):
    original_value = matrix[row][col]
    matrix[row][col] = num

    # Check row condition (sum of the first two equals the sum of the last two in each row)
    if col == 3:
        if sum(matrix[row][:2]) != matrix[row][2] + num:
            matrix[row][col] = original_value
            return False

    # Check column condition (sum of the 1st and 2nd rows equal the 3rd and 4th)
    if row == 3:
        if matrix[0][col] + matrix[1][col] != matrix[2][col] + num:
            matrix[row][col] = original_value
            return False

    # Positivity check (sum of any element added to any other element is positive)
    if col == 3:
        matrix[row][col] = num
        if matrix[row][0] + matrix[row][2] < 0:
            matrix[row][col] = original_value
            return False
        if matrix[row][0] + matrix[row][3] < 0:
            matrix[row][col] = original_value
            return False
        if matrix[row][1] + matrix[row][2] < 0:
            matrix[row][col] = original_value
            return False
        if matrix[row][1] + matrix[row][3] < 0:
            matrix[row][col] = original_value
            return False

    matrix[row][col] = original_value
    return True

# Checking the positivity test for each extendable matrix after inverse Tau
valid_matrices = []
invalid_matrices = []

for matrix in inverse_transformed_matrices:
    valid = True
    rows, cols = matrix.shape
    # Iterate over each element in the matrix and test validity
    for row in range(rows):
        for col in range(cols):
            if not is_valid(matrix, rows, cols, row, col, matrix[row][col]):
                valid = False
                break
        if not valid:
            break

    if valid:
        valid_matrices.append(matrix)
    else:
        invalid_matrices.append(matrix)

print(f"Number of matrices passing the positivity test: {len(valid_matrices)}")
print(f"Number of matrices failing the positivity test: {len(invalid_matrices)}")
np.save('positiveafterinversetau.npy', valid_matrices)

positive_and_extendable = 0
positive_and_nonextend = 0

for matrix in valid_matrices:
    if is_extendable(matrix):
        positive_and_extendable += 1
    else:
        positive_and_nonextend += 1

print(f"Number of extendables in positive: {positive_and_extendable}")
print(f"Number of Nonextendables in positive: {positive_and_nonextend}")

inverse_transformed_extendable_matrices = []
valid_matrices2 = []
invalid_matrices2 = []
extendable_after_inverse_tau = 0
still_non_extendable_after_inverse_tau = 0

extendable_matrices = np.load("/content/extendableMappings.npy")

tau_inverse_matrix = np.array([
    [3/4, -1/4, 3/4, -1/4],
    [-1/4, 3/4, -1/4, 3/4],
    [-1/4, 3/4, 3/4, -1/4],
    [3/4, -1/4, -1/4, 3/4]
])

def apply_inverse_tau(matrix):
    transformed_matrix = np.zeros_like(matrix)
    for i in range(4):
        transformed_matrix[i] = np.dot(matrix[i], tau_inverse_matrix.T)
    return transformed_matrix

for matrix in extendable_matrices:
    inverse_transformed_extendable_matrix = apply_inverse_tau(matrix)
    inverse_transformed_extendable_matrices.append(inverse_transformed_extendable_matrix)

for matrix in inverse_transformed_extendable_matrices:
    valid = True
    rows, cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            if not is_valid(matrix, rows, cols, row, col, matrix[row][col]):
                valid = False
                break
        if not valid:
            break

    if valid:
        valid_matrices2.append(matrix)
    else:
        invalid_matrices2.append(matrix)

print(f"Positive in extendable matrices: {len(valid_matrices2)}")
print(f"Non-positive in extendable matrices: {len(invalid_matrices2)}")


for matrix in valid_matrices2:
    if is_extendable(matrix):
        extendable_after_inverse_tau += 1
    else:
        still_non_extendable_after_inverse_tau += 1

print(f"Extendable after positve after inverse tau of extendables: {extendable_after_inverse_tau}")
print(f"Non Extendable after positve after inverse tau of extendables: {still_non_extendable_after_inverse_tau}")
np.save('Nonpositivebutextends.npy', invalid_matrices2)

import numpy as np

# Function to apply inverse tau matrix transformation
def apply_inverse_tau(matrix, tau_inverse_matrix):
    transformed_matrix = np.zeros_like(matrix)
    for i in range(4):
        transformed_matrix[i] = np.dot(matrix[i], tau_inverse_matrix.T)
    return transformed_matrix

# Function to check positivity
def is_valid(matrix, rows, cols, row, col, num):
    original_value = matrix[row][col]
    matrix[row][col] = num

    # Check row condition
    if col == 3:
        if sum(matrix[row][:2]) != matrix[row][2] + num:
            matrix[row][col] = original_value
            return False

    # Check column condition
    if row == 3:
        if matrix[0][col] + matrix[1][col] != matrix[2][col] + num:
            matrix[row][col] = original_value
            return False

    # Positivity check
    if col == 3:
        matrix[row][col] = num
        if matrix[row][0] + matrix[row][2] < 0 or matrix[row][0] + matrix[row][3] < 0:
            matrix[row][col] = original_value
            return False
        if matrix[row][1] + matrix[row][2] < 0 or matrix[row][1] + matrix[row][3] < 0:
            matrix[row][col] = original_value
            return False

    matrix[row][col] = original_value
    return True

# Load extendable matrices
extendable_matrices = np.load("/content/extendableMappings.npy")

# Tau inverse matrix
tau_inverse_matrix = np.array([
    [3/4, -1/4, 3/4, -1/4],
    [-1/4, 3/4, -1/4, 3/4],
    [-1/4, 3/4, 3/4, -1/4],
    [3/4, -1/4, -1/4, 3/4]
])

# Calculate the inverse for the first matrix in the extendable list
first_matrix = extendable_matrices[0]
inverse_transformed_matrix = apply_inverse_tau(first_matrix, tau_inverse_matrix)

# Check if the inverse-transformed matrix is positive
valid = True
rows, cols = inverse_transformed_matrix.shape
for row in range(rows):
    for col in range(cols):
        if not is_valid(inverse_transformed_matrix, rows, cols, row, col, inverse_transformed_matrix[row][col]):
            valid = False
            break
    if not valid:
        break

# Output the matrix, its inverse, and the result of positivity check
first_matrix, inverse_transformed_matrix, valid

import numpy as np

# Define the validation function
def is_valid(matrix, row, col, num):
    original_value = matrix[row][col]  # Store the original value

    # Assign the current number to the matrix
    matrix[row][col] = num

    # Check row condition (sum of the first two equals the sum of the last two in each row)
    if col == 3:
        if sum(matrix[row][:2]) != matrix[row][2] + num:
            matrix[row][col] = original_value  # Revert back to the original value
            return False

    # Check column condition (sum of the 1st and 2nd rows equal the 3rd and 4th)
    if row == 3:
        if matrix[0][col] + matrix[1][col] != matrix[2][col] + num:
            matrix[row][col] = original_value  # Revert back to the original value
            return False

    # Positivity check (sum of any element added to any other element is positive)
    if col == 3:
        if matrix[row][0] + matrix[row][2] < 0 or matrix[row][0] + matrix[row][3] < 0:
            matrix[row][col] = original_value  # Revert back to the original value
            return False
        if matrix[row][1] + matrix[row][2] < 0 or matrix[row][1] + matrix[row][3] < 0:
            matrix[row][col] = original_value  # Revert back to the original value
            return False

    # Revert back to the original value after validation
    matrix[row][col] = original_value
    return True

# Load the matrices that went through inverse Tau
non_extendable_matrices_after_inverse_tau = np.load('NonExtendableMatricesAfterInverseTau.npy')

# List to store valid matrices
valid_matrices = []

# Iterate through each matrix and check validity
for matrix in non_extendable_matrices_after_inverse_tau:
    # Assume we are validating for each element in the matrix (4x4 matrix)
    valid = True
    for row in range(4):
        for col in range(4):
            num = matrix[row][col]  # Current value at this position
            if not is_valid(matrix.copy(), row, col, num):  # Copy to avoid modifying original matrix
                valid = False
                break
        if not valid:
            break

    if valid:
        valid_matrices.append(matrix)

# Convert valid matrices to numpy array
valid_matrices = np.array(valid_matrices)

# Save the valid matrices to a new file
np.save('ValidMatricesAfterInverseTau.npy', valid_matrices)

# Print the number of valid matrices found
print(f"Number of valid matrices found: {len(valid_matrices)}")
print(len(non_extendable_matrices_after_inverse_tau))

import numpy as np

# Load both sets of matrices
extendable_matrices_after_inverse_tau = np.load('ExtendableMatricesAfterInverseTau.npy')
matrices = np.load('/content/extendableMappings.npy')

# Find the unique matrices in non_extendable_matrices_after_inverse_tau that are not in nonextendable_matrices
unique_matrices = []

# Loop through each matrix in non_extendable_matrices_after_inverse_tau
for matrix in extendable_matrices_after_inverse_tau:
    # Check if this matrix exists in nonextendable_matrices
    if not any(np.array_equal(matrix, extendable_matrix) for extendable_matrix in matrices):
        # If it's not found, add to the unique_matrices list
        unique_matrices.append(matrix)

# Convert unique_matrices to a numpy array
unique_matrices = np.array(unique_matrices)

# Save the unique matrices into a new fi le
np.save('UniqueNonExtendableMatrices.npy', unique_matrices)

# Print the number of unique matrices found
print(f"Number of unique matrices found: {len(unique_matrices)}")

import numpy as np

# Load both sets of matrices
non_extendable_matrices_after_inverse_tau = np.load('NonExtendableMatricesAfterInverseTau.npy')
matrices = np.load('/content/combinedMappings.npy')

# Initialize list to store unique matrices and a counter
unique_matrices = 0

# Loop through each matrix in non_extendable_matrices_after_inverse_tau
for matrix in non_extendable_matrices_after_inverse_tau:
    # Check if this matrix exists in nonextendable_matrices
    if not any(np.array_equal(matrix, nonextendable_matrix) for nonextendable_matrix in matrices):
        # If it's not found, add to the unique_matrices list
        unique_matrices += 1


# Print the number of unique matrices found
print(f"Number of unique matrices found: {unique_matrices}")

#thsi is just a code check
import numpy as np
from scipy.optimize import linprog

# Define the inverse tau transformation matrix
tau_inverse_matrix = np.array([
    [3/4, -1/4, 3/4, -1/4],
    [-1/4, 3/4, -1/4, 3/4],
    [-1/4, 3/4, 3/4, -1/4],
    [3/4, -1/4, -1/4, 3/4]
])

# Function to apply the inverse Tau transformation to a matrix
def apply_inverse_tau(matrix):
    transformed_matrix = np.zeros_like(matrix)
    for i in range(4):
        transformed_matrix[i] = np.dot(tau_inverse_matrix, matrix[i])
    return transformed_matrix

# Function to check extendability
def is_extendable(matrix):
    mat = [1, 1, 1, 1]
    for_A_ub = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1]]
    solve_for = [[min(matrix[0][2], matrix[0][3])],
                 [min(matrix[1][2], matrix[1][3])],
                 [min(matrix[2][2], matrix[2][3])],
                 [min(matrix[3][2], matrix[3][3])],
                 [-max(-matrix[0][0], -matrix[0][1])],
                 [-max(-matrix[1][0], -matrix[1][1])],
                 [-max(-matrix[2][0], -matrix[2][1])],
                 [-max(-matrix[3][0], -matrix[3][1])]]
    for_A_eq = [[1, 1, -1, -1]]
    for_b_eq = [0]

    result = linprog(c=mat, A_ub=for_A_ub, b_ub=solve_for, A_eq=for_A_eq, b_eq=for_b_eq, bounds=None)

    return result.success

# Example matrix to test
matrix = np.array([
    [1/2, 13/2, 1, 6],
    [9/2, 9/2, 9, 0],
    [0, 7, 5/2, 9/2],
    [5, 4, 15/2, 3/2]
])

'''
matrix = np.array([
    [-2, 9, 4, 3],
    [9, 0, 9, 0],
    [-1, 8, 6, 1],
    [8, 1, 7, 2]
])
'''

# Apply the inverse Tau transformation to the matrix
inverse_transformed_matrix = apply_inverse_tau(matrix)

# Print the transformed matrix
print("Transformed matrix after inverse Tau:")
print(inverse_transformed_matrix)

# Check if the transformed matrix is extendable
if is_extendable(inverse_transformed_matrix):
    print("The matrix is extendable after inverse Tau.")
else:
    print("The matrix is not extendable after inverse Tau.")

import numpy as np

# Define the inverse tau transformation matrix
tau_inverse_matrix = np.array([
    [1/2, -1/2, 1],
    [-1/2, 3/2, -1],
    [-1/2, 1/2, 1],
    [3/2, -1/2, -1]
])

# Function to apply the inverse Tau transformation to a matrix (ignoring the 4th column)
def apply_inverse_tau(matrix):
    transformed_matrix = np.zeros((4, 4))  # We will get back a 4x4 matrix for [a,b,c,d]
    for i in range(4):
        # Multiply the inverse tau matrix by the first 3 elements [x, y, z]
        transformed_matrix[i] = np.dot(tau_inverse_matrix, matrix[i, :3])
    return transformed_matrix

# Process the newly extendable matrices with the inverse Tau transformation
inverse_transformed_matrices = []

for matrix in extendable_matrices:
    inverse_transformed_matrix = apply_inverse_tau(matrix)
    inverse_transformed_matrices.append(inverse_transformed_matrix)

# Save the inverse transformed matrices
np.save('InverseTransformedMatrices.npy', inverse_transformed_matrices)

# Print the number of matrices processed
print(f"Number of newly extendable matrices processed with inverse Tau: {len(inverse_transformed_matrices)}")

# just checking if the t actually checks everything off of the original code


import numpy as np
from scipy.optimize import linprog

# Load the provided .npy files
matrices = np.load("/content/combinedMappings.npy")

# Function to check extendability
def is_extendable(matrix):
    mat = [1, 1, 1, 1]
    for_A_ub = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ]
    solve_for = [
        [min(matrix[0][2], matrix[0][3])],
        [min(matrix[1][2], matrix[1][3])],
        [min(matrix[2][2], matrix[2][3])],
        [min(matrix[3][2], matrix[3][3])],
        [-max(-matrix[0][0], -matrix[0][1])],
        [-max(-matrix[1][0], -matrix[1][1])],
        [-max(-matrix[2][0], -matrix[2][1])],
        [-max(-matrix[3][0], -matrix[3][1])]
    ]
    for_A_eq = [[1, 1, -1, -1]]
    for_b_eq = [0]

    result = linprog(c=mat, A_ub=for_A_ub, b_ub=solve_for, A_eq=for_A_eq, b_eq=for_b_eq, bounds=None)

    return result.success

# Separate matrices into extendable and non-extendable
extendable_matrices = []
non_extendable_matrices = []

for matrix in matrices:
    if is_extendable(matrix):
        extendable_matrices.append(matrix)
    else:
        non_extendable_matrices.append(matrix)

# Function to run the specific test on non-extendable matrices
def specific_test(matrix):
    all_rows_satisfy_condition = True

    for row in matrix:
        a, b, c, d = row[0], row[1], row[2], row[3]

        l = np.max([-a, -b])
        r = np.min([c, d])

        # Check if there exists a value t in the range [l, r] in the current row
        if not any(l <= value <= r for value in row):
            all_rows_satisfy_condition = False
            break  # No need to check further rows if this one doesn't satisfy

        # Check if -min(a,b) <= t <= max(c,d) holds for all t in the matrix
        if not (l <= min(c, d)):
            all_rows_satisfy_condition = False
            break

        # Check if -a, -b <= t <= c, d
        for t in row:
            if not (-a <= t <= c and -b <= t <= d):
                all_rows_satisfy_condition = False
                break

    return all_rows_satisfy_condition

# Run the specific test on non-extendable matrices and count how many pass
Have_A_t = 0

for matrix in non_extendable_matrices:
    if specific_test(matrix):
        Have_A_t += 1

# Print the results
print(f"Number of extendable matrices: {len(extendable_matrices)}")
print(f"Number of non-extendable matrices: {len(non_extendable_matrices)}")
print(f"Number of non-extendable matrices passing the specific test: {Have_A_t}")

import numpy as np
from scipy.optimize import linprog

'''
tau = np.array([[1, 0, 0.5, 0.5],
                [0, 1, 0.5, 0.5],
                [0.5, 0.5, 1, 0],
                [0.5, 0.5, 0, 1]])
 '''

tau = np.array([[0.5, 0.5, 1, 0],
                [1, 0, 0.5, 0.5],
                [0, 1, 0.5, 0.5],
                [0.5, 0.5, 0, 1]])



#psi = np.array()
matrices = np.load("/content/combinedMappings.npy")

# Perform matrix multiplication
multiplied_matrices = [np.dot(tau, matrix) for matrix in matrices]
multiplied_matrices = np.array(multiplied_matrices)

newlyExtendable = 0
stillnonExtendable = 0
failedTAU2 = []
new_extendable_matrices = []
new_non_extendable_matrices = []

for matrix in multiplied_matrices:
  mat = [1, 1, 1, 1]
  for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],]

  solve_for = [[min(matrix[0][2], matrix[0][3])],
             [min(matrix[1][2], matrix[1][3])],
             [min(matrix[2][2], matrix[2][3])],
             [min(matrix[3][2], matrix[3][3])],
             [-max(-matrix[0][0], -matrix[0][1])],
             [-max(-matrix[1][0], -matrix[1][1])],
             [-max(-matrix[2][0], -matrix[2][1])],
             [-max(-matrix[3][0], -matrix[3][1])]]
  for_A_eq = [[1,1,-1,-1]]
  for_b_eq = [0]

  result = linprog(c=mat, A_ub = for_A_ub, b_ub = solve_for, A_eq = for_A_eq, b_eq = for_b_eq, bounds = None)

  if (result.success):
    newlyExtendable += 1
    new_extendable_matrices.append(matrix)
  else:
    stillnonExtendable += 1
    failedTAU2.append(matrix)

np.save('ExtendableMatricesTauDotProduct.npy', new_extendable_matrices)
np.save('NonExtendableMatricesTauDotProduct.npy', failedTAU2)

print("newly extendable", newlyExtendable)
print("still not extendable", stillnonExtendable)
failedTAU2 = np.array(failedTAU2)

import numpy as np
from scipy.optimize import linprog

tau = np.array([[1, 0, 0.5, 0.5],
                [0, 1, 0.5, 0.5],
                [0.5, 0.5, 1, 0],
                [0.5, 0.5, 0, 1]])


#psi = np.array()
matrices = np.load("/content/combinedMappings.npy")

# Perform matrix multiplication
multiplied_matrices = [np.dot(tau, matrix) for matrix in matrices]
multiplied_matrices = np.array(multiplied_matrices)

newlyExtendable = 0
stillnonExtendable = 0
failedTAU2 = []

for matrix in multiplied_matrices:
  mat = [1, 1, 1, 1]
  for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],]

  solve_for = [[min(matrix[0][2], matrix[0][3])],
             [min(matrix[1][2], matrix[1][3])],
             [min(matrix[2][2], matrix[2][3])],
             [min(matrix[3][2], matrix[3][3])],
             [-max(-matrix[0][0], -matrix[0][1])],
             [-max(-matrix[1][0], -matrix[1][1])],
             [-max(-matrix[2][0], -matrix[2][1])],
             [-max(-matrix[3][0], -matrix[3][1])]]
  for_A_eq = [[1,1,-1,-1]]
  for_b_eq = [0]

  result = linprog(c=mat, A_ub = for_A_ub, b_ub = solve_for, A_eq = for_A_eq, b_eq = for_b_eq, bounds = None)

  if (result.success):
    newlyExtendable += 1
  else:
    stillnonExtendable += 1
    failedTAU2.append(matrix)

print("newly extendable", newlyExtendable)
print("still not extendable", stillnonExtendable)
failedTAU2 = np.array(failedTAU2)