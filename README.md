# Yes-math
for math project :))
# This code was made by Raja Darshini of Purdue University, NOT me. I'm just practicing GitHub right now

import random # Generates pseudo-random numbers
from sympy import symbols, solve, And, Eq
from sympy.solvers.inequalities import reduce_rational_inequalities
import sympy as sym
from sympy import symbols
from sympy import sympify as S, Symbol
from sympy.solvers.inequalities import reduce_inequalities

def get_random_number():
    return random.randrange(9)

def is_valid(matrix, rows, cols, row, col, num):
    # Check row condition
    if col == 3:
        if sum(matrix[row][:2]) != num + matrix[row][2]:
            return False

    # Check column condition
    if row == 3:
        if sum([matrix[i][col] for i in range(2)]) != num + matrix[2][col]:
            return False

    # Check diagonal condition
    if row == 3 and col == 3:
        if matrix[0][3] + matrix[3][0] != matrix[1][2] + matrix[2][1]:
            return False

    # Check for row uniqueness
    if col == 3:
        matrix[row][col] = num
        if tuple(matrix[row]) in set(map(tuple, matrix[:row])):
            matrix[row][col] = None
            return False
        matrix[row][col] = None

    # Check that only one digit can be 0 in a row
    if col == 3:
        zero_count = sum([1 for x in matrix[row] if x == 0])
        if zero_count > 1:
            return False

    if all(x is not None for row in matrix for x in row):
        for i in range(cols):
            if matrix[1][i] + matrix[2][i] != matrix[0][i] + matrix[3][i]:
                return False

    return True


def create_matrix(rows, cols):
    matrix = [[None for _ in range(cols)] for _ in range(rows)]

    def backtrack(row, col):
        if row == rows:
            return True

        nums = list(range(10))
        random.shuffle(nums)

        for num in nums:
            if is_valid(matrix, rows, cols, row, col, num):
                matrix[row][col] = num

                next_row = row
                next_col = col + 1
                if next_col == cols:
                    next_row += 1
                    next_col = 0

                if backtrack(next_row, next_col):
                    return True

        matrix[row][col] = None
        return False

    backtrack(0, 0)

    return matrix

rows = 4
cols = 4
matrix = create_matrix(rows, cols)

matrix[1], matrix[3] = matrix[3], matrix[1]

for row in matrix:
    print(row)

x = [0, 0, 0, 0]
y = [0, 0, 0, 0]
z = [0, 0, 0, 0]
w = [0, 0, 0, 0]

a = symbols("a")
b = symbols("b")
c = symbols("c")
d = symbols("d")

mat = [a, b, c, d]

matrix = []
matrix.append(mat)
matrix.append([f'{x[i]}-{y[i]}+{mat[i]}' for i in range(len(x))])
matrix.append([f'{y[i]}-{mat[i]}' for i in range(len(y))])
matrix.append([f'{w[i]}-{mat[i]}' for i in range(len(w))])

for row in matrix:
    print(row)

print("\n")
new_matrix = []
new_matrix.append(mat)
new_matrix.append([x[i] - y[i] + mat[i] for i in range(len(x))])  # Removed f-string
new_matrix.append([y[i] - mat[i] for i in range(len(y))])        # Removed f-string
new_matrix.append([w[i] - mat[i] for i in range(len(w))])        # Removed f-string


for row in new_matrix:
    print(row)
print("\n")

ineq1 = new_matrix[1][0] + new_matrix[1][2] >= 0
ineq2 = new_matrix[2][0] + new_matrix[2][2] >= 0
ineq3 = new_matrix[3][0] + new_matrix[3][2] >= 0
ineq4 = new_matrix[1][0] + new_matrix[1][3] >= 0
ineq5 = new_matrix[2][0] + new_matrix[2][3] >= 0
ineq6 = new_matrix[3][0] + new_matrix[3][3] >= 0
ineq7 = new_matrix[1][1] + new_matrix[1][2] >= 0
ineq8 = new_matrix[2][1] + new_matrix[2][2] >= 0
ineq9 = new_matrix[3][1] + new_matrix[3][2] >= 0
ineq10 = new_matrix[1][1] + new_matrix[1][3] >= 0
ineq11 = new_matrix[2][1] + new_matrix[2][3] >= 0
ineq12 = new_matrix[3][1] + new_matrix[3][3] >= 0
ineq13 = Eq(a + b, c + d)

# Now use reduce_inequalities with SymPy expressions
result1 = reduce_inequalities(ineq1, [a])
result2 = reduce_inequalities(ineq2, [a])
result3 = reduce_inequalities(ineq3, [a])
result4 = reduce_inequalities(ineq4, [a])
result5 = reduce_inequalities(ineq5, [a])
result6 = reduce_inequalities(ineq6, [a])
result7 = reduce_inequalities(ineq7, [b])
result8 = reduce_inequalities(ineq8, [b])
result9 = reduce_inequalities(ineq9, [b])
result10 = reduce_inequalities(ineq10, [b])
result11 = reduce_inequalities(ineq11, [b])
result12 = reduce_inequalities(ineq12, [b])

print("Result 1:", result1)
print("Result 2:", result2)
print("Result 3:", result3)
print("Result 4:", result4)
print("Result 5:", result5)
print("Result 6:", result6)
print("Result 7:", result7)
print("Result 8:", result8)
print("Result 9:", result9)
print("Result 10:", result10)
print("Result 11:", result11)
print("Result 12:", result12)
ineq_combined = And(ineq1, ineq2, ineq3, ineq4, ineq5, ineq6, ineq7, ineq8, ineq9, ineq10, ineq11, ineq12, ineq13)

# Now use reduce_inequalities with the combined inequality
result = reduce_inequalities(ineq_combined, [a, c , b , d])

print("Result:", result)
