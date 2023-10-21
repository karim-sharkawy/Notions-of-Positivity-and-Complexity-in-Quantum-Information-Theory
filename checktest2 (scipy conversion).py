#This code was worked on and edited by Luke Luschwitz and Karim El-Sharkawy

from scipy.optimize import linprog

# The points in P(2,2) that were mapped from E(2,2).
# Points are mapped by f: E(2,2) -> P(2,2)
x = [0,0,0,0] # x is a mapped point in P(2,2)
y = [0,0,0,0] # y is a mapped point in P(2,2)
z = [0,0,0,0] # z is a mapped point in P(2,2)
w = [0,0,0,0] # w is a mapped point in P(2,2)

# These are what x, y, z, and w represent
# x = [a1,b1,c1,d1]
# y = [a2,b2,c2,d2]
# z = [a3,b3,c3,d3]
# w = [a4,b4,c4,d4]

# The matrix that we are trying to map to R4
matrix = [x,
          y,
          z,
          w]

mat = [1, 1, 1, 1]
a = mat[0]
b = mat[1]
c = mat[2]
d = mat[3]

for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],]

# A_eq = ((a+b),(c+d)) #original ineq13

solve_for = [[min(matrix[0][2],matrix[0][3])],
             [min(matrix[1][2], matrix[1][3])],
             [min(matrix[2][2], matrix[2][3])],
             [min(matrix[3][2], matrix[3][3])],
             [-max(-matrix[0][0], -matrix[0][1])],
             [-max(-matrix[1][0], -matrix[1][1])],
             [-max(-matrix[2][0], -matrix[2][1])],
             [-max(-matrix[3][0], -matrix[3][1])]]


# Solve for inequalities with scipy.optimize.linprog
result = linprog(c=mat, A_ub = for_A_ub, b_ub = solve_for, bounds = None)

# Print whether the inequality solving was successful (output of the linprog function)
print(result)
