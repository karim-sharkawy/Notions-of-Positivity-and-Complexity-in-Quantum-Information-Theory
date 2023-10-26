from scipy.optimize import linprog

# Author: Luke Luschwitz and Karim El-Sharkawy
# Last edit: 10/26/23 6:00pm

# link to scipy.optimize.linprog: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

# Reference images from Professor Sinclair
# https://cdn.discordapp.com/attachments/993798931777585153/1165299134165696522/image0.jpg?ex=65465863&is=6533e363&hm=6a59d0a154f6bfa745b5194ccad740a781b196437bcee525abc6a2cf0768779d&
# https://cdn.discordapp.com/attachments/993798931777585153/1165291718418112603/IMG_7540.jpg?ex=6546517b&is=6533dc7b&hm=01582d955a387eac71189654067dff37c5b3988983e7b475c129e27bc427c6ca&


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

# In the scipy.optimize.linprog documentation, mat is c: "The coefficients of the linear objective function to be minimized".
# On Prof. SinClair's blackboard, mat is t: the vector of variables t1,t2,t3,t4 that need to be minimized/maximized.
# QUESTION: is this the parameter c in the scipy.optimize.linprog documentation?
# QUESTION: should mat be all 1's or two 1's and two -1's?
mat = [1, 1, -1, -1]
a = mat[0]
b = mat[1]
c = mat[2]
d = mat[3]

# for_A_ub represents the system of inequalities
for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],]

# In the scipy.optimize.linprog documentation, solve_for is b_ub: "The inequality constraint vector. Each element represents an upper bound on the corresponding value of A_ub @ x.".
# On Prof. SinClair's blackboard, this is r: [r1,r2,r3,r4,-l1,-l2,-l3,-l4].
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
# for how this function works, it has it such that A_ub <= b_ub at whatever c is equal to

# Print whether the inequality solving was successful (output of the linprog function)
print(result)
