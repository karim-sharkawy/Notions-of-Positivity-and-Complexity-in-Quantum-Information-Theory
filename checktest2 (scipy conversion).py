from scipy.optimize import linprog

# Author: Luke Luschwitz and Karim El-Sharkawy
# Last edit: 10/26/23 7:35pm

# link to scipy.optimize.linprog: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

# Reference images from Professor Sinclair
# https://cdn.discordapp.com/attachments/993798931777585153/1165299134165696522/image0.jpg?ex=65465863&is=6533e363&hm=6a59d0a154f6bfa745b5194ccad740a781b196437bcee525abc6a2cf0768779d&
# https://cdn.discordapp.com/attachments/993798931777585153/1165291718418112603/IMG_7540.jpg?ex=6546517b&is=6533dc7b&hm=01582d955a387eac71189654067dff37c5b3988983e7b475c129e27bc427c6ca&


# The points in P(2,2) that were mapped from E(2,2).
# Points are mapped by f: E(2,2) -> P(2,2)

# This matrix we know is extendible
# x = [-1,3,1,1] # x is a mapped point in P(2,2)
# y = [0,1,1,0] # y is a mapped point in P(2,2)
# z = [0,2,1,1] # z is a mapped point in P(2,2)
# w = [1,0,1,0] # w is a mapped point in P(2,2)

# This matrix we know is not extendible
# x = [-1,3,1,1] # x is a mapped point in P(2,2)
# y = [0,1,0,1] # y is a mapped point in P(2,2)
# z = [0,2,2,0] # z is a mapped point in P(2,2)
# w = [1,0,1,0] # w is a mapped point in P(2,2)

# This matrix we know is extendible
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
# On Prof. SinClair's blackboard, mat is t: the vector of variables t1,t2,t3,t4 that need to be minimized/maximized. In our case, the objective function does not matter, we only care whether the solution exists.
mat = [1, 1, 1, 1]

# for_A_ub represents the system of inequalities
for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],]

# In the scipy.optimize.linprog documentation, solve_for is b_ub: "The inequality constraint vector. Each element represents an upper bound on the corresponding value of A_ub @ x."
# On Prof. SinClair's blackboard, this is r: [r1,r2,r3,r4,-l1,-l2,-l3,-l4].
solve_for = [[min(matrix[0][2],matrix[0][3])],
             [min(matrix[1][2], matrix[1][3])],
             [min(matrix[2][2], matrix[2][3])],
             [min(matrix[3][2], matrix[3][3])],
             [-max(-matrix[0][0], -matrix[0][1])],
             [-max(-matrix[1][0], -matrix[1][1])],
             [-max(-matrix[2][0], -matrix[2][1])],
             [-max(-matrix[3][0], -matrix[3][1])]]

# In the scipy.optimize.linprog documentation, for_A_eq is A_eq: "The equality constraint matrix. Each row of A_eq specifies the coefficients of a linear equality constraint on x."
# On Prof. SinClair's blackboard, this is t1-t2-t3+t4=0.
for_A_eq = [[1,-1,-1,1]]

# In the scipy.optimize.linprog documentation, for_b_eq is b_eq: "The equality constraint vector. Each element of A_eq @ x must equal the corresponding element of b_eq."
# On Prof. SinClair's blackboard, this is t1-t2-t3+t4=0.
for_b_eq = [0]


# Solve for inequalities with scipy.optimize.linprog
result = linprog(c=mat, A_ub = for_A_ub, b_ub = solve_for, A_eq = for_A_eq, b_eq = for_b_eq, bounds = None)
# for how this function works, it has it such that A_ub <= b_ub at whatever c is equal to

# Print whether the inequality solving was successful (output of the linprog function)
for row in matrix:
    print(row)
print(result)
