from sympy import symbols, solve, And, Eq
from sympy.solvers.inequalities import reduce_rational_inequalities
import sympy as sym
from sympy import symbols
from sympy import sympify as S, Symbol
from sympy.solvers.inequalities import reduce_inequalities
from scipy.optimize import linprog

x = [0, 0, 0, 0] # is it seeing them as vectors?
y = [0, 0, 0, 0]
z = [0, 0, 0, 0]
w = [0, 0, 0, 0]

matrix = [x,y,z,w]

mat = [1, 1, 1, 1]
a = mat[0]
b = mat[1]
c = mat[2]
d = mat[3]

print("\n")
new_matrix = []
new_matrix.append(mat)
new_matrix.append([x[i] - y[i] + mat[i] for i in range(len(x))])  # Removed f-string
new_matrix.append([y[i] - mat[i] for i in range(len(y))])        # Removed f-string
new_matrix.append([w[i] - mat[i] for i in range(len(w))])        # Removed f-string


for row in new_matrix:
    print(row)
print("\n")

for_A_ub = [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            [-1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,-1],]

A_eq = ((a+b),(c+d)) #original ineq13

solve_for = [[min(matrix[0][2],matrix[0][3])],
             [min(matrix[1][2], matrix[1][3])],
             [min(matrix[2][2], matrix[2][3])],
             [min(matrix[3][2], matrix[3][3])],
             [-max(-matrix[0][0], -matrix[0][1])],
             [-max(-matrix[1][0], -matrix[1][1])],
             [-max(-matrix[2][0], -matrix[2][1])],
             [-max(-matrix[3][0], -matrix[3][1])]]


# Now use reduce_inequalities with SymPy expressions
#result1 = reduce_inequalities(ineq1, [a])
#result2 = reduce_inequalities(ineq2, [a])
#result3 = reduce_inequalities(ineq3, [a])
#result4 = reduce_inequalities(ineq4, [a])
#result5 = reduce_inequalities(ineq5, [a])
#result6 = reduce_inequalities(ineq6, [a])
#result7 = reduce_inequalities(ineq7, [b])
#result8 = reduce_inequalities(ineq8, [b])
#result9 = reduce_inequalities(ineq9, [b])
#result10 = reduce_inequalities(ineq10, [b])
#result11 = reduce_inequalities(ineq11, [b])
#result12 = reduce_inequalities(ineq12, [b])

#print("Result 1:", result1)
#print("Result 2:", result2)
#print("Result 3:", result3)
#print("Result 4:", result4)
#print("Result 5:", result5)
#print("Result 6:", result6)
#print("Result 7:", result7)
#print("Result 8:", result8)
#print("Result 9:", result9)
#print("Result 10:", result10)
#print("Result 11:", result11)
#print("Result 12:", result12)
#ineq_combined = And(ineq1, ineq2, ineq3, ineq4, ineq5, ineq6, ineq7, ineq8, ineq9, ineq10, ineq11, ineq12, ineq13) # what is And?
#print(type(ineq_combined))
bnds = [0, 0, 0, 0]

# Now use reduce_inequalities with the combined inequality
result = linprog(c=mat, A_ub = for_A_ub, b_ub = solve_for, bounds = None)

print("Result:", result)

