from sympy import symbols, solve, And, Eq
from sympy.solvers.inequalities import reduce_rational_inequalities
import sympy as sym
from sympy import symbols
from sympy import sympify as S, Symbol
from sympy.solvers.inequalities import reduce_inequalities
from scipy.optimize import linprog
from scipy.optimize import minimize

x = [0, 0, 0, 0] # is it seeing them as vectors?
y = [0, 0, 0, 0]
z = [0, 0, 0, 0]
w = [0, 0, 0, 0]

mat = [0, 0, 0, 0]
a = mat[0]
b = mat[1]
c = mat[2]
d = mat[3]

def objective(mat):
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

con1 = {'type':'ineq', 'fun': ineq1}
con2 = {'type':'ineq', 'fun': ineq2}
con3 = {'type':'ineq', 'fun': ineq3}
con4 = {'type':'ineq', 'fun': ineq4}
con5 = {'type':'ineq', 'fun': ineq5}
con6 = {'type':'ineq', 'fun': ineq6}
con7 = {'type':'ineq', 'fun': ineq7}
con8 = {'type':'ineq', 'fun': ineq8}
con9 = {'type':'ineq', 'fun': ineq9}
con10 = {'type':'ineq', 'fun': ineq10}
con11 = {'type':'ineq', 'fun': ineq11}
con12 = {'type':'ineq', 'fun': ineq12}
con13 = {'type':'ineq', 'fun': ineq13}


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
ineq_combined = (ineq1, ineq2, ineq3, ineq4, ineq5, ineq6, ineq7, ineq8, ineq9, ineq10, ineq11, ineq12) # what is And? ineq13 removed
# b_ub = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
bu = [0, 0]
bnds = [bu, bu, bu, bu]
cons = [con1, con2, con3, con4, con5, con6, con7, con8, con9, con10, con11, con12, con13]

# Now use reduce_inequalities with the combined inequality
#result = linprog(c=mat, A_ub = ineq_combined, b_ub = b_ub, bounds = bnds) #A_ub can only accepts dimension of 2
result = minimize(objective, mat, method='SLSQP', \
                  bounds=bnds, constraints=cons)

print("Result:", result)
