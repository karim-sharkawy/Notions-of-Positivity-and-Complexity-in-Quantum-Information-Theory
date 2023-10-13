import random # Generates pseudo-random numbers
import numpy as np
from scipy import linalg

# create the matrix with random numbers
n = random.randint(2,10) # there's a diff between random and np.random
A = np.random.randint(10, size = (n,n)) #this is a square matrix that chooses random numbers between 0-9 (inclusive)
pos_matrices: []
neg_matrices: []
print(A)

def positive_req(): #checks for positivity (semidefinite)
    la, v = linalg.eig(A)
    print(la)
    is_semidefinite = True #this is a flag
    for i in la:
        if i < 0: 
            is_semidefinite = False
            break
    if is_semidefinite:
        print("This is semidefinite")
        pos_matrices.append(A)
    else:
        print("Oh no, this is not semidefinite")
        neg_matrices.append(A)
        
positive_req() #make sure to add an .append for positive and negative so we can see difference between them

def symmetric_req(): # makes sure the matrix is symmetric
    if np.array_equal(A.T,A): #returns boolean
        print("Matrix is Symmetric!")
    else:
        print("Matrix is not symmetric")
symmetric_req()



