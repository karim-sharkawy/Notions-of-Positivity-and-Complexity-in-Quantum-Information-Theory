import random # Generates pseudo-random numbers
import numpy as np
from scipy import linalg

# create the matrix with random numbers
n = 2 # figure out how to make this any number
A = np.random.randint(10, size = (n,n)) #this is a square matrix that chooses random numbers between 0-9 (inclusive)

def positive_req(): #checks for positivity (semidefinite)
    la, v = linalg.eig(A)
    print(la)
    for i in la:
        if i < 0:
            print("Oh no! This is not semidefinite")
            break
        else:
            print("This is semidefinite :)")
            # look into flags!!
positive_req()

def symmetric_req(): # makes sure the matrix is symmetric
    if np.array_equal(A.T,A): #returns boolean
        print("Matrix is Symmetric!")
    else:
        print("Matrix is not symmetric")
symmetric_req()



