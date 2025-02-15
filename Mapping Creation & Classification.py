### 1) Creating positive mappings and classifying extensions

# Creating 4x4 positive mappings according to the properties of $E_{2,2}$
# The following block was created by Darshini Rajamani with editing from Luke
import random;
import numpy as np
from scipy.optimize import linprog

def get_random_number():
    return random.randint(-9, 9)

# Setting the requirements for Positive matrices
def is_valid(matrix, rows, cols, row, col, num):
    # initial assigment to check validation
    matrix[row][col] = num

    # Check row condition (check in E(2,2)): sum of the first two equals the sum of the last two in each row
    if col == 3:
        if sum(matrix[row][:2]) != matrix[row][2] + num:
            matrix[row][col] = None
            return False

    # Check column condition: sum of the 1st and 4th rows equals the sum of the 2nd and 3rd
    if row == 3:
        if matrix[0][col] + matrix[1][col] != matrix[2][col] + num:
            matrix[row][col] = None
            return False

    # Positivity check: Sum of any element added to any other element is positive
    if col == 3:
        matrix[row][col] = num
        if matrix[row][0] + matrix[row][2] < 0:
            matrix[row][col] = None
            return False
        if matrix[row][0] + matrix[row][3] < 0:
            matrix[row][col] = None
            return False
        if matrix[row][1] + matrix[row][2] < 0:
            matrix[row][col] = None
            return False
        if matrix[row][1] + matrix[row][3] < 0:
            matrix[row][col] = None
            return False

    matrix[row][col] = None
    return True

# Creating the mappings (matrices)
def create_matrix(rows, cols):
    matrix = [[None for _ in range(cols)] for _ in range(rows)] # initialize every element as 'none'

    backtrackFailedAttempts = 0
    def backtrack(row, col):  # to check for complete and valid solution
        nonlocal backtrackFailedAttempts
        if backtrackFailedAttempts > 1000:
            return False

        if row == rows:
            return True

        nums = [i for i in range(-9, 10)]  # range [-9, 10)
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
        backtrackFailedAttempts = backtrackFailedAttempts + 1
        return False

    backtrack(0, 0)  # recursion

    return matrix

# code below was created by Luke Luschwitz and Karim with influence from Abbas and Darshini

numberOfMappingsToCreate = 100000

listOfMappings = []
for i in range(numberOfMappingsToCreate):
  rows = 4
  cols = 4

  isAValidMatrix = False

  while not isAValidMatrix:
    isAValidMatrix = True
    matrix = create_matrix(rows, cols) # creating mappings
    for row in matrix:
      for element in row:
        if element is None:
          isAValidMatrix = False

  matrix[1], matrix[3] = matrix[3], matrix[1]

  listOfMappings.append(matrix)

# saving the mappings to .npy file
listOfMappings_array = np.array(listOfMappings)
np.save('/content/listOfMappings.npy', listOfMappings_array)

# Checking for Extendability using Linear Programming
extendableMappings = []
nonExtendableMappings = []

# Linear programming to determine extendability
for matrix in listOfMappings:
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

  # uncomment the block below if you want to print every matrix and its extendability
  '''
  for row in matrix:
    print(row)
  if (result.success):
    extendableMappings.append(matrix)
    print("Extendable")
  else:
    nonExtendableMappings.append(matrix)
    print("Not extendable")
  print()
  '''

# Save the sample data as files
np.save('/content/extendableMappings.npy', extendableMappings)
np.save('/content/nonExtendableMappings.npy', nonExtendableMappings)

print(f"{len(extendableMappings)} Extendable Maps")
print(f"{len(nonExtendableMappings)} Nonextendable Maps")

### 2) Farthest Bs and Making Classifiers using SVMs
# All the code below was made by Luke with slight influence from Abbas

# This function returns the matrix B that is farthest away from the matrices in setA
def farthestBFromA(setB, setA):
  totalDistancesFromA = []
  # iterate through each B matrix
  for B in setB:
    currentBTotalDistance = 0
    # accumulate sum of distances from A to B (distance squared)
    for A in setA:
      currentBTotalDistance = currentBTotalDistance + (np.linalg.norm(np.subtract(B, A)) ** 2)
    totalDistancesFromA.append(currentBTotalDistance)

  orderedBsByDistance = []
  for i in range(len(totalDistancesFromA)):
    index = np.argmax(totalDistancesFromA)
    orderedBsByDistance.append(setB[index])
    totalDistancesFromA[index] = -np.inf
  return orderedBsByDistance

farthestBs = farthestBFromA(nonExtendableMappings, extendableMappings)

for i in range(len(farthestBs[0:3])):
  for row in farthestBs[i]:
      print(row)
  print(f"{i}th farthest from the extendables")
  print()

# Save the farthestBs to a file
np.save('/content/farthestBs.npy', farthestBs)
print(f"len(farthestBs): {len(farthestBs)}")

farthestBs = np.load('/content/farthestBsMORE.npy')

# Using SVMs to find classifiers than linearly seperate extendable from nonextendable mappings.
from sklearn import svm
from sklearn.metrics import accuracy_score

def CreateClassifier(farthestBIndex):
  # Declare global variables
  global extendableMappings
  global nonExtendableMappings
  global farthestBs

  # Ensure everything is a numpy array
  extendableMappings = np.array(extendableMappings)
  nonExtendableMappings = np.array(nonExtendableMappings)
  farthestBs = np.array(farthestBs)

  # Generate input by adding the farthestB matrix to the list of extendableMappings
  features = np.concatenate((
      extendableMappings,
      [farthestBs[farthestBIndex]],
      [extendableMappings[0]/1000],
      [farthestBs[farthestBIndex]/1000],
      [np.zeros((4, 4))]
      ))
  # print(f"len(features): {len(features)}")

  # Flatten the matrices into 1D arrays
  features = features.reshape(len(features), -1)

  # Generate labels (0 or 1)
  labels = [0]*len(extendableMappings) + [1,0,1,1]
  # print(f"len(labels): {len(labels)}")
  # print(f"labels: {labels}")

  # Create and train a support vector classifier
  model = svm.SVC(kernel='linear', C=1e10, coef0=0.0, tol=1e-5)
  model.fit(features, labels)

  # Set the bias (intercept) to be zero
  model.intercept_ = [0.0]

  # Make predictions on the test set
  predictions = model.predict(features)

  # Evaluate the accuracy of the model
  accuracy = accuracy_score(labels, predictions)
  # print(f"Accuracy: {accuracy}")

  # Get the coefficients (weights) of the hyperplane
  # coefficients = model.coef_.reshape(4,4)
  coefficients = model.coef_

  # Intercept of the hyperplane
  intercept = model.intercept_

  # print("Coefficients:\n", coefficients) #the matrix that we get from the ML program
  # print("Intercept:", intercept)

  return coefficients

# Tweaks the classifier slightly to find a 'nicer' classifier with whole numbers and low significant digits

def RoundClassifier(classifier):
  # Make sure classifier is a numpy array
  classifier = np.array(classifier)

  # Make a copy of the classifier
  c = classifier.copy()

  return np.round(c/10) # this is same as coefficients, but times 10 and rounded to integers

# Test if the input classifier correctly classifies all the generated sample data
def CheckClassifierAgainstSamples(classifier):
  # Combine all the samples into one array
  allMappings = np.concatenate((extendableMappings, nonExtendableMappings), axis=0)
  allMappings = allMappings.reshape(len(allMappings), -1)

  # Initialize the total counting variables
  totalMappingsInClass1 = 0
  totalMappingsInClass2 = 0
  classifiedMappingResults = []

  # Classify each sample by calculating the dot product of the classifier and each sample
  for i in range(len(allMappings)):
    dotProduct = np.dot(classifier, allMappings[i])
    dotProduct = np.sum(dotProduct)
    dotProductNormalized = dotProduct/(np.linalg.norm(classifier) * np.linalg.norm(allMappings[i]))
    extendableOrNot = "E" if i<len(extendableMappings) else "N"
    if dotProduct <= 0:
      # Supposed to be Extendable
      # print(f"{extendableOrNot} {dotProduct} (class 1)")
      classifiedMappingResults.append((allMappings[i], extendableOrNot, 1, i, dotProduct, dotProductNormalized))
      totalMappingsInClass1 += 1
    else:
      # Supposed to be nonExtendable
      # print(f"{extendableOrNot} {dotProduct} (class 2)")
      classifiedMappingResults.append((allMappings[i], extendableOrNot, 2, i, dotProduct, dotProductNormalized))
      totalMappingsInClass2 += 1
      print(allMappings[i].reshape((4,4)))
      print(extendableOrNot)
      print(dotProduct)
      print(dotProductNormalized)
      print()

  # Check if samples were misclassified
  largestMisclassification = 0 # the largest dot product of a misclassified sample
  isGoodClassifier = True
  for mapping, extendability, _class, mapID, dotProduct, dotProductNormalized in classifiedMappingResults:
    if (extendability == "E" and _class == 2):
      # problemMappings = np.concatenate((problemMappings, [mapping]), axis=0)
      # print("Bad classifier, extendableMapping in class 2")
      # print(mapping.reshape(4,-1), extendability, _class, mapID, dotProduct, dotProductNormalized)
      isGoodClassifier = False
      if dotProductNormalized > largestMisclassification:
        largestMisclassification = dotProductNormalized
      # break
  if totalMappingsInClass1 < 1:
    # print("Bad classifier, no mappings in class 1")
    isGoodClassifier = False
  if totalMappingsInClass2 < 1:
    # print("Bad classifier, no mappings in class 2")
    isGoodClassifier = False

  # print(f"totalMappingsInClass1: {totalMappingsInClass1}")
  # print(f"totalMappingsInClass2: {totalMappingsInClass2}")
  # print(f"Largest misclassification: {largestMisclassification}")

  return isGoodClassifier

# Check that the found classifier is a true classifier using linear programming
from scipy.optimize import linprog

def CheckIfTrueClassifier(classifier):
  classifier = np.array(classifier)

  c = -classifier.reshape(1, 16)

  c = np.concatenate((c, np.zeros((1, 4))), axis=1) # Add 4 dummy variables to check for extendability

  for_A_ub = [[-1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [-1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0], # positivity check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0], # positivity check

              [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0], # extendability check
              [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0], # extendability check
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # extendability check
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # extendability check
              [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], # extendability check
              [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], # extendability check
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # extendability check
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # extendability check
              [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0], # extendability check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0], # extendability check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0], # extendability check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], # extendability check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1], # extendability check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1], # extendability check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], # extendability check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]] # extendability check

  for_b_ub = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  for_A_eq = [[1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0], # linearity check
              [0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0], # linearity check
              [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0], # linearity check
              [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0], # linearity check
              [1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # E(2,2) check
              [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # E(2,2) check
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0], # E(2,2) check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0], # E(2,2) check
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1]] # extendability check
  for_b_eq = [0, 0, 0, 0, 0, 0, 0, 0, 0]

  return linprog(c=c, A_ub=for_A_ub, b_ub=for_b_ub, A_eq=for_A_eq, b_eq=for_b_eq, bounds=(None, None))

classifier = [0, -4, -4, 0,
              -7, 1, -2, -4,
              -1, -1, -3, 1,
              -6, -2, -3, -5]

print(CheckIfTrueClassifier(classifier))

print()

print(CheckClassifierAgainstSamples(classifier))

# Find a classifier that linearly separates the extendable matrices from the nonextendable matrices using an SVM

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

# Initialize true classifier array
trueClassifiers = np.zeros((1,16))

# Create classifiers and test if they are true classifiers
for i in range(2000):
  # Create a classifier using the ith farthest nonExtendable mapping
  classifier = CreateClassifier(i)

  # Round the classifier (elementwize)
  classifier = RoundClassifier(classifier)

  # Test if the classifier is a true classifier
  result = CheckIfTrueClassifier(classifier)

  if result.success:
    trueClassifiers = np.concatenate((trueClassifiers, classifier), axis=0)

trueClassifiers = trueClassifiers[1:]
print(f"len(trueClassifiers): {len(trueClassifiers)}\n")
print(trueClassifiers)

# Check if the classifiers correctly classify on all the sample data

trueClassifiersGood = np.zeros((1,16))
trueClassifiersBad = np.zeros((1,16))

allMappings = np.concatenate((extendableMappings, nonExtendableMappings), axis=0)
allMappings = allMappings.reshape(len(allMappings), -1)

for k in range(len(trueClassifiers)):
  totalMappingsInClass1 = 0
  totalMappingsInClass2 = 0
  classifiedMappingResults = []

  for i in range(len(allMappings)):
    dotProduct = np.dot(trueClassifiers[k], allMappings[i])
    dotProduct = np.sum(dotProduct)
    extendableOrNot = "E" if i<len(extendableMappings) else "N"
    if dotProduct <= 0:
      # extendable or nonextendable
      # print(f"{extendableOrNot} {dotProduct} (class 1)")
      classifiedMappingResults.append((allMappings[i], extendableOrNot, 1, i, dotProduct))
      totalMappingsInClass1 += 1
    else:
      # nonextendable
      # print(f"{extendableOrNot} {dotProduct} (class 2)")
      classifiedMappingResults.append((allMappings[i], extendableOrNot, 2, i, dotProduct))
      totalMappingsInClass2 += 1

  goodClassifierFlag = True
  for mapping, extendability, _class, mapID, dotProduct in classifiedMappingResults:
    if (extendability == "E" and _class == 2):
      print("Bad classifier, extendableMapping in class 2")
      print(mapping.reshape(4,-1), extendability, _class, mapID, dotProduct)
      goodClassifierFlag = False
      break
    if totalMappingsInClass1 < 1:
      print("Bad classifier, no mappings in class 1")
      goodClassifierFlag = False
      break
    if totalMappingsInClass2 < 1:
      print("Bad classifier, no mappings in class 2")
      goodClassifierFlag = False
      break

  if goodClassifierFlag:
    trueClassifiersGood = np.concatenate((trueClassifiersGood, [trueClassifiers[k]]), axis=0)
  else:
    trueClassifiersBad = np.concatenate((trueClassifiersBad, [trueClassifiers[k]]), axis=0)

  print(f"trueClassifiers[{k}]: \n{trueClassifiers[k].reshape(4,-1)}")
  print(f"totalMappingsInClass1: {totalMappingsInClass1}")
  print(f"totalMappingsInClass2: {totalMappingsInClass2}")
  print()

trueClassifiersGood = trueClassifiersGood[1:]
trueClassifiersBad = trueClassifiersBad[1:]

print(len(trueClassifiersGood))
print(len(trueClassifiersBad))

for i in range(len(trueClassifiersGood)):
  print(f"trueClassifiersGood[{i}]:")
  print(trueClassifiersGood[i].reshape(4,-1))
  print()

# Save trueClassifiersGood to a file
np.save('/content/trueClassifiersGood.npy', trueClassifiersGood)

for i in range(len(extendableMappings[0:100])):
  for row in extendableMappings[i]:
      print(row)
  print(f"{i}th extendable")
  print()

largestMisclassification = 0
largestMisclassificationClassifierIndex = 0

for k in range(len(trueClassifiersBad)):
  totalMappingsInClass1 = 0
  totalMappingsInClass2 = 0
  classifiedMappingResults = []

  for i in range(len(allMappings)):
    dotProduct = np.dot(trueClassifiersBad[k], allMappings[i])
    dotProduct = np.sum(dotProduct)
    dotProductNormalized = dotProduct/(np.linalg.norm(trueClassifiersBad[k]) * np.linalg.norm(allMappings[i]))
    extendableOrNot = "E" if i<len(extendableMappings) else "N"
    if dotProduct <= 0:
      # print(f"{extendableOrNot} {dotProduct} (class 1)")
      classifiedMappingResults.append((allMappings[i], extendableOrNot, 1, i, dotProduct, dotProductNormalized))
      totalMappingsInClass1 += 1
    else:
      # print(f"{extendableOrNot} {dotProduct} (class 2)")
      classifiedMappingResults.append((allMappings[i], extendableOrNot, 2, i, dotProduct, dotProductNormalized))
      totalMappingsInClass2 += 1

  for mapping, extendability, _class, mapID, dotProduct, dotProductNormalized in classifiedMappingResults:
    if (extendability == "E" and _class == 2):
      # print("Bad classifier, extendableMapping in class 2")
      # problemMappings = np.concatenate((problemMappings, [mapping]), axis=0)
      # print(mapping.reshape(4,-1), extendability, _class, mapID, dotProduct, dotProductNormalized)
      if dotProductNormalized > largestMisclassification:
        largestMisclassification = dotProductNormalized
        largestMisclassificationClassifierIndex = k
      # break
    if totalMappingsInClass1 < 1:
      print("Bad classifier, no mappings in class 1")
      break
    if totalMappingsInClass2 < 1:
      print("Bad classifier, no mappings in class 2")
      break

  print(f"trueClassifiersBad[{k}]:")
  print(f"totalMappingsInClass1: {totalMappingsInClass1}")
  print(f"totalMappingsInClass2: {totalMappingsInClass2}")
  print()

print(f"From trueClassifiersBad[{largestMisclassificationClassifierIndex}]")
print(f"Largest misclassification: {largestMisclassification}")

# Load the saved trueClassifiersGood
trueClassifiersGood = np.load('/content/trueClassifiersGood.npy')
print(f"len(trueClassifiersGood): {len(trueClassifiersGood)}")

classifier = np.array([[0, -4, -4, 0],
                       [-7, 1, -2, -4],
                       [-1, -1, -3, 1],
                       [-6, -2, -3, -5]])
for e in extendableMappings:

  flag = False
  if (8*e[1][1])/11 > e[1][2] + e[1][3]:
    # b is positive
    if e[0, 1] < 0:
      flag = True
    if e[0, 2] < 0:
      flag = True
    if e[0, 3] < 0:
      flag = True
    # c is positive
    if e[1, 1] < 0:
      flag = True
    if e[1, 2] < 0:
      flag = True
    if e[1, 3] < 0:
      flag = True
    # d is positive
    if e[3, 1] < 0:
      flag = True
    if e[3, 2] < 0:
      flag = True
    if e[3, 3] < 0:
      flag = True

  if (flag):
    print("BAD CLASSIFICATION")
    print(e)
    break

### 3) Removing Duplicates
# The code below was created by Karim and Darshini

def remove_duplicates(matrix_array):
    seen = set()
    unique_matrices = []

    for matrix in matrix_array:
        matrix_tuple = tuple(matrix.flatten())
        if matrix_tuple not in seen:
            seen.add(matrix_tuple)
            unique_matrices.append(matrix)

    return np.array(unique_matrices)

# Load the provided .npy files
extendable_class = np.load("/content/extendableMappings.npy")
nonextendable_class = np.load("/content/nonExtendableMappings.npy")
goodclassifier_class = np.load("/content/trueClassifiersGood.npy")

# Print initial number of matrices
print("Initial number of extendables: ", len(extendable_class))
print("Initial number of nonextendables: ", len(nonextendable_class))
print("Initial number of good classifiers: ", len(goodclassifier_class))

# Remove duplicates from each class
extendable_class_unique = remove_duplicates(extendable_class)
nonextendable_class_unique = remove_duplicates(nonextendable_class)
classifier_class_unique = remove_duplicates(goodclassifier_class)

# Save the unique matrices to new .npy files
np.save('ExtendableClassUnique.npy', extendable_class_unique)
np.save('NonextendableClassUnique.npy', nonextendable_class_unique)
np.save('ClassifierClassUnique.npy', classifier_class_unique)

# Print number of matrices after removing duplicates
print("\n Number of unique extendables: ", len(extendable_class_unique))
print("Number of unique nonextendables: ", len(nonextendable_class_unique))
print("Number of unique good classifiers: ", len(classifier_class_unique))

print("\n Unique matrices saved.")

extendable_class = np.load("/content/ExtendableClass.npy")
nonextendable_class = np.load("/content/NonextendableClass.npy")
classifier_class = np.load("/content/ClassifierClass.npy")

# Reshaping the classifiers so they're the same format as the other mappings
extendable_class = np.load("/content/extendableMappings.npy")
nonextendable_class = np.load("/content/nonExtendableMappings.npy")
goodclassifier_class = np.load("/content/2900trueGoodClassifiers.npy")

# Reshaping classifiers
reshaped_matrices = []
for i in range(len(goodclassifier_class)):
    reshaped_matrix = goodclassifier_class[i].reshape((4, 4))
    reshaped_matrices.append(reshaped_matrix)
reshaped_goodclassifier_class = np.array(reshaped_matrices)
reshaped_goodclassifier_class = reshaped_goodclassifier_class.astype(int)
goodclassifier_class = reshaped_goodclassifier_class

extendable_class = np.array(extendable_class)
nonextendable_class = np.array(nonextendable_class)
goodclassifier_class = np.array(goodclassifier_class)

print("Extendable count: ", len(extendable_class))
print("Nonextendable count: ", len(nonextendable_class))
print("Classifier count: ", len(goodclassifier_class))