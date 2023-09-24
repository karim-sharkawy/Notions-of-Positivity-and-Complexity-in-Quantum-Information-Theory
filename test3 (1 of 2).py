import random # Generates pseudo-random numbers

def get_random_number():
    return random.randrange(10)

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

