class Tensor2D:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0

    def dot_product(self, other):
        # Ensure the matrices can be multiplied
        if self.cols != other.rows:
            raise ValueError("The number of columns in the first matrix must equal the number of rows in the second.")

        # Initialize the result matrix
        result = [[0 for _ in range(other.cols)] for _ in range(self.rows)]

        # Perform the dot product
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Tensor2D(result)

    def __repr__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])

