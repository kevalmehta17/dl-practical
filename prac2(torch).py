# Import PyTorch library
# PyTorch is used for tensor-based numerical computations
import torch


# -------------------------------
# SCALAR OPERATIONS
# -------------------------------

# Define scalar values using torch.tensor
a = torch.tensor(10)
b = torch.tensor(5)

# Scalar addition
print("Scalar Addition:", a + b)

# Scalar multiplication
print("Scalar Multiplication:", a * b)


# -------------------------------
# VECTOR OPERATIONS
# -------------------------------

# Define vectors as 1-D tensors
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])

# Vector addition
print("Vector Addition:", v1 + v2)

# Element-wise vector multiplication
print("Vector Multiplication:", v1 * v2)

# Dot product of vectors
print("Dot Product:", torch.dot(v1, v2))


# -------------------------------
# MATRIX OPERATIONS
# -------------------------------

# Define matrices as 2-D tensors
m1 = torch.tensor([[1, 2],
                   [3, 4]], dtype=torch.float32)

m2 = torch.tensor([[5, 6],
                   [7, 8]], dtype=torch.float32)

# Matrix addition
print("Matrix Addition:\n", m1 + m2)

# Matrix multiplication
print("Matrix Multiplication:\n", torch.matmul(m1, m2))


# -------------------------------
# TRANSPOSE OF MATRIX
# -------------------------------

# Transpose converts rows into columns
print("Transpose of Matrix:\n", m1.t())


# -------------------------------
# DIAGONAL MATRIX
# -------------------------------

# Create a diagonal matrix
diag = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
print("Diagonal Matrix:\n", diag)


# -------------------------------
# TRIANGULAR MATRICES
# -------------------------------

# Upper triangular matrix
upper = torch.triu(m1)

# Lower triangular matrix
lower = torch.tril(m1)

print("Upper Triangular Matrix:\n", upper)
print("Lower Triangular Matrix:\n", lower)


# -------------------------------
# ORTHOGONAL MATRIX
# -------------------------------

# Identity matrix is an orthogonal matrix
q = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]])

# Check orthogonality: Q × Qᵀ = I
check = torch.matmul(q, q.t())
print("Orthogonal Matrix Check:\n", check)


# -------------------------------
# EIGENVALUES AND EIGENVECTORS
# -------------------------------

# Compute eigenvalues and eigenvectors
eigvals, eigvecs = torch.linalg.eig(m1)

print("Eigenvalues:\n", eigvals)
print("Eigenvectors:\n", eigvecs)
