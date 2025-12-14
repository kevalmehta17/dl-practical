# Import TensorFlow library
# TensorFlow is used for tensor-based mathematical operations
import tensorflow as tf


# -------------------------------
# SCALAR OPERATIONS
# -------------------------------

# Define scalar values using tf.constant
a = tf.constant(10)
b = tf.constant(5)

# Perform scalar addition
print("Scalar Addition:", a + b)

# Perform scalar multiplication
print("Scalar Multiplication:", a * b)


# -------------------------------
# VECTOR OPERATIONS
# -------------------------------

# Define vectors as 1-D tensors
v1 = tf.constant([1, 2, 3])
v2 = tf.constant([4, 5, 6])

# Vector addition
print("Vector Addition:", v1 + v2)

# Element-wise vector multiplication
print("Vector Multiplication:", v1 * v2)

# Dot product of vectors
# axes=1 means sum of element-wise multiplication
print("Dot Product:", tf.tensordot(v1, v2, axes=1))


# -------------------------------
# MATRIX OPERATIONS
# -------------------------------

# Define matrices as 2-D tensors
m1 = tf.constant([[1, 2],
                  [3, 4]], dtype=tf.float32)

m2 = tf.constant([[5, 6],
                  [7, 8]], dtype=tf.float32)

# Matrix addition
print("Matrix Addition:\n", m1 + m2)

# Matrix multiplication
print("Matrix Multiplication:\n", tf.matmul(m1, m2))


# -------------------------------
# TRANSPOSE OF MATRIX
# -------------------------------

# Transpose swaps rows and columns
print("Transpose of Matrix:\n", tf.transpose(m1))


# -------------------------------
# DIAGONAL MATRIX
# -------------------------------

# Create a diagonal matrix from a vector
diag = tf.linalg.diag([1.0, 2.0, 3.0])
print("Diagonal Matrix:\n", diag)


# -------------------------------
# TRIANGULAR MATRICES
# -------------------------------

# Upper triangular matrix (elements below main diagonal are zero)
upper = tf.linalg.band_part(m1, 0, -1)

# Lower triangular matrix (elements above main diagonal are zero)
lower = tf.linalg.band_part(m1, -1, 0)

print("Upper Triangular Matrix:\n", upper)
print("Lower Triangular Matrix:\n", lower)


# -------------------------------
# ORTHOGONAL MATRIX
# -------------------------------

# Identity matrix is an example of an orthogonal matrix
q = tf.constant([[1.0, 0.0],
                 [0.0, 1.0]])

# For orthogonal matrix: Q × Qᵀ = I
check = tf.matmul(q, tf.transpose(q))
print("Orthogonal Matrix Check:\n", check)


# -------------------------------
# EIGENVALUES AND EIGENVECTORS
# -------------------------------

# Compute eigenvalues and eigenvectors of matrix m1
eigvals, eigvecs = tf.linalg.eig(m1)

print("Eigenvalues:\n", eigvals)
print("Eigenvectors:\n", eigvecs)
