import torch
import numpy as np

# Create tensor from data directly
data = [[1,2], [3,4], [5,6]]
X_data = torch.tensor(data)

# Create tensor from numpy array
np_array = np.array(data)
X_np = torch.from_numpy(np_array)

# Create tensor from another tensor
X_ones = torch.ones_like(X_data)    # retains the properties of x_data
print(f"Ones tensor: \n {X_ones} \n")

X_random = torch.rand_like(X_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random tensor: \n {X_random} \n")

# With random or constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
print(f"Random Tensor: \n {rand_tensor} \n")

ones_tensor = torch.ones(shape)
print(f"Ones Tensor: \n {ones_tensor} \n")

zeros_tensor = torch.zeros(shape)
print(f"Zeros Tensor: \n {zeros_tensor}")

# Attributes of a Tensor
tensor = torch.rand(4,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Operations on Tensors
print(f"Tensor: \n {tensor} \n")
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(f"Tensor: \n {tensor} \n")

# Joining tensors
t1 = torch.cat([tensor, tensor], dim=-2)
print(t1)

# Arithmetic operations
tensor = torch.rand(4,4)
print(f"Tensor: \n {tensor} \n")
tensor2 = torch.rand(4,4)
print(f"Tensor2: \n {tensor2} \n")
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor2.T
y2 = tensor.matmul(tensor2.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor2.T, out=y3)
print(f"y1: \n {y1} \n")


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor2
z2 = tensor.mul(tensor2)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor2, out=z3)
print(f"z1: \n {z1} \n")

# Single-element tensors
agg = tensor.sum()
print(f"agg: \n {agg} \n")
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations
# Operations that store the result into the operand are called in-place.
# They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
# A change in the tensor reflects in the NumPy array.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)
# Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")