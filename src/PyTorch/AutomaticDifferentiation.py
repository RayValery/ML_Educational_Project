# https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)


# Tensor Gradients and Jacobian Products
# In many cases, we have a scalar loss function, and we need to compute the gradient with respect to some parameters.
# However, there are cases when the output function is an arbitrary tensor. In this case, PyTorch allows you to compute
# so-called Jacobian product, and not the actual gradient.
#
# Instead of computing the Jacobian matrix itself, PyTorch allows you to compute Jacobian Product for a given input vector.
# This is achieved by calling backward with v as an argument. The size of v  should be the same as the size of the original tensor,
# with respect to which we want to compute the product:
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")