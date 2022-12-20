"""
When training neural networks, the most frequently used algorithm is back
propagation. In this algorithm, parameters (model weights) are adjusted
according to the gradient of the loss function with respect to the given
parameter.

To compute those gradients, PyTorch has a built-in differentiation engine called
torch.autograd. It supports automatic computation of gradient for any
computational graph.

Consider the simplest one-layer neural network, with input x, parameters w and
b, and some loss function. It can be defined in PyTorch in the following manner.
"""

import torch

x = torch.ones(5)   # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# In this network, w and b are parameters, which we need to optimize. Thus, we
# need to be able to compute the gradients of loss function w.r.t. those
# variables. In order to do that, we set the requires_grad property of those
# tensors.

# A function that we apply to tensors to construct a computational graph is in
# fact an object of class Function. This object knows how to compute the
# function in the forward direction, and also how to compute its derivative
# during the backward propagation step. A reference to the backward propagation
# function is stored in grad_fn property of a tensor.

print(f"Gradient function ofr z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")


# To optimize weights of parameters in the neural net, we need to compute the
# derivatives of our loss function w.r.t. parameters, namely, we need d loss / d
# w, d loss / d b under some fixed values of x and y. To compute those
# derivatives, we call loss.backward(), and then retrieve the values from w.grad
# and b.grad:

loss.backward()
print(w.grad)
print(b.grad)

# We can stop tracking gradient computations by surrounding our computation code
# with torch.no_grad() block:

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Another way to achieve the same result is to use the detach() method on the
# tensor.

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

"""
There are reasons you might want to disable gradient tracking:
    - To mark some parameters in your neural net as frozen parameters.
    - To speed up computations when you are only doing forward pass, because
      computations on tensors that do not track gradients would be more
      efficient.
"""
