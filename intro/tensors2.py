import torch
import math

# As with any object in Python, assigning a tensor to a variable makes the
# variable a label of the tensor, and does not copy it!
a = torch.ones(2, 2)
b = a

a[0][1] = 561   # we change a...
print(b)        # ... and b is also altered

# The clone() method creates a separate copy.
a = torch.ones(2, 2)
b = a.clone()

assert b is not a       # different objects in memory...
print(torch.eq(a, b))   # ... but still with the same contents!

a[0][1] = 561           # a changes...
print(b)                # ... but b is still all ones

"""
If your source tensor has autograd enabled, then so will the clone.
"""
# If you don't want the cloned copy of your source tensor to track gradients,
# which improves performance, then you can use the .detach() method on the
# source tensor:

a = torch.rand(2, 2, requires_grad=True)    # turn on autograd
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)

print(a)


"""""""""""""""""""""""""""
Manipulating Tensor Shapes
"""""""""""""""""""""""""""

a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)

c = torch.rand(1, 1, 1, 1, 1)
print(c)


a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)


a = torch.ones(4, 3, 2)
b = torch.rand(   3)
c = b.unsqueeze(1)
print(c.shape)
print(a * c)


batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)

output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape( 6 * 20 * 20 )
print(input1d.shape)

# can also call it as a method on the torch module:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)

# When it can, reshape() will return a view on the tensor to be changed - that
# is, a separate tensor object looking at the same underlying region of memory.

import numpy as np

numpy_array = np.ones((2, 3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)

pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)


numpy_array[1, 1] = 23
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)
