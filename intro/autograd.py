"""
The power of autograd comes from the fact that it traces your computation
dynamically at runtime, meaning that if your model has decision branches, or
loops whose lengths are not known until runtime, the computation will still be
traced correctly, and you'll get correct gradients to drive learning.
"""

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

a = torch.linspace(0, 2. * math.pi, steps=25, requires_grad=True)
print(a)

b = torch.sin(a)
plt.plot(a.detach(), b.detach())
plt.show()

print(b)

c = 2 * b
print(c)

d = c + 1
print(d)

out = d.sum()
print(out)

print('d:')
print(d.grad_fn)
print(d.grad_fn.next_functions)
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
print('\nc:')
print(c.grad_fn)
print('\nb:')
print(b.grad_fn)
print('\na:')
print(a.grad_fn)

out.backward()
print(a.grad)
plt.plot(a.detach(), a.grad.detach())
plt.show()


BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(1000, 100)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_input = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = TinyModel()
