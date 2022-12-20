import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

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
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = TinyModel()

print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

prediction = model(some_input)

loss = (ideal_output - prediction).pow(2).sum()
print(loss)

loss.backward()
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])


for i in range(0, 5):
    prediction = model(some_input)
    loss = (ideal_output - prediction).pow(2).sum()
    loss.backward()

print(model.layer2.weight.grad[0][0:10])

optimizer.zero_grad()

print(model.layer2.weight.grad[0][0:10])


a = torch.ones(2, 3, requires_grad=True)
print(a)

b1 = 2 * a
print(b1)

a.requires_grad = False
b2 = 2 * a
print(b2)

a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = a + b
print(c1) 

with torch.no_grad():
    c2 = a + b

print(c2) 

c3 = a * b
print(c3)


def add_tensors1(x, y):
    return x + y

@torch.no_grad()
def add_tensors2(x, y):
    return x + y

a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = add_tensors1(a, b)
print(c1)

c2 = add_tensors2(a, b)
print(c2)


"""
So far, we've used variables to capture the intermediate values of a
computation. Autograd needs these intermediate values to perform gradient
computations. For this reason, you must be careful about using in-place
operations when using autograd. Doing so can destroy information you need to
compute derivatives in the backward() call.
"""

# PyTorch will even stop you if you attempt an in-place operation on a leaf
# variable that requires autograd, as shown below.

a = torch.linspace(0., 2 * math.pi, steps=25, requires_grad=True)
# torch.sin_(a)

# Autograd Profiler
device = torch.device("cpu")
run_on_gpu = False
if torch.cuda.is_available():
    device = torch.device("cuda")
    run_on_gpu = True

x = torch.randn(2, 3, requires_grad=True)
y = torch.rand(2, 3, requires_grad=True)
z = torch.ones(2, 3, requires_grad=True)

with torch.autograd.profiler.profile(use_cuda=run_on_gpu) as prf:
    for _ in range(1000):
        z = (z / x) * y

print(prf.key_averages().table(sort_by='self_cpu_time_total'))


x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 0.1, 0.0001], dtype=torch.float)     # stand-in for gradients.
y.backward(v)

print(x.grad)

"""
The High-Level API
"""

def exp_adder(x, y):
    return 2 * x.exp() + 3 * y

inputs = ( torch.rand(1), torch.rand(1) )  # arguments for the function
print(inputs)
grad = torch.autograd.functional.jacobian(exp_adder, inputs)
print(grad)

inputs = ( torch.rand(3), torch.rand(3) )  # arguments for the function
print(inputs)
grad = torch.autograd.functional.jacobian(exp_adder, inputs)
print(grad)


def do_some_doubling(x):
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    return y

inputs = torch.randn(3)
print(inputs)
my_gradients = torch.tensor([0.1, 1.0, 0.0001])
out = torch.autograd.functional.vjp(do_some_doubling, inputs, v=my_gradients)
print(out)
