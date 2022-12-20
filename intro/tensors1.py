import torch
import math

# The simplest way to create a tensor is with the torch.empty() call.
x = torch.empty(3, 4)
print(type(x))
print(x)

zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)

x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)



ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)

# Tensor broadcasting
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)

a = torch.ones(4, 3, 2)

b = a * torch.rand( 3, 2 )  # 3rd and 2nd dims identical to a, dim 1 absent
print(b)

c = a * torch.rand( 3, 1 )  # 3rd dim = 1, 2nd dim identical to a
print(c)

d = a * torch.rand( 1, 2 )  # 3rd dim identical to a, 2nd dim = 1
print(d)


a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.rand(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)

assert c is d        # test c & d are the same object, not just containing equal values
assert id(c), old_id # make sure that our new c is the same object as the old one

print(id(c))
print(old_id)
print(id(d))

torch.rand(2, 2, out=c)     # works for creation too!
print(c)
assert id(c), old_id        # still the same object!
