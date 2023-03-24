import torch
import numpy as np
from torch.autograd import Variable

np.random.seed(0);
a1 = np.random.rand(2,2);

torch.manual_seed(0);
a2 = torch.rand(2,2);

#print(a1,a2);

a3_nparray = np.ones((2,2));
#print(a3_nparray);
#print(type(a3_nparray));
a3_torcharray = torch.from_numpy(a3_nparray);
#print(a3_torcharray)

if torch.cuda.is_available():
    a3_cuda = a3_torcharray.cuda()

a3_resized = a3_cuda.view(4);
#print(a3_resized)

b1 = torch.ones(2,2);
b2 = b1+b1;
b1.add_(b2);

b3 = b1+b2;
b4 = b3.mul(b2);
b5 = b4.div(b1);
print(b1,b2,b3,b4,b5);


a1 = Variable(torch.Tensor([0.3, 0.3]),requires_grad = True); a2 = Variable(torch.t(a1),requires_grad = True);
a3 = torch.outer(a1,a2);
a3_norm = torch.norm(a3);
a3_norm.backward();
grad_a3_norm = a1.grad;
print(grad_a3_norm)
