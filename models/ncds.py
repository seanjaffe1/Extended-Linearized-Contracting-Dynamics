

import torch
import torch.nn as nn
import numpy as np

# from torchdiffeq import odeint as torch_odeint
# import torchode as to

def scalar_integrate(f, a, b, steps=5):
    """
    Integrate f from a to b using the trapezoidal rule with number of steps steps
    params:
    f: function to integrat (scalar torch function)
        a: start point
        b: end point
        steps: number of steps
    reutrns:
    integral: integral of f from a to b (torch scalar)
    """
    #t = torch.linspace(a, b, int((b - a) / h) + 1).view(-1, 1)
    t = torch.linspace(a, b, steps).view(-1, 1)
#    y=[]
#
#    # todo: can we vectorize this?
#    for i in range(t.shape[0]):
#        y.append(f(t[i]))
#    y = torch.stack(y)
    y = f(t)

    if steps != 1:
        h = (b - a) / (steps - 1)
        integral = h * (torch.sum(y, dim=0) - 0.5 * (y[0] + y[-1]))
    else:
        integral = y[0]
    return integral

def init_weights( m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)



class NCDS_Fast(nn.Module):
    def __init__(self, d, x_0, x_dot_0, hidden_dim=16):
        super().__init__()
        self.d = d

        self.d = d
        self.J_theta = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, d ** 2),
        )
        self.eps = .01
        self.J_theta.apply(init_weights)

        # self.x_0 = torch.nn.Parameter(torch.tensor(x_0).float()).requires_grad_(True)
        # self.x_dot_0 = torch.nn.Parameter(torch.tensor(x_dot_0).float()).requires_grad_(True)
        self.x_0 = torch.zeros_like(x_0)
        self.x_dot_0 = torch.zeros_like(x_dot_0)

    def J(self, x):
        J_theta = self.J_theta(x).reshape(x.shape[0], self.d, self.d)
        r = -torch.bmm(J_theta.permute(0,2,1), J_theta) - self.eps * torch.eye(self.d).to(x.device).repeat(x.shape[0],1,1)
        #r =  - J_theta.T @ J_theta - self.eps * torch.eye(self.d).to(x.device)
        return r
    
    def f(self, t, x):
            batch_size = x.shape[0]
            t_steps = t.shape[0]
            x_0 = self.x_0.repeat(t.shape[0], x.shape[0],1).to(x.device)
            x = x.repeat(t.shape[0], 1, 1)
            t = t.view(-1, 1, 1).to(x.device)
            c_x = ((1-t) * x_0 +  t*  x).view(-1, self.d)
            c_dot_x = (x-x_0).view(-1, self.d)


            y =  torch.bmm(self.J(c_x), c_dot_x.unsqueeze(-1))
            return y.view(t_steps,batch_size,  self.d)

    def forward(self, x):
        #y_0 = self.func(t=0, x=x)
        F = lambda t: self.f(t, x)
        integral = scalar_integrate(F, 0, 1)
        
        return integral + self.x_dot_0

    def forward_discrecte(self, x, dt=1):
        x_dot = self.forward(x)
        next_x = x + x_dot * dt
        return next_x
        
        
        
    


class NCDS_Jacobian(nn.Module):
    def __init__(self, d, hidden_dim_1, ):
        super().__init__()
        # assert hidden_dim_2 is sqrt of true hidden dim of layer 2
        self.d = d
        self.J_theta = nn.Sequential(
            nn.Linear(d, hidden_dim_1),
            nn.Tanh(),
            nn.Linear(hidden_dim_1, d ** 2),
        )
        self.eps = .01
        self.J_theta.apply(init_weights)

    def forward(self, x):
        J_theta = self.J_theta(x).reshape(self.d, self.d)
        r =  - J_theta.T @ J_theta - self.eps * torch.eye(self.d).to(x.device)
        return r

 
class NCDS(nn.Module):
    def __init__(self, d, x_0, x_dot_0, hidden_dim=10):
        super().__init__()
        print("makling ncds")
        self.d = d

        self.func = self.Func(d, x_0, x_dot_0, hidden_dim)
    
    def set_ic(self,x_0, x_dot_0):
        self.func.set_ic(x_0)
        print("set ic", x_0, x_dot_0)

    def forward(self, x):
        #y_0 = self.func(t=0, x=x)
        y_0  = torch.zeros_like(x).float().to(x.device)
        self.func.set_x(x)
   
        j_ = torch_odeint(self.func, y_0, t= torch.tensor([ 0,1]).float())[1]
        
        result = self.func.x_dot_0 + j_
        return result

    class Func(nn.Module):
        def __init__(self, d, x_0, x_dot_0, hidden_dim):
            super().__init__()
            self.J = self.J = NCDS_Jacobian(d, hidden_dim) 
            self.x_0 = torch.nn.Parameter(torch.tensor(x_0).float())
            self.x_dot_0 = torch.nn.Parameter(torch.tensor(x_dot_0).float())


        def set_x (self, x):
            self.x = torch.nn.Parameter(x).clone().detach().requires_grad_(True)

        def set_ic(self, x_0):
            self.x_0 = torch.nn.Parameter(x_0).clone().detach().requires_grad_(True)

        def c(self, x, t):
            c = (1-t) * self.x_0 +  t* x
            return c
            
        def c_dot(self, x, t):
            c_dot = x-self.x_0
            return  c_dot
            

        def forward(self, t, _):
            c = self.c(self.x, t)
            c_dot = self.c_dot(self.x, t)
            r =  (self.J(c)) @ c_dot 
            return r
class NCDS_Transform_Fast(torch.nn.Module):
    def __init__(self,  transform, model):
        super().__init__()
        self.transform = transform
        
        self.model = model

    def to(self, device):
        super().to(device)
        self.model.to(device)
        self.transform.to(device)
        return self


    def forward(self, x):
        z = self.encode(x)
        

        z_dot_pred = self.model(z)
            


        res, inv_jacobian = self.decode(z)
        # print(inv_jacobian.shape, z_dot_pred.shape)
        x_dot_pred = torch.bmm(inv_jacobian, z_dot_pred.unsqueeze(dim=2)).squeeze(-1)

        return x_dot_pred
    
    def set_x_eq(self, x_eq):
        pass
    
    def forward_discrete(self, x, dt=1):
        # x_dot = self.forward(x)
        # next_x = x + x_dot * dt

        z = self.encode(x)
        z_dot_pred = self.model(z)
        z = z + z_dot_pred * dt
        next_x = self.decode(z, full_jacobian=False)[0]
        return next_x

    def encode(self, x):
        h, _ = self.transform(x, full_jacobian=False)
        return h
    
    def decode(self, z, full_jacobian=True):
        # returns decoded z and jacobian of decode transform
        return self.transform.inverse(z, full_jacobian=full_jacobian)

class NCDS_Transform(torch.nn.Module):
    def __init__(self,  transform, model):
        super().__init__()
        self.transform = transform
        
        self.model = model

    def to(self, device):
        super().to(device)
        self.model.to(device)
        self.transform.to(device)
        return self


    def forward(self, x):
        z = self.encode(x)
        batch_size = z.shape[0]
        zs = []
        inv_jacs = []   
        for z_ in z:
            

            z_dot_pred = self.model(z_)
            

            zs.append(z_dot_pred)

        z_dot_pred = torch.stack(zs).view(z.shape[0], z.shape[1], 1)
        res, inv_jacobian = self.decode(z)
        # print(inv_jacobian.shape, z_dot_pred.shape)

        x_dot_pred = torch.bmm(inv_jacobian, z_dot_pred).squeeze(-1)

        return x_dot_pred
    
    def forward_discrete(self, x, dt=1):
        x_dot = self.forward(x)
        next_x = x + x_dot * dt
        return next_x

    def encode(self, x):
        h, _ = self.transform(x, full_jacobian=False)
        return h
    
    def decode(self, z, full_jacobian=True):
        # returns decoded z and jacobian of decode transform
        return self.transform.inverse(z, full_jacobian=full_jacobian)
    




    def set_x_eq(self, x_eq):
        pass
