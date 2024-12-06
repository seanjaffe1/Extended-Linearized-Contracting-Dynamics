import torch
from manifold_flow.utils import various
import torch.autograd as autograd
import numpy as np

def get_jacobian_eflow(net, x, output_dims, reshape_flag=True):
    """

    """

    if x.ndimension() == 1:
        n = 1
    else:
        n = x.size()[0]
    x_m = x.repeat(1, output_dims).view(-1, output_dims)
    x_m.requires_grad_(True)


    y_m = net(x_m)
    mask = torch.eye(output_dims).repeat(n, 1).to(x.device)
    # y.backward(mask)
    print(mask)
    J = autograd.grad(y_m, x_m, mask, create_graph=True)[0]
    if reshape_flag:
        J = J.reshape(n, output_dims, output_dims)
    return J

def calculate_jacobian(outputs, inputs, create_graph=True):
    """Computes the jacobian of outputs with respect to inputs.

    Based on gelijergensen's code at https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa.

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = torch.autograd.grad(out, inputs, retain_graph=True, create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())


def batch_jacobian(outputs, inputs, create_graph=True):
    """Computes the jacobian of outputs with respect to inputs, assuming the first dimension of both are the minibatch.

    Based on gelijergensen's code at https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa.

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """

    jac = calculate_jacobian(outputs, inputs)
    jac = jac.view((outputs.size(0), np.prod(outputs.size()[1:]), inputs.size(0), np.prod(inputs.size()[1:])))
    jac = torch.einsum("bibj->bij", jac)

    if create_graph:
        jac.requires_grad_()


    return jac

class ELCD(torch.nn.Module):
    def __init__(self, d, x_eq = None, hidden_dim=10, train_x_eq=False):
        super().__init__()
        self.d = d

        #self.x_eq = torch.nn.Parameter(torch.tensor(x_eq, dtype=torch.float32))
        if x_eq is not None:
            self.x_eq = x_eq
            self.x_eq.requires_grad = False
        # self.train_x_eq = train_x_eq
        # self.x_eq.requires_grad = self.train_x_eq 

        self.build_model(d, hidden_dim)
    
    def build_model(self, d, hidden_dim=10):
        
        self.P_s = torch.nn.Sequential(
            torch.nn.Linear(d, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, d **2),
        )

        self.P_a = torch.nn.Sequential(
            torch.nn.Linear(d, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, d **2),
        )

        # initialize the parameters

        self.eps = .01

        # randomly initalize parameters
        self.P_s.apply(self.init_weights)
        self.P_a.apply(self.init_weights)
        


    def set_x_eq(self, x_eq):

        assert x_eq.shape[1] == self.d
        assert x_eq.shape[0]  == 1
        with torch.no_grad():
            self.x_eq = torch.tensor(x_eq)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_A_matrix(self, x):
        b = x.shape[0]
        p_s = self.P_s(x).reshape(b, self.d, self.d)
        p_s_T = p_s.permute(0,2,1)
        
        p_a = self.P_a(x).reshape(b, self.d, self.d)
        p_a_T = p_a.permute(0,2,1)
        I = I = torch.eye(self.d).unsqueeze(0).repeat(b,1,1).to(x.device)
        A = -torch.bmm(p_s_T, p_s)  - self.eps * I + p_a - p_a_T

        if torch.isnan(A).any():
            print("x", x)
            print("p_s", p_s)
            print("p_a", p_a)
            print("A", A)
        return A

    def forward(self, x, x_eq = None):
        assert x.shape[1] == self.d


        A = self.get_A_matrix(x)


        if x_eq is None:
            x_eq = self.x_eq

        x_eq = x_eq.repeat(x.shape[0], 1).to(x.device)
        assert x_eq.shape[0] == x.shape[0]
        assert x_eq.shape[1] == x.shape[1]
        
        # print(A.shape, (x-x_eq).unsqueeze(-1).shape)

        mat = torch.bmm(A, (x-x_eq).unsqueeze(-1)).squeeze(-1)

        # if mat is nan print everything
        if torch.isnan(mat).any():
            print("x", x)
            print("x_eq", x_eq)
            print("A", A)
            print("mat", mat)
        return mat
    
    def forward_discrete(self, x, dt=1, x_eq=None):
        # evolves x forward by dt with first order euler method
        dx_dt =  self.forward(x, x_eq)
        return x + dx_dt * dt
    
class ELCD_Simple(ELCD):    
    def build_model(self, d, hidden_dim=10):
        self.P_s = torch.nn.Parameter(torch.zeros(d, d))
        self.P_a = torch.nn.Parameter(torch.zeros(d, d))

        # initialize the parameters
        torch.nn.init.xavier_uniform_(self.P_s)
        torch.nn.init.xavier_uniform_(self.P_a)


        self.eps = .01

    def get_A_matrix(self, x):
        I = torch.eye(self.d).to(x.device)
        A = -self.P_s.T @ self.P_s  - self.eps * I + self.P_a - self.P_a.T
        return A
    
    def forward(self, x, x_eq = None):
        assert x.shape[1] == self.d
        # assert x.shape[0]  == 1
        # x = x.reshape(self.d)

        A = self.get_A_matrix(x).unsqueeze(0).repeat(x.shape[0], 1, 1)
      
        
        if x_eq is None:
            x_eq = self.x_eq
        else:
            assert x_eq.shape[1] == self.d
            assert x_eq.shape[0]  == 1
        x_eq = x_eq.repeat(x.shape[0], 1)
        
        return torch.bmm(A,  (x-x_eq).unsqueeze(-1)).squeeze(-1)




    


class ELCD_Transform(torch.nn.Module):
    def __init__(self, x_eq, transform, model):
        super().__init__()
        self.transform = transform
        
        self.model = model
        self.set_x_eq(x_eq)
        print("Eq point:", self.x_eq)

    def to(self, device):
        super().to(device)
        self.x_eq = self.x_eq.to(device)
        self.model.to(device)
        self.transform.to(device)
        return self


    def forward(self, x, spec_jacobian = True):
        # Returns approximation of x_

        z_eq = self.transform(self.x_eq)[0]
        # print(x_eq.shape)
        
        z = self.encode(x)

        z_dot_pred = self.model(z, z_eq)

        if spec_jacobian:
            res, inv_jacobian = self.decode(z)
        else:
            res, abs_det = self.decode(z, full_jacobian=False)
            # inv_jacobian = batch_jacobian(res, z)

            inv_jacobian = get_jacobian_eflow(lambda x: self.transform.inverse(x)[0], z, z.shape[1])
            # print("inv_jacobian", inv_jacobian)
            # print("inv_jacobian_e", inv_jacobian_e)
            

        z_dot_pred = z_dot_pred.view(z.shape[0], z.shape[1], 1)
    



        x_dot_pred = torch.bmm(inv_jacobian, z_dot_pred).squeeze(-1)

        return x_dot_pred
    
    def forward_discrete(self, x, dt=1, t_steps = 1):
        ### BROKEN

        
        # evolves x forward by dt with RK45 step
        z_eq = self.transform(self.x_eq)[0]
        # print(x.shape)
        z = self.encode(x)
        print("z", z)
        for _ in range(t_steps):
            z_dot_pred = self.model(z,z_eq)

            print("z_dot_pred", z_dot_pred)
            # k1 = dt * z_dot_pred
            # k2 = dt * self.model(z + k1 / 2, z_eq)
            # k3 = dt * self.model(z + k2 / 2, z_eq)
            # k4 = dt * self.model(z + k3, z_eq)
            # z_next = z + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            z = z + z_dot_pred * dt

        # print("z", z)
        # z_dot_pred = self.model(z, x_eq)
        # print("z-dot", z_dot_pred)

        # next_z = z + z_dot_pred * dt
        # print("next z", next_z)
        # next_x, _ = self.decode(next_z, full_jacobian=False)
        print("next z", z)
        next_x = self.decode(z, full_jacobian=False)[0]
        return next_x

    def encode(self, x):
        h, _ = self.transform(x, full_jacobian=False)
        return h
    
    def decode(self, z, full_jacobian=True):
        # returns decoded z and jacobian of decode transform
        return self.transform.inverse(z, full_jacobian=full_jacobian)
    
    def set_x_eq(self, x_eq):
        # x_eq is a 1xd tensor
        #self.model.set_x_eq(x_eq)
        self.x_eq = x_eq
        self.x_eq.requires_grad = False
        self.model.set_x_eq(self.transform(x_eq)[0]) #TODO Untested
    


