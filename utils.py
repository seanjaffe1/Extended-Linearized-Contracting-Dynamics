import numpy

import matplotlib.pyplot as plt
from scipy.integrate import odeint
# from torchdiffeq import odeint as odeint_torch
import torch
import numpy as np

from manifold_flow.vector_transforms import create_vector_transform
from manifold_flow.transforms.projections import Projection, CompositeProjection
from models import ELCD, ELCD_Transform, NCDS_Fast, NCDS_Transform_Fast


def plot_manifold(transform, scale, grid_step_size, axis_step_size):
    """
    Plots the manifold by passing in a grid into transform
    params:
        transform: Transoform object
        scale: plot from -scale to scale on x and y axis
        grid_step_size: distance between grid lines
        axis_step_size: distance between points on each grid line
    """
    xs = np.arange(-scale, scale, grid_step_size)
    ys = np.arange(-scale, scale, grid_step_size)

    lines = []
    for x in xs:
        y_vals = np.arange(-scale, scale, axis_step_size)
        x_vals = [x] * len(y_vals)
        line = np.array([x_vals, y_vals])
        lines.append(line)

    for y in ys:
        x_vals = np.arange(-scale, scale, axis_step_size)
        y_vals = [y] * len(x_vals)
        line = np.array([x_vals, y_vals])
        lines.append(line)

    transformed_lines = []
    for line in lines:
        torch_line = torch.Tensor(line).T
        torch_line_transformed = transform(torch_line)[0].T.detach().numpy()
        transformed_lines.append(torch_line_transformed)
    
    
    for line in transformed_lines:
        plt.plot(line[0], line[1], c='b')
    plt.show()



def plot_2d_vector_field(model, x_min, x_mx, y_min, y_max, normalize =True, scale = None, points = 20):
    """
    plot the vector field of a function f within the given range.
    :params
        x_min: float, the minimum value of x
        x_max: float, the maximum value of x
        y_min: float, the minimum value of y
        y_max: float, the maximum value of y
        model_d: number of dimensions of the model data
    :return:
        none
    """
    fig_size = (8, 8)
    f, ax = plt.subplots(figsize=fig_size)
    x = numpy.linspace(x_min, x_mx, points)
    y = numpy.linspace(y_min, y_max, points)
    xy =  numpy.meshgrid(x, y)
    points_reshaped = np.array(xy).reshape(2, -1).T
    points_reshaped = torch.Tensor(points_reshaped)

    vector_field = model(points_reshaped)
    U = vector_field[:, 0].view(points, points).detach().numpy()
    V = vector_field[:, 1].view(points, points).detach().numpy()


    if normalize:
        U_ = U / numpy.sqrt(U ** 2 + V ** 2)
        V = V / numpy.sqrt(U ** 2 + V ** 2)
        U = U_
    if scale is not None:
        ax.quiver(xy[0], xy[1], U, V, units='width', scale = scale)
    else:
        ax.quiver(xy[0], xy[1], U, V)
    # draw the x-axis and y-axis, thin
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlim([x_min, x_mx])
    ax.set_ylim([y_min, y_max])
    return ax

    

def solve_ivp(f, x_0, t_start, t_end, t_step = 1):
    """
    generate the flow data of a function f from x_0 for t=t_range
    params:
        f: function
        x_0: numpy array, the initial point
        t_range: numpy array, the time range
    return:
        X: numpy array, the flow data
        x_dot, numpy array, the derivative of the flow data
    """
    t = numpy.arange(t_start, t_end, t_step)
    X = odeint(f, x_0, t)
    x_dot = numpy.zeros(X.shape)
    for i in range(X.shape[0]):
        x_dot[i] = f(X[i], 0)
    return X, x_dot

def solve_ivp_no_dot(f, x_0, t_start, t_end, t_step = 1):
    """
    generate the flow data of a function f from x_0 for t=t_range
    params:
        f: function
        x_0: numpy array, the initial point
        t_range: numpy array, the time range
    return:
        X: numpy array, the flow data
        x_dot, numpy array, the derivative of the flow data
    """
    t = numpy.arange(t_start, t_end, t_step)
    X = odeint(f, x_0, t)
    # x_dot = numpy.zeros(X.shape)
    # for i in range(X.shape[0]):
    #     x_dot[i] = f(X[i], 0)
    return X

def model_to_f(model):
    # this function converts a model to a numpy function
    return lambda x, t: model(torch.Tensor(x))

def single_ode45_step(f, x, dt):
    """
    solve the ode using ode45 method
    """
    k1 = dt * f(x)
    k2 = dt * f(x + k1 / 2)
    k3 = dt * f(x + k2 / 2)
    k4 = dt * f(x + k3)
    y =  x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


def forward_step(f, x0, t_steps, dt):
    """
    solve the ode using forward euler method
    :params
        f: dscrete time function f(x(t), dt) = x(t+dt) where x is (b, d)
        x0: numpy array, the initial point
        t_steps: int, the number of steps
        dt: float, the time step
    :return:
        X: numpy array, the flow data
    """
    X = torch.zeros((t_steps, x0.shape[0], x0.shape[1]))
    X[0] = x0
    for i in range(1, t_steps):
        X[i] = f(X[i - 1], dt)
    return X
    
def forward_step_until_converge(f, x0, dt, x_eq = None, max_steps = 1000, tol = 1e-5):
    """
    solve the ode using forward euler method
    :params
        f: dscrete time function f(x(t), dt) = x(t+dt) where x is (b, d)
        x0: numpy array, the initial point
        t_steps: int, the number of steps
        dt: float, the time step
    :return:
        X: numpy array, the flow data
    """
    # print("running with ode 45")
    X = torch.zeros((max_steps, x0.shape[0], x0.shape[1]))
    X[0] = x0
    for i in range(1, max_steps):
        X[i] = f(X[i - 1], dt)
        # X[i] = single_ode45_step(f, X[i - 1], dt)
        if torch.norm(X[i] - X[i-1]) < tol:
            break
    return X[:i+1]


def DTWD(trajectory, reference):
    """
    calculate the DTW distance between trajectory and reference
    :params
        trajectory: torch array [n,d], the trajectory
        reference: list of N torch [d] arrays
    :return:
        DTWD distance
    """

    def dist(x, y):
        return torch.norm(x - y)

    n = trajectory.shape[0]
    N = len(reference)
    dists = torch.zeros((n, N))
    for i in range(n):
        for j in range(N):
            dists[i, j] = dist(trajectory[i], reference[j])
    
    min_x = torch.min(dists, dim=0).values / N
    min_y = torch.min(dists, dim=1).values / n
    assert min_x.shape[0] == N
    assert min_y.shape[0] == n
    return (min_x.sum() + min_y.sum()).item()

def dtwd_np(traj, reference):
    assert traj.shape[1] == reference[0].shape[0]
    def distance(x, y):
        return numpy.linalg.norm(x - y)
    
    n = traj.shape[0]
    N = len(reference)
    dists = numpy.zeros((n, N))
    for i in range(n):
        for j in range(N):
            dists[i, j] = distance(traj[i], reference[j])
    # min_x = numpy.min(dists, axis=0) / N
    # min_y = numpy.min(dists, axis=1) / n
    # return (numpy.sum(min_x) + numpy.sum(min_y))
    return numpy.mean(numpy.min(dists, axis=0)) + numpy.mean(numpy.min(dists, axis=1))


def generate_eflow_transform(input_dim:int, num_blocks:int = 10, num_hidden:int=200, sigma:float=.45,  s_act=None, t_act=None, coupling_network_type:str = "rffn"):
    
    taskmap_net = BijectionNet(num_dims=input_dim, num_blocks=num_blocks, num_hidden=num_hidden, s_act=s_act, t_act=t_act,
                           sigma=sigma,
                           coupling_network_type=coupling_network_type)

    return EFlowTransformWrapper(taskmap_net)

def generate_mf_transform(
        flow_steps: int, 
        input_dim: int , 
        latent_dim: int, 
        linear_transform_type:str ="lu", 
        base_transform_type="rq-coupling", 
        num_bins=10) -> CompositeProjection:

    rq_transform = create_vector_transform( input_dim,
        flow_steps,
        linear_transform_type=linear_transform_type,
        base_transform_type=base_transform_type,
        hidden_features=30,
        num_transform_blocks=2,
        dropout_probability=0.0,
        use_batch_norm=False,
        num_bins=num_bins,
        tail_bound=10,
        apply_unconditional_transform=False,
        context_features=None)
    return CompositeProjection(rq_transform, input_dim, latent_dim)

def generate_ncds_mf_transform(flow_steps: int, input_dim: int, latent_dim: int, hidden_dim: int) -> NCDS_Transform_Fast:
    
    transform = generate_mf_transform(flow_steps=flow_steps, input_dim=input_dim, latent_dim=latent_dim)
    x_0 = torch.zeros(latent_dim)
    x_dot_0 = torch.ones(latent_dim)
    ncds = NCDS_Fast(d=latent_dim,  hidden_dim = hidden_dim, x_0=x_0, x_dot_0=x_dot_0)

    return NCDS_Transform_Fast( transform=transform, model=ncds)

def generate_ELCD_mf_transform(flow_steps: int, input_dim: int, latent_dim: int, hidden_dim: int, x_eq: torch.FloatTensor) -> ELCD_Transform:

    transform = generate_mf_transform(flow_steps=flow_steps, input_dim=input_dim, latent_dim=latent_dim)
    elcd = ELCD(d=latent_dim, x_eq = None, hidden_dim = hidden_dim)

    return ELCD_Transform(x_eq=x_eq, transform=transform, model=elcd)
    

# def generate_ELCD_eflow_transform(input_dim:int, hidden_dim:int, x_eq:torch.FloatTensor,  num_blocks:int = 10, num_hidden:int=200, sigma:float=.45,  s_act=None, t_act=None, coupling_network_type:str = "rffn"):
#     """
#     Generate a ELCD transform with a eflow transform
#     """
#     transform = generate_eflow_transform(input_dim=input_dim, num_blocks=num_blocks, num_hidden=num_hidden, sigma=sigma, s_act=s_act, t_act=t_act, coupling_network_type=coupling_network_type)
#     elcd = ELCD(d=input_dim, x_eq = x_eq, hidden_dim = hidden_dim)
#     return ELCD_Transform(x_eq=x_eq, transform=transform, model=elcd)

def generate_sdd(input_dim: int, device='cpu'):
    lsd = input_dim
    h_dim = 100
    ph_dim = 60
    fhat = nn.Sequential(nn.Linear(lsd, h_dim), nn.ReLU(),
                            nn.Linear(h_dim, h_dim), nn.ReLU(),
                            nn.Linear(h_dim, lsd))
    v = ICNN([lsd, ph_dim, h_dim, 1])
    alpha = 0.01

    model = Dynamics(fhat, v, alpha).to(device)

    # def model_wrapper(x):
    #     x = torch.autograd.Variable(x).requires_grad_(True)
    #     return model(x)
    

    return model

def generate_eflow(input_dim:int, x_eq, num_blocks:int = 10, num_hidden:int=200, sigma:float=.45,  s_act=None, t_act=None, coupling_network_type:str = "rffn", eps=1e-5, device='cpu'):
    taskmap_net = BijectionNet(num_dims=input_dim, num_blocks=num_blocks, num_hidden=num_hidden, s_act=s_act, t_act=t_act,
                           sigma=sigma,
                           coupling_network_type=coupling_network_type)


    y_pot_grad_fcn = lambda y: F.normalize(y)   # potential fcn gradient (can use quadratic potential instead)

    # pulled back dynamics (natural gradient descent system)
    euclideanization_net = NaturalGradientDescentVelNet(taskmap_fcn=taskmap_net,
                                                        grad_potential_fcn=y_pot_grad_fcn,
                                                        origin=x_eq,
                                                        scale_vel=True,
                                                        is_diffeomorphism=True,
                                                        n_dim_x=input_dim,
                                                        n_dim_y=input_dim,
                                                        eps=eps,
                                                        device=device)
    return euclideanization_net

def get_model_from_config(config:dict, data_dim:int, x_eq:torch.FloatTensor)->torch.nn.Module:
    model_type = config['type'] if 'type' in config else config['model_type']
    if model_type == "ELCD":
        return  generate_ELCD_mf_transform(flow_steps=config['flow_steps'], input_dim=data_dim, latent_dim=config['latent_dim'], hidden_dim=config['hidden_dim'], x_eq=x_eq)
    elif model_type == "NCDS":
        return  generate_ncds_mf_transform(flow_steps=config['flow_steps'], input_dim=data_dim, latent_dim=config['latent_dim'], hidden_dim=config['hidden_dim'])
    elif model_type == "SDD":
        return  generate_sdd(data_dim)
    elif model_type == "EFLOW":
        return generate_eflow(input_dim=data_dim, x_eq=x_eq)
    else:
        raise ValueError(f"Model type {model_type} not supported")


