import torch
import numpy as np
import os

# import torch dataset, loader
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import pyLasaDataset as lasa

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

data_set_names = ['Angle',
        'BendedLine',
        'CShape',
        'DoubleBendedLine',
        'GShape',
        'JShape',
        'JShape_2',
        'Khamesh',
        'LShape',
        'Leaf_1',
        'Leaf_2',
        'Line',
        'Multi_Models_1',
        'Multi_Models_2',
        'Multi_Models_3',
        'Multi_Models_4',
        'NShape',
        'PShape',
        'RShape',
        'Saeghe',
        'Sharpc',
        'Sine',
        'Snake',
        'Spoon',
        'Sshape',
        'Trapezoid',
        'WShape',
        'Worm',
        'Zshape',
        'heee']

def load_data_from_config( data_config):
    normalized = data_config['normalized'] if 'normalized' in data_config else False
    num_demos = data_config['num_demos']
    batch_size = data_config['batch_size'] if 'batch_size' in data_config else 1
    if data_config['class'] == 'LASA':
        dataset_names = data_config['label']
        num_demos = data_config['num_demos']


        pos, vel = lasa_to_torch_stacked(dataset_names)
        normalized = True # always normalize
        data_dim = len(dataset_names) * 2
        pos_eq = None


    elif data_config['class'] == 'Pendulum':
        n = data_config['label']
        
        pos, vel = pendulum_to_torch(n)
        pos_eq = torch.zeros((num_demos, 2 * n))


        data_dim = n * 2
    elif data_config['class'] == 'Rosenbrock':
        n = data_config['label']

        path = os.path.join(CURR_DIR, f"data/rosenbrock_{n}.npz")
        loaded_data = np.load(path)
        pos = torch.Tensor(loaded_data['pos'])
        vel = torch.Tensor(loaded_data['vel'])
        pos = pos - 1
        pos_eq = torch.zeros((num_demos, n))
        data_dim = n
    else:
        raise NotImplementedError()
    if normalized:
        dataset = NormalisedTrajData(pos, vel, num_demos=num_demos, pos_eq=pos_eq)
    else:
        dataset = TrajData(pos, vel, num_demos=num_demos, pos_eq=pos_eq)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x_eq = dataset.pos_eq[0:1].float()
    return train_loader, data_dim, x_eq


def pendulum_to_torch(n):
    path = os.path.join(CURR_DIR, f"data/pendulum/n-{n}.npz")
    data = np.load(path)
    my_X_phy = data['my_X_phy']
    my_X_dot_phy = data['my_X_dot_phy']
    my_X_next = data['my_X_next']

    pos = torch.tensor(my_X_phy, dtype=torch.float32).permute(1,2,0)
    vel = torch.tensor(my_X_dot_phy, dtype=torch.float32).permute(1,2,0)
    return pos, vel

def lasa_to_torch(dataset_name, start = 15):
    assert dataset_name in data_set_names
    lasa_data = getattr(lasa.DataSet, dataset_name)
    demos = lasa_data.demos
    pos = torch.tensor(np.array([demo.pos for demo in demos]))[:, :, start:].float()
    vel = torch.tensor(np.array([demo.vel for demo in demos]))[:, :, start:].float()
    return pos, vel



def lasa_to_torch_stacked(dataset_names, start = 15):

    pos_list = []
    vel_list = []
    for dataset_name in dataset_names:
        assert dataset_name in data_set_names
        lasa_data = getattr(lasa.DataSet, dataset_name)
        demos = lasa_data.demos
        pos = torch.tensor(np.array([demo.pos for demo in demos]))[:, :, start:].float()
        vel = torch.tensor(np.array([demo.vel for demo in demos]))[:, :, start:].float()
        pos_list.append(pos)
        vel_list.append(vel)
    pos = torch.cat(pos_list, dim=1)
    vel = torch.cat(vel_list, dim=1)
    return pos, vel

def get_rosenbrock_data(n=16, discrete = True):
    path = os.path.join(CURR_DIR, f"data/rosenbrock_{n}.npz")
    loaded_data = np.load(path)
    pos = loaded_data['pos']
    vel = loaded_data['vel']
    pos_eq = torch.ones((1, n))
    if discrete:
        return DiscreteTrajData(torch.Tensor(pos), torch.Tensor(vel), pos_eq = pos_eq)
    else:
        return TrajData(torch.Tensor(pos), torch.Tensor(vel),pos_eq = pos_eq)



class TrajData(Dataset):
    def __init__(self, pos, vel, pos_eq = None, pos_vel = None, start = 15, num_demos = None):
        # sets equilibrium pos and vel to the last pos and vel in the trajectory if not given
        
        super().__init__()
        self.pos, self.vel = pos, vel
        if num_demos is not None and 0 < num_demos < self.pos.shape[0]:
            self.pos = self.pos[0:num_demos]
            self.vel = self.vel[0:num_demos]
        self.start = start



        self.traj_len = self.pos.shape[2]

        self.pos_eq = pos_eq if pos_eq is not None else self.pos[:, :, -1]
        self.vel_eq = pos_vel if pos_vel is not None else self.vel[:, :, -1]




        
    def __len__(self):
        return self.pos.shape[0] * self.traj_len
    
    def __getitem__(self, idx):
        return self.pos[idx // self.traj_len,:, idx % self.traj_len], self.vel[idx // self.traj_len,:,  idx % self.traj_len]




class NormalisedTrajData(TrajData):
    def __init__(self, pos, vel, start = 15, pos_eq=None, num_demos = None, subtract_mean = False):
        
        # normalise pos data
        super().__init__(pos, vel, start=start, pos_eq=pos_eq, num_demos=num_demos)

        self.subtract_mean = subtract_mean
        self.pos_std = self.pos.std(dim=(0,2))
        if subtract_mean:

            self.pos_mean = self.pos.mean(dim=(0,2))
            assert self.pos_mean.shape[0] ==  self.pos.shape[1]

            self.pos = (self.pos - self.pos_mean.unsqueeze(1)) / self.pos_std.unsqueeze(1)
            # self.next_pos = (self.next_pos - self.pos_mean) / self.pos_std
            # print(self.pos_eq.shape, self.pos_mean.shape, self.pos_std.shape)
            self.pos_eq = (self.pos_eq - self.pos_mean) / self.pos_std
        else:
            self.pos = (self.pos ) / self.pos_std.unsqueeze(1)
            # self.next_pos = (self.next_pos ) / self.pos_std
            self.pos_eq = (self.pos_eq ) / self.pos_std


        self.vel = self.vel / self.pos_std.unsqueeze(1)

    def standardize_pos(self, pos):
        return (pos - self.pos_mean.unsqueeze(1)) / self.pos_std.unsqueeze(1)
    
    def standardize_vel(self, vel):
        return vel / self.pos_std.unsqueeze(1)
    
    def unstandardize_pos(self, pos):
        return pos * self.pos_std.unsqueeze(1) + self.pos_mean.unsqueeze(1)
    
    def unstandardize_vel(self, vel):
        return vel * self.pos_std.unsqueeze(1)
    


class DiscreteTrajData(Dataset):
    def __init__(self, pos, vel, start = 15, pos_eq = None, num_demos = None, t_steps = 1):
        super().__init__()
        self.pos, self.vel = pos, vel

        if num_demos is not None and 0 < num_demos < self.pos.shape[0]:
            self.pos = self.pos[0:num_demos]
            self.vel = self.vel[0:num_demos]
        self.start = start
        self.traj_len = self.pos.shape[2] - 1
        self.pos_eq = self.pos[:, :, -1] if pos_eq is None else pos_eq

        self.t_steps = t_steps
  

    def __len__(self):
        return self.traj_len * (self.pos.shape[0])
    
    def __getitem__(self, idx):
        # pos is (num_demos, 2, num_points)
        if idx % self.traj_len == self.traj_len - self.t_steps:
            return self.pos[idx // self.traj_len,:, idx % self.traj_len], self.pos[idx // self.traj_len,:,  self.traj_len]

        return self.pos[idx // self.traj_len,:, idx % self.traj_len], self.pos[idx // self.traj_len,:,  idx % self.traj_len + 1]
    

    
class DiscreteNormalisedTrajData(DiscreteTrajData):
    def __init__(self, pos, vel, start = 15, pos_eq = None, num_demos = None, subtract_mean = False, t_steps=1):
        super().__init__(pos, vel, start=start, pos_eq=pos_eq, num_demos=num_demos, t_steps=t_steps)
        # normalise pos data
        
        # assert pos_mean is of size [len(dataset_names) * 2]
        self.subtract_mean = subtract_mean
        self.pos_std = self.pos.std(dim=(0,2))
        if subtract_mean:

            self.pos_mean = self.pos.mean(dim=(0,2))
            assert self.pos_mean.shape[0] ==  self.pos.shape[1]

            self.pos = (self.pos - self.pos_mean.unsqueeze(1)) / self.pos_std.unsqueeze(1)
            # self.next_pos = (self.next_pos - self.pos_mean) / self.pos_std
            # print(self.pos_eq.shape, self.pos_mean.shape, self.pos_std.shape)
            self.pos_eq = (self.pos_eq - self.pos_mean) / self.pos_std
        else:
            self.pos = (self.pos ) / self.pos_std.unsqueeze(1)
            # self.next_pos = (self.next_pos ) / self.pos_std
            self.pos_eq = (self.pos_eq ) / self.pos_std

    def transform(self, x):
        if self.subtract_mean:
            return (x - self.pos_mean) / self.pos_std
        else:
            return x / self.pos_std
    
    def untransform(self, x):
        if self.subtract_mean:
            return x * self.pos_std + self.pos_mean
        else:
            return x * self.pos_std