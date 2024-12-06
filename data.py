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


def get_pendulum_dataset(n,  num_demos = None, discrete = False, normalized = False):
    """
    n: int
    num_demos: int
    discrete: bool
    normalized: bool
    """
    pos, vel = pendulum_to_torch(n)
    eq = torch.zeros((1, 2 * n))
    if discrete:
        if normalized:
            return DiscreteNormalisedTrajData(pos, vel, start=0, pos_eq=eq, num_demos=num_demos)
        else:
            return  DiscreteTrajData(pos, vel, start=0, pos_eq = eq, num_demos=num_demos)
    else:
        if normalized:
            raise NotImplementedError
        else:
            return TrajData(pos, vel, start=0, pos_eq=eq, num_demos=num_demos)


def get_lasa_dataset(dataset_names, start = 15, num_demos = None,discrete = False, normalized = False):
    """
    dataset_names: list of strings
    start: int
    num_demos: int
    normalized: bool
    """
    print("new thing")
    traj_datasets = []
    for name in dataset_names:
        pos, vel = lasa_to_torch(name, start)
        if discrete:
            traj_datasets.append(DiscreteTrajData(pos, vel, start=start, num_demos=num_demos))
        else:
            traj_datasets.append(TrajData(pos, vel, start=start, num_demos=num_demos))
    if discrete:
        if normalized:
            return DiscreteNormalisedTrajData(pos, vel, start=0, num_demos=num_demos)
        else:
            return  DiscreteTrajData(pos, vel, start=0, num_demos=num_demos)

    else:
        if normalized:
            return NormalisedStackedTrajData(traj_datasets, start=start, num_demos=num_demos)
        else:
            return StackedTrajData(traj_datasets, start=start, num_demos=num_demos)

class LASAData(Dataset):
    def __init__(self, dataset_name, start = 15, num_demos = None):
        super().__init__()
        self.pos, self.vel = lasa_to_torch(dataset_name, start)
        if num_demos is not None and 0 < num_demos < self.pos.shape[0]:
            self.pos = self.pos[0:num_demos]
            self.vel = self.vel[0:num_demos]
        self.dataset_name = dataset_name
        self.start = start

        # pos is (num_demos, 2, num_points)
        # vel is (num_demos, 2, num_points)

        self.traj_len = self.pos.shape[2]

        self.pos_eq = self.pos[:, :, -1]
        self.vel_eq = self.vel[:, :, -1]


        self.pos = rearrange(self.pos, 'd c n -> (d n) c').float()
        self.vel = rearrange(self.vel, 'd c n -> (d n) c').float()

        
    def __len__(self):
        return self.pos.shape[0]
    
    def __getitem__(self, idx):
        demo_idx = idx // self.traj_len
        return self.pos[idx], self.vel[idx], self.pos_eq[demo_idx], self.vel_eq[demo_idx]
    

    

class StackedLASAData(Dataset):
    def __init__(self, dataset_names,  start = 15, num_demos = None):
        super().__init__()
        datasets = []
        for dataset_name in dataset_names:
            dataset = LASAData(dataset_name, start=start, num_demos=num_demos)
            datasets.append(dataset)
        
        self.traj_len  = datasets[0].traj_len # should be the same for all 
        
        # stack pos data
        self.pos = torch.cat([dataset.pos for dataset in datasets], dim=1)
        # stack vel data
        self.vel = torch.cat([dataset.vel for dataset in datasets], dim=1)
        #stack eq data
        self.pos_eq = torch.cat([dataset.pos_eq for dataset in datasets], dim=1)
        self.vel_eq = torch.cat([dataset.vel_eq for dataset in datasets], dim=1)
    
        
    def __len__(self):
        return self.pos.shape[0]
    
    def __getitem__(self, idx):
        demo_idx = idx // self.traj_len
        return self.pos[idx], self.vel[idx], self.pos_eq[demo_idx], self.vel_eq[demo_idx]

class NormalisedStackedLASAData(StackedLASAData):
    def __init__(self, dataset_names, start = 15, num_demos = None):
        super().__init__(dataset_names, start, num_demos)
        
        # normalise pos data


        self.pos_mean = self.pos.mean(dim=0)
        self.pos_std = self.pos.std(dim=0)
        # assert pos_mean is of size [len(dataset_names) * 2]


        assert self.pos_mean.shape[0] == len(dataset_names) * 2


        self.pos = (self.pos - self.pos_mean) / self.pos_std

        self.vel = self.vel / self.pos_std


        self.pos_eq = (self.pos_eq - self.pos_mean) / self.pos_std

    def standardize_pos(self, pos):
        return (pos - self.pos_mean) / self.pos_std
    
    def standardize_vel(self, vel):
        return vel / self.vel_std
    
    def unstandardize_pos(self, pos):
        return pos * self.pos_std + self.pos_mean
    
    def unstandardize_vel(self, vel):
        return vel * self.vel_std 
    
class DiscreteNormalisedLASAData(NormalisedStackedLASAData):
    def __init__(self, dataset_names, start = 15, num_demos = None):
        super().__init__(dataset_names=dataset_names, start=start, num_demos=num_demos)
    
    def __getitem__(self, idx):
        if idx % self.traj_len == self.traj_len - 1:
            return self.pos[idx], self.pos[idx]
        return self.pos[idx], self.pos[idx + 1] 




class DiscreteLASAData(Dataset):
    def __init__(self, dataset_name, start = 15, num_demos = None, t=1):
        # t is the number of time steps to predict into the future
        super().__init__()
        self.pos, self.vel = lasa_to_torch(dataset_name, start)
        if num_demos is not None and 0 < num_demos < self.pos.shape[0]:
            self.pos = self.pos[0:num_demos]

        self.dataset_name = dataset_name
        self.start = start

        # pos is (num_demos, 2, num_points)
        # vel is (num_demos, 2, num_points)

        self.traj_len = self.pos.shape[2]

        self.pos_eq = self.pos[:, :, -1]
        
        self.next_pos = self.pos[:, :, t:]
        # add pos_eq to the end of next_pos, t times
        self.next_pos = torch.cat([self.next_pos, self.pos_eq.unsqueeze(2).repeat(1, 1, t)], dim=2)
        self.pos = rearrange(self.pos, 'd c n -> (d n) c')
        self.next_pos = rearrange(self.next_pos, 'd c n -> (d n) c')

    def __len__(self):
        return self.pos.shape[0]
    
    def __getitem__(self, idx):
        return self.pos[idx], self.next_pos[idx]
    
class DiscreteStackedLASAData(Dataset):
    def __init__(self, dataset_names, start = 15, num_demos = None, t=1):
        super().__init__()
        datasets = []
        for dataset_name in dataset_names:
            dataset = DiscreteLASAData(dataset_name, start=start, num_demos=num_demos, t=t)
            datasets.append(dataset)
        
        self.traj_len  = datasets[0].traj_len # should be the same for all 
        
        
        # stack pos data
        self.pos = torch.cat([dataset.pos for dataset in datasets], dim=1)
        # stack vel data
        self.next_pos = torch.cat([dataset.next_pos for dataset in datasets], dim=1)
        #stack eq data
        self.pos_eq = torch.cat([dataset.pos_eq for dataset in datasets], dim=1)

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return self.pos[idx], self.next_pos[idx]
    
class DiscreteNormalisedStackedLASAData(DiscreteStackedLASAData):
    def __init__(self, dataset_names, start=15, num_demos = None, subtract_mean = False, t=1):
        super().__init__(dataset_names, start, num_demos, t=t)
        # normalise pos data
        
        # assert pos_mean is of size [len(dataset_names) * 2]
        self.subtract_mean = subtract_mean
        self.pos_std = self.pos.std(dim=0)
        if subtract_mean:

            self.pos_mean = self.pos.mean(dim=0)
            assert self.pos_mean.shape[0] == len(dataset_names) * 2

            self.pos = (self.pos - self.pos_mean) / self.pos_std
            self.next_pos = (self.next_pos - self.pos_mean) / self.pos_std
            self.pos_eq = (self.pos_eq - self.pos_mean) / self.pos_std
        else:
            self.pos = (self.pos ) / self.pos_std
            self.next_pos = (self.next_pos ) / self.pos_std
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
    


def get_LASA_dataloader(dataset_name, batch_size=1, start = 15, shuffle = True, num_demos = None, ic=False):
    if ic:
        pass
        dataset = LASAData_ic(dataset_name, start, num_demos)
    else:
        dataset = LASAData(dataset_name, start, num_demos)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return data_loader


def get_stacked_LASA_dataloader(dataset_names, batch_size=1, start = 15, shuffle = True, num_demos = None, ic=False):
    
    if ic:
        pass
        dataset = StackedLASAData_ic(dataset_names, start, num_demos)
    else:
        dataset = StackedLASAData(dataset_names, start, num_demos)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return data_loader




class NormalisedStackedLASADataPendulum(Dataset):
    def __init__(self, dataset_path):
        data = np.load(dataset_path)
        X = data['X']  # Assuming this is position data
        Y = data['Y']  # Assuming this represents velocity or other target data
        
        # Convert to torch tensors
        self.pos = torch.tensor(X, dtype=torch.float32)
        self.vel = torch.tensor(Y, dtype=torch.float32)
        
        # Normalize position data
        self.pos_mean = self.pos.mean(dim=0)
        # self.pos_std = self.pos.std(dim=0)
        # self.pos = (self.pos - self.pos_mean) / self.pos_std
        
        # Normalize velocity data similarly if applicable
        # Adapt if velocity has its own mean/std. Using pos mean/std for simplicity
        # self.vel = (self.vel - self.pos_mean) / self.pos_std
        
        # Calculate equilibrium position (pos_eq) - example calculation
        # Here, simply taking the mean as a placeholder. Adjust based on actual requirements.
        # self.pos_eq = self.pos_mean.unsqueeze(0)  # Ensure it is 2D for subscripting

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        return self.pos[idx], self.vel[idx]

    def standardize_X(self, X):
        return (X - self.X_mean) / self.X_std
    
    def standardize_Y(self, Y):
        return (Y - self.Y_mean) / self.Y_std
    
    def unstandardize_X(self, X):
        return X * self.X_std + self.X_mean
    
    def unstandardize_Y(self, Y):
        return Y * self.Y_std + self.Y_mean
    



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


        # self.pos = rearrange(self.pos, 'd c n -> (d n) c').float()
        # self.vel = rearrange(self.vel, 'd c n -> (d n) c').float()

        
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
    
# class DiscreteStackedTrajData(Dataset):
#     def __init__(self, traj_datasets, start = 15, num_demos = None):
#         super().__init__()
#         self.traj_len  = traj_datasets[0].traj_len # should be the same for all 
#         # stack pos data
#         self.pos = torch.cat([dataset.pos for dataset in traj_datasets], dim=1)
#         # stack vel data
#         # self.next_pos = torch.cat([dataset.next_pos for dataset in traj_datasets], dim=1)
#         #stack eq data
#         self.pos_eq = torch.cat([dataset.pos_eq for dataset in traj_datasets], dim=1)
        

#     def __len__(self):
#         return self.pos.shape[0]

#     def __getitem__(self, idx):
#         return self.pos[idx], self.pos[idx + 1]
    
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