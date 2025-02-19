from ._utils import *

class DataLoader:
    def __init__(self, dataset, t_eval=None, batch_size=-1, int_cutoff=1.0, shuffle=True, adaptation=False, device='cpu'):

        if isinstance(dataset, str):
            raw_dat = np.load(dataset)
            self.dataset, self.t_eval = np.asarray(raw_dat['X']), np.asarray(raw_dat['t'])
        else:
            self.dataset = dataset
            self.t_eval = t_eval

        self.shuffle = shuffle
        self.device = device

        assert np.ndim(self.dataset) == 4, "Dataset must be of shape (nb_envs, nb_trajs_per_env, nb_steps_per_traj, data_size)"
        assert self.t_eval.shape[0] == self.dataset.shape[2], "t_eval must have the same length as the number of steps in the dataset"

        datashape = self.dataset.shape
        self.nb_envs = datashape[0]
        self.nb_trajs_per_env = datashape[1]
        self.nb_steps_per_traj = datashape[2]
        self.data_size = datashape[3]

        self.int_cutoff = int(int_cutoff*self.nb_steps_per_traj)    ## integration cutoff

        self.dataset = torch.tensor(self.dataset, requires_grad=False, dtype=torch.float32, device=self.device)
        self.t_eval = torch.tensor(self.t_eval, requires_grad=False, dtype=torch.float32, device=self.device)

        if batch_size < 0 or batch_size > self.nb_trajs_per_env:
            self.batch_size = self.nb_trajs_per_env
        else:
            self.batch_size = batch_size

        self.adaptation = adaptation    ## True if this a dataset for adaptation

    def __iter__(self):
        nb_batches = self.nb_trajs_per_env // self.batch_size

        perm_dataset = self.dataset
        for batch_id in range(nb_batches):
            traj_start, traj_end = batch_id*self.batch_size, (batch_id+1)*self.batch_size

            torch_dat = perm_dataset[:, traj_start:traj_end, :self.int_cutoff, :]
            torch_t_eval = self.t_eval[:self.int_cutoff]

            yield torch_dat, torch_t_eval
            # yield torch_dat.to(self.device), torch_t_eval.to(self.device)

    def __len__(self):
        return self.nb_envs * self.nb_trajs_per_env
