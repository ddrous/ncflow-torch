
# %%

""" This ia a script to generate data for the Selkov system. """

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import Image

import diffrax
from nodax import RK4
import jax.numpy as jnp

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

## Parse the three arguments from the command line: "train", the foldername, and the seed

import argparse


if _in_ipython_session:
	# args = argparse.Namespace(split='train', savepath='tmp/', seed=42)
	args = argparse.Namespace(split='test', savepath="tmp/", seed=2026, verbose=1)
else:
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('--split', type=str, help='Generate "train", "test", "adapt", "adapt_test", or "adapt_huge" data', default='train', required=False)
	parser.add_argument('--savepath', type=str, help='Description of optional argument', default='tmp/', required=False)
	parser.add_argument('--seed',type=int, help='Seed to gnerate the data', default=2026, required=False)
	parser.add_argument('--verbose',type=int, help='Whether to print details or not ?', default=1, required=False)

	args = parser.parse_args()


split = args.split
assert split in ["train", "test", "adapt", "adapt_test", "adapt_huge"], "Split must be either 'train', 'test', 'adapt', 'adapt_test', 'adapt_huge'"

savepath = args.savepath
seed = args.seed

if args.verbose != 0:
  print("Running this script in ipython (Jupyter) session ?", _in_ipython_session)
  print('=== Parsed arguments to generate data ===')
  print(' Split:', split)
  print(' Savepath:', savepath)
  print(' Seed:', seed)
  print()


## Set numpy seed for reproducibility
np.random.seed(seed)



#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

import jax.numpy as jnp

# Define the Selkov system: https://en.wikipedia.org/wiki/Hopf_bifurcation
def selkov(t, y, a, b):
    x, y = y
    dx = -x + a*y + (x**2)*y
    dy = b - a*y - (x**2)*y
    return np.array([dx, dy])

if split == "train" or split=="test":
  ## Meta-training environments
  environments = [(0.1, b) for b in list(np.linspace(-1, -0.25, 7))\
        + list(np.linspace(-0.1, 0.1, 7))\
        + list(np.linspace(0.25, 1., 7))]
elif split == "adapt" or split == "adapt_test":
  ## Meta-testing environments
  environments = [(0.1, b) for b in [-1.25, -0.65, -0.05, 0.02, 0.6, 1.2]]


if split == "train":
  n_traj_per_env = 4     ## training
elif split == "test" or split == "adapt_test" or split == "adapt_huge":
  n_traj_per_env = 4     ## testing (both in-domain and out-of-domain)
elif split == "adapt":
  n_traj_per_env = 1     ## adaptation

n_steps_per_traj = 11
data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))

# Time span for simulation
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj)  # Fewer frames

## The seed is used for different splits
max_seed = np.iinfo(np.int32).max

## Generate data for each environment
for j in range(n_traj_per_env):
    np.random.seed(j if not split not in ["test", "adapt_test", "adapt_huge"] else max_seed - j)
    initial_state = np.random.uniform(0, 3, 2)

    for i, selected_params in enumerate(environments):
        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(selkov, t_span, initial_state, args=(selected_params[0], selected_params[1]), t_eval=t_eval)
        data[i, j, :, :] = solution.y.T

# Save t_eval and the solution to a npz file
if split == "train":
  filename = savepath+'train_data.npz'
elif split == "test":
  filename = savepath+'test_data.npz'
elif split == "adapt":
  filename = savepath+'adapt_data.npz'
elif split == "adapt_test":
  filename = savepath+'adapt_test_data.npz'
elif split == "adapt_huge":
  filename = savepath+'adapt_huge_data.npz'

np.savez(filename, t=t_eval, X=data)




# %%

## Plot the data for sanity checks
if _in_ipython_session:
  a = .1
  y0 = [0, 2]
  # y0 = [1, 1]
  t = np.linspace(*t_span, 1000)

  for b in np.linspace(0., 1.2, 15)[:]:
      t_span = [0, 1000]
      sol = solve_ivp(selkov, t_span, y0, args=(a, b), dense_output=True)
      y = sol.sol(t)
      plt.plot(y[0], y[1], label=f'b={b:.2f}')

  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()
  plt.show()
