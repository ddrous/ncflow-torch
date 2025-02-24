#%%

from ncf_torch import *

T0_runs = ["20250221-122109-T0", "20250221-182241-T0", "20250222-061306-T0", "20250222-064141-T0", "20250222-071257-T0", "20250222-074424-T0", "20250223-221317-T0"]
T1_runs = ["20250221-125251-T1", "20250222-081237-T1", "20250222-154020-T1", "20250223-001532-T1", "20250223-095720-T1", "20250223-160116-T1"]

## Load the losses from all runs
T0_losses = []
T1_losses = []
T0_val_losses = []
T1_val_losses = []
for run in T0_runs:
    losses = np.load(f"../runs/{run}/val_losses.npy")
    T0_val_losses.append(losses[:, 1])

    losses = np.load(f"../runs/{run}/train_histories.npz")["losses_node"]
    T0_losses.append(losses)

for run in T1_runs:
    losses = np.load(f"../runs/{run}/val_losses.npy")
    T1_val_losses.append(losses[:, 1])

    losses = np.load(f"../runs/{run}/train_histories.npz")["losses_node"]
    T1_losses.append(losses)


## Plot the mean losses, with shaded standard deviation
T0_losses = np.concatenate(T0_losses, axis=1).T
T1_losses = np.concatenate(T1_losses, axis=1).T

T0_mean = np.mean(T0_losses, axis=0)
T0_std = np.std(T0_losses, axis=0)

T1_mean = np.mean(T1_losses, axis=0)
T1_std = np.std(T1_losses, axis=0)

plt.figure(figsize=(10, 4))
plt.plot(T0_mean, label="Neural ODE")
plt.fill_between(range(len(T0_mean)), T0_mean - T0_std, T0_mean + T0_std, alpha=0.2)

plt.plot(T1_mean, label="Neural Context Flow")
plt.fill_between(range(len(T1_mean)), T1_mean - T1_std, T1_mean + T1_std, alpha=0.2)

plt.xlabel("Epochs")
plt.title("Train Losses")
plt.yscale("log")
plt.ylim(4e-4, 5)
plt.legend()
plt.draw()

plt.savefig("train_loss.png", transparent=False, dpi=300)


#%%

epochs_val = np.load(f"../runs/{T0_runs[0]}/val_losses.npy")[:, 0]

T0_val_losses = np.array(T0_val_losses)
T1_val_losses = np.array(T1_val_losses)

print(T0_val_losses.shape)
print(T1_val_losses.shape)

T0_mean = np.mean(T0_val_losses, axis=0)
T0_std = np.std(T0_val_losses, axis=0)

T1_mean = np.mean(T1_val_losses, axis=0)
T1_std = np.std(T1_val_losses, axis=0)

plt.figure(figsize=(10, 4))
plt.plot(epochs_val, T0_mean, label="Neural ODE")
plt.fill_between(epochs_val, T0_mean - T0_std, T0_mean + T0_std, alpha=0.2)

plt.plot(epochs_val, T1_mean, label="Neural Context Flow")
plt.fill_between(epochs_val, T1_mean - T1_std, T1_mean + T1_std, alpha=0.2)

plt.xlabel("Epochs")
plt.title("Validation Losses")
plt.yscale("log")
plt.ylim(4e-4, 5)
plt.legend()
plt.draw()

plt.savefig("val_loss.png", transparent=False, dpi=300)


#%%

## Collect the adaptation losses
T0_adapt_losses = []
T1_adapt_losses = []
skip_run = 5
for i, run in enumerate(T0_runs):
    if i == skip_run:
        continue
    losses = np.load(f"../runs/{run}/adapt/adapt_histories_.npz")["losses_adapt"]
    T0_adapt_losses.append(losses)

for run in T1_runs:
    losses = np.load(f"../runs/{run}/adapt/adapt_histories_.npz")["losses_adapt"]
    T1_adapt_losses.append(losses)

T0_adapt_losses = np.concatenate(T0_adapt_losses, axis=1).T
T1_adapt_losses = np.concatenate(T1_adapt_losses, axis=1).T

## Print the row and columnot T0 with the maximum value
print(np.argmax(T0_adapt_losses))
print(np.argmax(T0_adapt_losses) // T0_adapt_losses.shape[1], np.argmax(T0_adapt_losses) % T0_adapt_losses.shape[1])

print(T0_adapt_losses.shape)
print(T1_adapt_losses.shape)

T0_mean = np.mean(T0_adapt_losses, axis=0)
T0_std = np.std(T0_adapt_losses, axis=0)

T1_mean = np.mean(T1_adapt_losses, axis=0)
T1_std = np.std(T1_adapt_losses, axis=0)
factor = 0.4

plt.figure(figsize=(10, 4))
plt.plot(T0_mean, label="Neural ODE")
plt.fill_between(range(len(T0_mean)), T0_mean - factor*T0_std, T0_mean + factor*T0_std, alpha=0.2)

plt.plot(T1_mean, label="Neural Context Flow")
plt.fill_between(range(len(T1_mean)), T1_mean - factor*T1_std, T1_mean + factor*T1_std, alpha=0.2)

plt.xlabel("Epochs")
plt.title("Adaptation Losses")
plt.yscale("log")
# plt.ylim(4e-4, 5)
plt.legend()
plt.draw()

plt.savefig("adapt_loss.png", transparent=False, dpi=300)




#%%


## Reset the matplotlib style to default
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style="")
import numpy as np

np.random.seed(19680801)
fruit_weights = [
    np.random.normal(130, 10, size=100),
    np.random.normal(125, 20, size=100),
    np.random.normal(120, 30, size=100),
]
labels = ['peaches', 'oranges', 'tomatoes']
colors = ['peachpuff', 'orange', 'tomato']

fig, ax = plt.subplots()
ax.set_ylabel('fruit weight (g)')

bplot = ax.boxplot(fruit_weights,
                   patch_artist=True,
                     labels=labels)  # will be used to label x-ticks

# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.draw()



#%%
## Collect he per-environment OOD losses
T0_ood_losses = []
T1_ood_losses = []
skip_run = 60

# Parse the nohup file and look for the line : ## Per-Environment OOD score: [0.03593841 0.02264551 0.00381217 0.01129208]

for i, run in enumerate(T0_runs):
    if i == skip_run:
        continue
    with open(f"../runs/{run}/nohup.log", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Per-Environment OOD score" in line:
                ood_score = np.array([float(x) for x in line.split("[")[1].split("]")[0].split()])
                T0_ood_losses.append(ood_score)

for run in T1_runs:
    with open(f"../runs/{run}/nohup.log", "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Per-Environment OOD score" in line:
                ood_score = np.array([float(x) for x in line.split("[")[1].split("]")[0].split()])
                T1_ood_losses.append(ood_score)

print(len(T0_ood_losses))
T0_ood_losses = np.array(T0_ood_losses)
T1_ood_losses = np.array(T1_ood_losses)

print(T0_ood_losses)
print(T1_ood_losses.shape)

labels = ["Env 1", "Env 2", "Env 3", "Env 4"]
# Make 4 differnt boxplots, for both Neural ODE (T0) and Neural Context Flow (T1)
plt.figure(figsize=(10, 4))
plt.boxplot(T0_ood_losses, positions=np.arange(1, 5), widths=0.3, patch_artist=True, labels=labels)

## Adjust the fill color of the boxes
plt.boxplot(T1_ood_losses, positions=np.arange(1.5, 5.5), widths=0.3, patch_artist=True, labels=labels, boxprops=dict(facecolor="orange"))

plt.xticks([1.25, 2.25, 3.25, 4.25], ["Env 1", "Env 2", "Env 3", "Env 4"], fontsize=18)
# plt.xlabel("Environments")
# plt.ylabel("OOD score")

plt.yscale("log")

plt.title("Per-Environment OOD MSEs")
plt.draw()

plt.savefig("ood_mse.png", transparent=False, dpi=300)
