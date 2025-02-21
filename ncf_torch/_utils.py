import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jvp
from torch import vmap
import torch.nn.functional as F

from ._config import *

import numpy as np
np.set_printoptions(suppress=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='notebook', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
mpl.rcParams['savefig.facecolor'] = 'w'

from torchdiffeq import odeint
from functools import partial

import os
import time

def seconds_to_hours(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds

## Wrapper function for matplotlib and seaborn
def sbplot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, **kwargs):
    if ax==None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    # sns.despine(ax=ax)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.plot(*args, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if "label" in kwargs.keys():
        ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    return ax

## Alias for sbplot
def plot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, **kwargs):
  return sbplot(*args, ax=ax, figsize=figsize, x_label=x_label, y_label=y_scale, title=title, x_scale=x_scale, y_scale=y_scale, xlim=xlim, ylim=ylim, **kwargs)


def get_id_current_time():
    """ Returns a string of the current time in the format as an ID """
    # return time.strftime("%Y%m%d-%H%M%S")
    return time.strftime("%H%M%S")


def flatten_params(parameters):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**

    """
    l = [torch.flatten(p) for p in parameters]
    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s+size))
        s += size
    flat = torch.cat(l).view(-1, 1)
    return {"params": flat, "indices": indices}


def recover_flattened(flat_params, indices, model):
    """
    Gives a list of recovered parameters from their flattened form
    :param flat_params: [#params, 1]
    :param indices: a list detaling the start and end index of each param [(start, end) for param]
    :param model: the model that gives the params with correct shapes
    :return: the params, reshaped to the ones in the model, with the same order as those in the model
    """
    l = [flat_params[s:e] for (s, e) in indices]
    for i, p in enumerate(model.parameters()):
        l[i] = l[i].view(*p.shape)
    return l

def params_diff_norm(params1, params2):
    """ norm of the parameters difference"""
    diff = [torch.norm(p1-p2) for p1, p2 in zip(params1, params2)]
    return diff

def params_diff_norm_squared(params1, params2):
    """ normalised squared norm of the parameters difference """
    diff = [torch.norm(p1-p2)**2 for p1, p2 in zip(params1, params2)]
    return sum(diff) / len(diff)

def params_norm_squared(params):
    """ normalised squared norm of the parameter """
    norm = [torch.norm(p)**2 for p in params.parameters()]
    return sum(norm) / len(norm)
