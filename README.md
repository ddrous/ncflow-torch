
# Neural Context Flow

PyTorch implementation of [Neural Context Flows for Meta-Learning of Dynamical Systems](https://arxiv.org/abs/2405.02154), accepted at ICLR 2025.
> [!WARNING]  
> This implementation is not optimal ! We observed huge slowdowns when performing contextual self-modulation, mostly related to the following:
    - forward-mode Jacobian-Vector Product (JVP) primitive to facilitate the Taylor expansions
    - functions transformations `torch.vmap` and `torch.compile` proving too restrictive
    We are grateful for any pull-request addressing these issues. In the meantime, we recommend the optimized JAX implementation available [at this link](https://github.com/ddrous/ncflow).

<p align="center">
<img src="docs/assets/hydra.png" alt="drawing" width="350"/>
</p>

## Overview
Neural Context Flow (NCF) is a framework for learning dynamical systems that can adapt to different environments/contexts, making it particularly valuable for scientific machine learning applications where the underlying system dynamics may vary across physical parameter values.

NCF is powered by the __contextual self-modulation__ regularization mechanism. This inductive bias performs Taylor expansion of the vector field about the context vectors, resulting in several candidate trajectories. While the resulting training loss might be higher than the naive Neural ODE, we observe lower losses at adaptation time.

<p align="center">
<img src="docs/assets/losses.png" alt="drawing" width="800"/>
</p>

The NCF package is built around 4 extensible modules: 
- a __DataLoader__: to load the dynamics datasets
- a __Learner__: a model, a context and the loss function
- A __Trainer__: the training and adaptation algorithms
- a __VisualTester__: to test and visualize the results

## Getting Started
To run an experiment, follow the steps below:
1. Install the package: `pip install -e .`
2. Navigate to the problem of interest in the `examples` folder
3. Download the data from [Gen-Dynamics](https://github.com/ddrous/gen-dynamics) and place it in the `data` folder
4. Set its hyperparameters, and run the `main.py` script to both train and adapt the model to various environments. One can either run it in Notebook or script mode. We recommend using `nohup` to log the results: 
    > ```nohup python main.py > nohup.log &```
5. Once trained, move to the corresponding run folder saved in `runs`. Toggle the `train` flag in the `main.py` to `False`. Rerun `main.py` to perform additional experiments such as uncertainty estimation, interpretability, etc. 

## Notes
The same `main.py` in `examples/lotka` can be used (relatively) unchanged for other problems.

## ToDos:
- [ ] Fix the slow self-modulation issues
- [ ] Test the installation in neutral conda environments

If you use this work, please cite the corresponding paper:
```
@inproceedings{
    nzoyem2025neural,
    title={Neural Context Flows for Meta-Learning of Dynamical Systems},
    author={Roussel Desmond Nzoyem and David A.W. Barton and Tom Deakin},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=8vzMLo8LDN}
}
```
