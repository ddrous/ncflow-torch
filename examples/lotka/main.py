#%%
# %load_ext autoreload
# %autoreload 2

## Import all the necessary libraries
from ncf_torch import *
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# from torch.autograd.functional import jvp

## Set the torch seed for reproducibility
seed = 35
torch.manual_seed(seed)
np.random.seed(seed)

## NCF main hyperparameters ##
context_size = 2                 ## Size of the context vector
taylor_order = 0
context_pool_size = 1 if taylor_order==0 else 3               ## Number of neighboring contexts j to use for a flow in env e

## General training hyperparameters ##
print_error_every = 10              ## Print the error every n epochs
ivp_args = {"rtol":1e-3, "atol":1e-6, "method":"rk4"} ## Arguments for the integrator
learning_rates = (1e-3, 1e-3)      ## Learning rates for the weights and the contexts
nb_epochs = 2000                       ## Number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_folder = None                   ## Folder to save the results of the run
train = True                       ## Train the model, or load a pre-trained model

## Adaptation hyperparameters ##
adapt_test = True                   ## Test the model on an adaptation datase t
adapt_restore = False               ## Restore a trained adaptation model

nb_epochs_adapt = 2000              ## Number of epochs to adapt


#%%

if train == True:

    ## Create a folder for the runs
    if not os.path.exists('./runs'):
        os.mkdir('./runs')

    # Make a new folder inside for the current run
    run_folder = './runs/'+time.strftime("%Y%m%d-%H%M%S")+'/'
    os.mkdir(run_folder)
    # run_folder = './runs/19022025-103059-Test/'
    print("Run folder created successfuly:", run_folder)

    # Make the checkpoint folder
    os.mkdir(run_folder+"checkpoints")

    # Save this main and the dataset gneration scripts in that folder
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {run_folder}")

    # Save the nodax module files as well
    os.system(f"cp -r ../../ncf_torch {run_folder}")
    print("Completed copied scripts ")

    data_folder = "./data/"

else:
    run_folder = "./"
    data_folder = "../../data/"
    print("No training. Loading data and results from:", run_folder)

## Create a folder for the adaptation results
adapt_folder = run_folder+"adapt/"
if not os.path.exists(adapt_folder):
    os.mkdir(adapt_folder)

#%%

## Define dataloader for training
train_dataloader = DataLoader(data_folder+"train.npz", batch_size=-1, shuffle=True, device=device)

## Useful information about the data - As per the Gen-Dynamics interface
nb_envs = train_dataloader.nb_envs
nb_trajs_per_env = train_dataloader.nb_trajs_per_env
nb_steps_per_traj = train_dataloader.nb_steps_per_traj
data_size = train_dataloader.data_size

print('Training dataser properties: \n - Number of environments:', nb_envs, "\n - Number of trajectories per environment:", nb_trajs_per_env, "\n - Number of steps per trajectory:", nb_steps_per_traj, "\n - Data size:", data_size)

## Define dataloader for validation (for selecting the best model during training)
val_dataloader = DataLoader(data_folder+"test.npz", batch_size=-1, shuffle=False, device=device)


#%%

class NeuralNet(nn.Module):
    """ Nueral Network for the neural ODE's vector field """
    layers_data: list
    layers_context: list
    layers_shared: list

    def __init__(self, data_size, int_size, context_size):
        super(NeuralNet, self).__init__()
        self.layers_context = nn.ModuleList()
        # mid_size = (context_size + int_size)//2 if context_size<=2 else context_size//4
        mid_size = int_size
        self.layers_context.append(nn.Linear(context_size, mid_size))
        self.layers_context.append(nn.Linear(mid_size, int_size))
        self.layers_context.append(nn.Linear(int_size, int_size))

        self.layers_data = nn.ModuleList()
        self.layers_data.append(nn.Linear(data_size, int_size))
        self.layers_data.append(nn.Linear(int_size, int_size))
        self.layers_data.append(nn.Linear(int_size, int_size))

        self.layers_shared = nn.ModuleList()
        self.layers_shared.append(nn.Linear(2*int_size, int_size))
        self.layers_shared.append(nn.Linear(int_size, int_size))
        self.layers_shared.append(nn.Linear(int_size, int_size))
        self.layers_shared.append(nn.Linear(int_size, data_size))

    def toggle_grad(self, true_or_false):
        for layer in self.layers_shared + self.layers_data + self.layers_context:
            for param in layer.parameters():
                param.requires_grad = true_or_false

    def forward(self, t, y, ctx):
        # print('==>',self.layers_context[0].weight.requires_grad)
        for i, layer in enumerate(self.layers_context):
            ctx = layer(ctx)
            if i != len(self.layers_context)-1:
                ctx = F.softplus(ctx)

        for i, layer in enumerate(self.layers_data):
            y = layer(y)
            if i != len(self.layers_data)-1:
                y = F.softplus(y)

        ctx = torch.unsqueeze(ctx, 0)
        ctx = torch.broadcast_to(ctx, y.shape)

        y = torch.concatenate([y, ctx], dim=1)
        for i, layer in enumerate(self.layers_shared):
            y = layer(y)
            if i != len(self.layers_shared)-1:
                y = F.softplus(y)

        return y

### Define the Taylor expantion about the context vector ###
class SelfModulatedVectorField(nn.Module):
    neuralnet: nn.Module
    taylor_order: int

    def __init__(self, neuralnet, taylor_order=1):
        super(SelfModulatedVectorField, self).__init__()
        self.neuralnet = neuralnet
        self.taylor_order = taylor_order

    # @torch.compile
    def __call__(self, t, x, ctxs):
        ctx, ctx_ = ctxs

        ##Print whether the neuralnet requires grad or not
        # print('==>',self.neuralnet.layers_context[0].weight.requires_grad)

        vf = lambda xi: self.neuralnet(t, x, xi)
        if self.taylor_order == 0:
            return vf(ctx)

        else:
            gradvf = lambda xi_: jvp(vf, (xi_,), (ctx-xi_,))[1]
            # gradvf = torch.compile(lambda xi_: jvp(vf, (xi_,), (ctx-xi_,))[1])

            if self.taylor_order == 1:
                return vf(ctx_) + 1.0*gradvf(ctx_)
                # return vf(ctx_) + 1.0*jvp(vf, (ctx_,), (ctx-ctx_,))[1]

            elif self.taylor_order == 2:
                scd_order_term = jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
                return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term

            else:
                raise ValueError("Only taylor orders 0, 1, 2 are currently supported.")

## Create the neural network (accounting for the unknown in the system), and use physics is problem is known
neuralnet = NeuralNet(data_size=2, int_size=64, context_size=context_size)
vectorfield = SelfModulatedVectorField(neuralnet, taylor_order)
## Define the context parameters for all environwemts in a single module
contexts = ContextParams(nb_envs, context_size)

print("\nTotal number of shared parameters in the neural ODE model:", sum(p.numel() for p in neuralnet.parameters()))
print("Total number of environment-specific parameters in the contexts:", sum(p.numel() for p in contexts.parameters()), "\n")

## Define a custom loss function
# @torch.compile
def loss_fn_env(model, trajs, t_eval, ctx, all_ctx_s):

    ## Define the context pool using the Random-All strategy
    # ind = torch.random.permutation(all_ctx_s.shape[0])[:context_pool_size]

    ## Define the context pool using the nearest-first strategy
    _, ind = torch.sort(torch.sum(torch.abs(all_ctx_s-ctx), dim=1), dim=0)
    ind = ind[:context_pool_size]
    ctx_s = all_ctx_s[ind, :]

    trajs_hat = []
    nb_steps = []
    for j in range(context_pool_size):
        traj_hat, nb_step = model(trajs[:, 0, :], t_eval, ctx, ctx_s[j:j+1, :])
        trajs_hat.append(traj_hat)
        nb_steps.append(nb_step)

    trajs_hat = torch.stack(trajs_hat)
    nb_steps_mean = torch.sum(torch.Tensor(nb_steps))/ctx_s.shape[0]

    # term1 = torch.mean((trajs[None,...]-trajs_hat)**2)      ## reconstruction

    trajs = torch.broadcast_to(trajs[None,...], trajs_hat.shape)
    term1 = F.mse_loss(trajs_hat, trajs)      ## reconstruction

    term2 = torch.mean(torch.abs(ctx))                      ## context regularisation
    # term3 = params_norm_squared(model)                      ## weight regularisation

    # loss_val = term1 + 1e-3*term2 + 1e-3*term3
    # loss_val = term1 + 1e-3*term2
    loss_val = term1

    return loss_val, (nb_steps_mean, term1, term2)   ## The neural ODE integrator returns the number of steps taken, which are handy for analysis

## Send everything tot he gpu
vectorfield.to(device)
contexts.to(device)

## Finnaly, create the learner
learner = Learner(vectorfield, contexts, loss_fn_env, ivp_args)


###%%
## Create the trainer
trainer = Trainer(train_dataloader, learner, learning_rates)

#%%

if train == True:
    ## Ordinary alternating minimsation to train the NCF model
    trainer.train(nb_epochs=nb_epochs, 
                print_error_every=print_error_every, 
                update_context_every=1, 
                save_path=run_folder, 
                val_dataloader=val_dataloader)
else:
    restore_folder = run_folder
    trainer.restore_trainer(path=restore_folder)



#%%

## Test and visualise the results on a test dataloader (same as the validation dataset)
visualtester = VisualTester(trainer)

ind_crit, ind_crit_all = visualtester.test(val_dataloader)
visualtester.visualize(val_dataloader, save_path=run_folder+"results_in_domain.png");
print("## Per-Environment InD score:", ind_crit_all)


#%%

## Adaptation of the model to a new dataset
if adapt_test:
    adapt_dataloader = DataLoader(data_folder+"ood_train.npz", adaptation=True, device=device)
    adapt_dataloader_test = DataLoader(data_folder+"ood_test.npz", adaptation=True, device=device)

    if adapt_restore == False:
        trainer.adapt(adapt_dataloader, 
                        nb_epochs=nb_epochs_adapt, 
                        print_error_every=print_error_every, 
                        save_path=adapt_folder)
    else:
        print("Save_id for restoring trained adapation model:", adapt_dataloader.data_id)
        trainer.restore_adapted_trainer(path=adapt_folder, data_loader=adapt_dataloader)

    ## Evaluate the model on the adaptation test dataset
    ood_crit, ood_crit_all = visualtester.test(adapt_dataloader_test)
    print("## Per-Environment OOD score:", ood_crit_all, flush=True)

    visualtester.visualize(adapt_dataloader_test, save_path=adapt_folder+"results_ood.png");


#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")
