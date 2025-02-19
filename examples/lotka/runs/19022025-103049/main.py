### THIS IS THE MAIN SCRIPT TO TRAIN EITHER A NCF-T1 or -T2 ###

## Import all the necessary libraries
from ncf_torch import *

## Set the seed for reproducibility
seed = 2026

## NCF main hyperparameters ##
context_pool_size = 6               ## Number of neighboring contexts j to use for a flow in env e
context_size = 1024                 ## Size of the context vector

nb_outer_steps_max = 2           ## maximum number of outer steps when using NCF-T2
nb_inner_steps_max = 2             ## Maximum number of inner steps when using NCF-T2 (for both weights and contexts)
proximal_beta = 1e2                 ## Proximal coefficient, see beta in https://proceedings.mlr.press/v97/li19n.html
inner_tol_node = 1e-9               ## Tolerance for the inner optimisation on the weights
inner_tol_ctx = 1e-8                ## Tolerance for the inner optimisation on the contexts
early_stopping_patience = nb_outer_steps_max//1       ## Number of outer steps to wait before stopping early

## General training hyperparameters ##
print_error_every = 10              ## Print the error every n epochs
integrator = "Dopri5"         ## Integrator to use for the learner
ivp_args = {"dt_init":1e-4, "rtol":1e-3, "atol":1e-6, "max_steps":40000}

run_folder = None                   ## Folder to save the results of the run
train = True                       ## Train the model, or load a pre-trained model
save_trainer = True                 ## Save the trainer object after training
finetune = False                    ## Finetune a trained model

## Adaptation hyperparameters ##
adapt_test = True                   ## Test the model on an adaptation dataset
adapt_restore = False               ## Restore a trained adaptation model

nb_epochs_adapt = 1500              ## Number of epochs to adapt


#%%


if train == True:

    ## Create a folder for the runs
    if not os.path.exists('./runs'):
        os.mkdir('./runs')

    # Make a new folder inside for the current run
    run_folder = './runs/'+time.strftime("%d%m%Y-%H%M%S")+'/'
    os.mkdir(run_folder)
    print("Run folder created successfuly:", run_folder)

    # Save this main and the dataset gneration scripts in that folder
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {run_folder}")
    os.system(f"cp dataset.py {run_folder}")

    # Save the nodax module files as well
    os.system(f"cp -r ../../nodax {run_folder}")
    print("Completed copied scripts ")

else:
    print("No training. Loading data and results from:", run_folder)

## Create a folder for the adaptation results
adapt_folder = run_folder+"adapt/"
if not os.path.exists(adapt_folder):
    os.mkdir(adapt_folder)

#%%

## Define dataloader for training
train_dataloader = DataLoader(run_folder+"train.npz", batch_size=-1, shuffle=True)

## Useful information about the data - As per the Gen-Dynamics interface
nb_envs = train_dataloader.nb_envs
nb_trajs_per_env = train_dataloader.nb_trajs_per_env
nb_steps_per_traj = train_dataloader.nb_steps_per_traj
data_size = train_dataloader.data_size

## Define dataloader for validation (for selecting the best model during training)
val_dataloader = DataLoader(run_folder+"test.npz", shuffle=False)

#%%

## Define model and loss function for the learner
class Swish(nn.Module):
    """Swish activation function"""
    beta: torch.Tensor
    def __init__(self):
        beta = np.random.uniform(key, shape=(1,), minval=0.01, maxval=1.0)
        self.beta = torch.Tensor(beta)
    def __call__(self, x):
        return x * nn.sigmoid(self.beta * x)


class NeuralNet(nn.Module):
    """ Nueral Network for the neural ODE's vector field """
    layers_data: list
    layers_context: list
    layers_shared: list

    def __init__(self, data_size, int_size, context_size):
        super(NeuralNet, self).__init__()
        self.layers_context = nn.ModuleList()
        self.layers_context.append(nn.Linear(context_size, context_size//4))
        self.layers_context.append(nn.Linear(context_size//4, int_size))
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


    def __call__(self, t, y, ctx):

        for i, layer in enumerate(self.layers_context):
            ctx = layer(ctx)
            if i != len(self.layers_context)-1:
                ctx = F.softplus(ctx)

        for i, layer in enumerate(self.layers_data):
            y = layer(y)
            if i != len(self.layers_data)-1:
                y = F.softplus(y)

        y = torch.concatenate([y, ctx], dim=0)
        for i, layer in enumerate(self.layers_shared):
            y = layer(y)
            if i != len(self.layers_shared)-1:
                y = F.softplus(y)

        return y


### Define the Taylor expantion about the context vector ###
class ContextFlowVectorField(nn.Module):
    neuralnet: nn.Module

    def __init__(self, neuralnet):
        self.neuralnet = neuralnet

    def __call__(self, t, x, ctxs):
        ctx, ctx_ = ctxs

        vf = lambda xi: self.augmentation(t, x, xi)

        gradvf = lambda xi_: jvp(vf, (xi_,), (ctx-xi_,))[1]
        scd_order_term = jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]

        # return vf(ctx_) + 1.0*gradvf(ctx_)                            ## If NCF-T1
        return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term         ## If NCF-T2

## Create the neural network (accounting for the unknown in the system), and use physics is problem is known
neuralnet = NeuralNet(data_size=2, int_size=64, context_size=context_size, key=seed)
vectorfield = ContextFlowVectorField(neuralnet, physics=None)

print("\n\nTotal number of parameters in the model:", sum(p.numel() for p in neuralnet.parameters()), "\n\n")

## Define the context parameters for all environwemts in a single module
contexts = ContextParams(nb_envs, context_size, key=None)

## Define a custom loss function
def loss_fn_ctx(model, trajs, t_eval, ctx, all_ctx_s):

    ## Define the context pool using the Random-All strategy
    ind = torch.random.permutation(all_ctx_s.shape[0])[:context_pool_size]
    ctx_s = all_ctx_s[ind, :]

    trajs_hat, nb_steps = vmap(model, in_dims=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_s)
    new_trajs = torch.broadcast_to(trajs, trajs_hat.shape)

    term1 = torch.mean((new_trajs-trajs_hat)**2)  ## reconstruction
    term2 = toch.mean(torch.abs(ctx))              ## context regularisation
    term3 = params_norm_squared(model)          ## weight regularisation

    loss_val = term1 + 1e-3*term2 + 1e-3*term3

    return loss_val, (torch.sum(nb_steps)/ctx_s.shape[0], term1, term2)   ## The neural ODE integrator returns the number of steps taken, which are handy for analysis

## Finnaly, create the learner
learner = Learner(vectorfield, contexts, loss_fn_ctx, integrator, ivp_args, key=seed)


#%%
## Create the trainer
trainer = Trainer(train_dataloader, learner)

#%%

if train == True:
    ## Ordinary alternating minimsation to train the NCF model
    trainer.train(nb_epochs=nb_outer_steps_max*nb_inner_steps_max, 
                print_error_every=print_error_every, 
                update_context_every=1, 
                save_path=run_folder if save_trainer == True else False, 
                key=seed, 
                val_dataloader=val_dataloader, 
                int_prop=1.0)

    # ## Proximal alternating minimisation to train the NCF-t2 model
    # trainer.train_proximal(nb_outer_steps_max=nb_outer_steps_max, 
    #                         nb_inner_steps_max=nb_inner_steps_max, 
    #                         proximal_reg=proximal_beta, 
    #                         inner_tol_node=inner_tol_node, 
    #                         inner_tol_ctx=inner_tol_ctx,
    #                         print_error_every=print_error_every*(2**0), 
    #                         save_path=trainer_save_path, 
    #                         val_dataloader=val_dataloader, 
    #                         patience=early_stopping_patience,
    #                         int_prop=1.0,
    #                         key=seed)
else:
    restore_folder = run_folder
    trainer.restore_trainer(path=restore_folder)



#%%

## Test and visualise the results on a test dataloader (same as the validation dataset)
test_dataloader = DataLoader(run_folder+"test.npz", shuffle=False)
visualtester = VisualTester(trainer)

ind_crit = visualtester.test(test_dataloader)
visualtester.visualize(test_dataloader, save_path=run_folder+"results_in_domain.png");


#%%

## Adaptation of the model to a new dataset
if adapt_test:
    adapt_dataloader = DataLoader(adapt_folder+"ood_train.npz", adaptation=True)           ## TRAIN
    adapt_dataloader_test = DataLoader(adapt_folder+"ood_test.npz", adaptation=True) ## TEST

    if adapt_restore == False:
        trainer.adapt_sequential(adapt_dataloader, nb_epochs=nb_epochs_adapt, print_error_every=print_error_every, save_path=adapt_folder)
    else:
        print("Save_id for restoring trained adapation model:", adapt_dataloader.data_id)
        trainer.restore_adapted_trainer(path=adapt_folder, data_loader=adapt_dataloader)

    ## Evaluate the model on the adaptation test dataset
    ood_crit, _ = visualtester.test(adapt_dataloader_test)

    visualtester.visualize(adapt_dataloader_test, save_path=adapt_folder+"results_ood.png");
