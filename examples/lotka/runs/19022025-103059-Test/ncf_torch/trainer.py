import pickle

from ncf_torch.dataloader import DataLoader
from ncf_torch.learner import ContextParams
from ncf_torch.visualtester import VisualTester
from ._utils import *

class Trainer:
    def __init__(self, dataloader, learner, optimiser_lrs):
        self.dataloader = dataloader
        self.learner = learner

        self.learning_rates = optimiser_lrs
        lr_node, lr_ctx = optimiser_lrs
        self.opt_node = optim.Adam(self.learner.neuralode.parameters(), lr=lr_node)
        self.opt_ctx = optim.Adam(self.learner.contexts.parameters(), lr=lr_ctx)

        self.losses_node = []
        self.losses_ctx = []
        self.nb_steps_node = []
        self.nb_steps_ctx = []

        self.val_losses = []


    def train(self, 
              nb_epochs, 
              int_prop=1.0, 
              update_context_every=1, 
              print_error_every=100, 
              save_path=False, 
              val_dataloader=None, 
              val_criterion=None):

        loss_fn = self.learner.loss_fn
        node = self.learner.neuralode
        contexts = self.learner.contexts

        nb_train_steps_per_epoch = int(np.ceil(self.dataloader.nb_trajs_per_env / self.dataloader.batch_size))
        total_steps = nb_epochs * nb_train_steps_per_epoch

        assert update_context_every <= nb_train_steps_per_epoch, "Update_context_every must be smaller than nb_train_steps_per_epoch"

        assert int_prop>0 and int_prop<=1.0, "The proportion of trajectory length to consider for training must be between 0 and 1"
        self.dataloader.int_cutoff = int(int_prop*self.dataloader.nb_steps_per_traj)

        if val_dataloader is not None:
            tester = VisualTester(self)

        print(f"\n\n=== Beginning training ... ===")
        print(f"    Number of examples in a batch: {self.dataloader.batch_size}")
        print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
        print(f"    Number of training epochs: {nb_epochs}")
        print(f"    Total number of training steps: {total_steps}")

        start_time = time.time()

        losses = []
        nb_steps = []

        val_losses = []

        for epoch in range(nb_epochs):
            nb_batches = 0
            loss_sum = torch.zeros(1)
            nb_steps_sum = 0

            for i, batch in enumerate(self.dataloader):

                self.opt_node.zero_grad()
                self.opt_ctx.zero_grad()

                loss, (nb_steps_, term1, term2) = loss_fn(node, contexts, batch)
                loss.backward()

                self.opt_node.step()
                self.opt_ctx.step()

                loss_sum += torch.Tensor([loss])
                nb_steps_sum += nb_steps_

                nb_batches += 1

            loss_epoch = loss_sum/nb_batches
            nb_steps_eph = nb_steps_sum

            losses.append(loss_epoch)
            nb_steps.append(nb_steps_eph)

            if epoch%print_error_every==0 or epoch<=3 or epoch==nb_epochs-1:

                if val_dataloader is not None:
                    self.learner.neuralode = node
                    self.learner.contexts = contexts
                    ind_crit,_ = tester.test(val_dataloader, int_cutoff=1.0, criterion=val_criterion, verbose=False)
                    val_losses.append(np.array([epoch, ind_crit]))
                    print(f"    Epoch: {epoch:-5d}      LossTrajs: {loss_epoch[0]:-.8f}     ContextsNorm: {torch.mean(term2):-.8f}     ValIndCrit: {ind_crit:-.8f}", flush=True)
                else:
                    print(f"    Epoch: {epoch:-5d}      LossTrajs: {loss_epoch[0]:-.8f}     ContextsNorm: {torch.mean(term2):-.8f}", flush=True)

        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_node.append(torch.vstack(losses))
        self.losses_ctx.append(torch.vstack(losses))        ## Same things
        self.nb_steps_node.append(np.array(nb_steps))
        self.nb_steps_ctx.append(np.array(nb_steps))     ## Same things

        if val_dataloader is not None:
            self.val_losses.append(np.vstack(val_losses))

        self.learner.neuralode = node
        self.learner.contexts = contexts

        if save_path:
            self.save_trainer(save_path)











    # def train_ordinary(self, nb_epochs, int_prop=1.0, update_context_every=1, print_error_every=100, save_path=False, val_dataloader=None, val_criterion=None):

    #     loss_fn = self.learner.loss_fn
    #     node = self.learner.neuralode
    #     contexts = self.learner.contexts

    #     def train_step_node(node, contexts, batch, weights):

    #         for param in contexts.parameters():
    #             param.requires_grad = False

    #         self.opt_node.zero_grad()
    #         loss, aux_data = loss_fn(node, contexts, batch, weights)
    #         loss.backward()
    #         self.opt_node.step()

    #         for param in contexts.parameters():
    #             param.requires_grad = True

    #         return node, contexts, loss, aux_data

    #     def train_step_ctx(node, contexts, batch, weights):

    #         for param in node.parameters():
    #             param.requires_grad = False

    #         self.opt_ctx.zero_grad()
    #         loss, aux_data = loss_fn(node, contexts, batch, weights)
    #         loss.backward()
    #         self.opt_ctx.step()

    #         for param in node.parameters():
    #             param.requires_grad = True

    #         return node, contexts, loss, aux_data

    #     nb_train_steps_per_epoch = int(np.ceil(self.dataloader.nb_trajs_per_env / self.dataloader.batch_size))
    #     total_steps = nb_epochs * nb_train_steps_per_epoch

    #     assert update_context_every <= nb_train_steps_per_epoch, "Update_context_every must be smaller than nb_train_steps_per_epoch"

    #     assert int_prop>0 and int_prop<=1.0, "The proportion of trajectory length to consider for training must be between 0 and 1"
    #     self.dataloader.int_cutoff = int(int_prop*self.dataloader.nb_steps_per_traj)

    #     if val_dataloader is not None:
    #         tester = VisualTester(self)


    #     print(f"\n\n=== Beginning training ... ===")
    #     print(f"    Number of examples in a batch: {self.dataloader.batch_size}")
    #     print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
    #     print(f"    Number of training epochs: {nb_epochs}")
    #     print(f"    Total number of training steps: {total_steps}")

    #     start_time = time.time()

    #     losses_node = []
    #     losses_ctx = []
    #     nb_steps_node = []
    #     nb_steps_ctx = []

    #     val_losses = []

    #     weightings = torch.ones(self.learner.nb_envs) / self.learner.nb_envs

    #     for epoch in range(nb_epochs):
    #         nb_batches_node = 0
    #         nb_batches_ctx = 0
    #         loss_sum_node = torch.zeros(1)
    #         loss_sum_ctx = torch.zeros(1)
    #         nb_steps_eph_node = 0
    #         nb_steps_eph_ctx = 0

    #         for i, batch in enumerate(self.dataloader):

    #             node, contexts, loss_node, (nb_steps_node_, term1, term2) = train_step_node(node, contexts, batch, weightings)

    #             loss_sum_node += torch.Tensor([loss_node])
    #             nb_steps_eph_node += nb_steps_node_

    #             nb_batches_node += 1

    #             if i%update_context_every==0:
    #                 node, contexts, loss_ctx, (nb_steps_ctx_, term1, term2) = train_step_ctx(node, contexts, batch, weightings)

    #                 loss_sum_ctx += torch.Tensor([loss_ctx])
    #                 nb_steps_eph_ctx += nb_steps_ctx_

    #                 nb_batches_ctx += 1

    #         loss_epoch_node = loss_sum_node/nb_batches_node
    #         loss_epoch_ctx = loss_sum_ctx/nb_batches_ctx

    #         losses_node.append(loss_epoch_node)
    #         losses_ctx.append(loss_epoch_ctx)
    #         nb_steps_node.append(nb_steps_eph_node)
    #         nb_steps_ctx.append(nb_steps_eph_ctx)

    #         if epoch%print_error_every==0 or epoch<=3 or epoch==nb_epochs-1:

    #             if val_dataloader is not None:
    #                 self.learner.neuralode = node
    #                 self.learner.contexts = contexts
    #                 ind_crit,_ = tester.test(val_dataloader, int_cutoff=1.0, criterion=val_criterion, verbose=False)
    #                 val_losses.append(np.array([epoch, ind_crit]))
    #                 print(f"    Epoch: {epoch:-5d}      LossTrajs: {loss_epoch_node[0]:-.8f}     ContextsNorm: {torch.mean(term2):-.8f}     ValIndCrit: {ind_crit:-.8f}", flush=True)
    #             else:
    #                 print(f"    Epoch: {epoch:-5d}      LossTrajs: {loss_epoch_node[0]:-.8f}     ContextsNorm: {torch.mean(term2):-.8f}", flush=True)

    #     wall_time = time.time() - start_time
    #     time_in_hmsecs = seconds_to_hours(wall_time)
    #     print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)
    #     print("Environment weights at the end of the training:", weightings)

    #     self.losses_node.append(torch.vstack(losses_node))
    #     self.losses_ctx.append(torch.vstack(losses_ctx))
    #     self.nb_steps_node.append(torch.array(nb_steps_node))
    #     self.nb_steps_ctx.append(torch.array(nb_steps_ctx))

    #     if val_dataloader is not None:
    #         self.val_losses.append(np.vstack(val_losses))

    #     self.learner.neuralode = node
    #     self.learner.contexts = contexts

    #     if save_path:
    #         self.save_trainer(save_path)



    # def train_proximal(self, nb_outer_steps_max, int_prop=1.0, inner_tol_node=1e-2, inner_tol_ctx=1e-2, nb_inner_steps_max=10, proximal_reg=100., patience=None, print_error_every=1, save_path=False, val_dataloader=None, val_criterion=None, key=None):
    #     """ Train the model using the proximal gradient descent algorithm. Algorithm 2 in https://proceedings.mlr.press/v97/li19n.html"""

    #     opt_state_node = self.opt_node_state
    #     opt_state_ctx = self.opt_ctx_state

    #     loss_fn = self.learner.loss_fn

    #     node = self.learner.neuralode
    #     contexts = self.learner.contexts

    #     @eqx.filter_jit
    #     def train_step_node(node, node_old, contexts, batch, weights, opt_state, key):
    #         print('\nCompiling function "train_step" for neural ode ...')

    #         def prox_loss_fn(node, contexts, batch, weights, key):
    #             loss, aux_data = loss_fn(node, contexts, batch, weights, key)
    #             diff_norm = params_diff_norm_squared(node, node_old)
    #             return loss + proximal_reg * diff_norm / 2., (*aux_data, diff_norm)

    #         (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(node, contexts, batch, weights, key)

    #         updates, opt_state = self.opt_node.update(grads, opt_state)
    #         # updates = jax.tree_map(lambda x: -x*1e-4, grads)
    #         node = eqx.apply_updates(node, updates)

    #         return node, contexts, opt_state, loss, aux_data


    #     @eqx.filter_jit
    #     def train_step_ctx(node, contexts, contexts_old, batch, weights, opt_state, key):
    #         print('\nCompiling function "train_step" for context ...')

    #         def prox_loss_fn(contexts, node, batch, weights, key):
    #             loss, aux_data = loss_fn(node, contexts, batch, weights, key)
    #             diff_norm = params_diff_norm_squared(contexts, contexts_old)
    #             return loss + proximal_reg * diff_norm / 2., (*aux_data, diff_norm)

    #         (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(contexts, node, batch, weights, key)

    #         updates, opt_state = self.opt_ctx.update(grads, opt_state)
    #         # updates = jax.tree_map(lambda x: -x*1e-4, grads)
    #         contexts = eqx.apply_updates(contexts, updates)

    #         return node, contexts, opt_state, loss, aux_data


    #     assert int_prop>0 and int_prop<=1.0, "The proportion of trajectory length to consider for training must be between 0 and 1"
    #     self.dataloader.int_cutoff = int(int_prop*self.dataloader.nb_steps_per_traj)

    #     if val_dataloader is not None:
    #         tester = VisualTester(self)

    #     print(f"\n\n=== Beginning training with proximal alternating minimization ... ===")
    #     print(f"    Number of examples in a batch: {self.dataloader.batch_size}")
    #     print(f"    Maximum number of steps per inner minimization: {nb_inner_steps_max}")
    #     print(f"    Maximum number of outer minimizations: {nb_outer_steps_max}")
    #     print(f"    Maximum total number of training steps: {nb_outer_steps_max*nb_inner_steps_max}")

    #     start_time = time.time()

    #     losses_node = []
    #     losses_ctx = []
    #     nb_steps_node = []
    #     nb_steps_ctx = []

    #     val_losses = []

    #     weights = jnp.ones(self.learner.nb_envs) / self.learner.nb_envs

    #     loss_key = get_new_key(key)

    #     early_stopping_count = 0

    #     for out_step in range(nb_outer_steps_max):

    #         node_old = jax.tree_util.tree_map(lambda x: x, node)
    #         contexts_old = jax.tree_util.tree_map(lambda x: x, contexts)

    #         # node_prev = node        ## TODO check that this is passed by value and not by reference
    #         node_prev = jax.tree_util.tree_map(lambda x: x, node)
    #         for in_step_node in range(nb_inner_steps_max):

    #             nb_batches_node = 0
    #             loss_sum_node = jnp.zeros(1)
    #             nb_steps_eph_node = 0

    #             for i, batch in enumerate(self.dataloader):
    #                 loss_key = get_new_key(loss_key)

    #                 node, contexts, opt_state_node, loss_node, (nb_steps_node_, term1, term2, diff_node_) = train_step_node(node, node_old, contexts, batch, weights, opt_state_node, loss_key)

    #                 # if i%1==0:
    #                     # term1 = term1 + 1e-16
    #                     # weights = term1 / jnp.sum(term1)

    #                 loss_sum_node += jnp.array([loss_node])
    #                 nb_steps_eph_node += nb_steps_node_

    #                 nb_batches_node += 1

    #             diff_node = params_diff_norm_squared(node, node_prev) / params_norm_squared(node_prev)
    #             if diff_node < inner_tol_node or out_step==0:       ## Break early to see how big the loss is at the beginning
    #                 break
    #             # diff_node_prev = diff_node
    #             node_prev = node

    #         loss_epoch_node = loss_sum_node/nb_batches_node


    #         # contexts_prev = contexts
    #         contexts_prev = jax.tree_util.tree_map(lambda x: x, contexts)
    #         # diff_ctx_prev = 2*inner_tol
    #         for in_step_ctx in range(nb_inner_steps_max):

    #             nb_batches_ctx = 0
    #             loss_sum_ctx = jnp.zeros(1)
    #             nb_steps_eph_ctx = 0

    #             for i, batch in enumerate(self.dataloader):

    #                 node, contexts, opt_state_ctx, loss_ctx, (nb_steps_ctx_, term1, term2, diff_ctx_) = train_step_ctx(node, contexts, contexts_old, batch, weights, opt_state_ctx, loss_key)

    #                 # term1 = term1 + 1e-8
    #                 # weights = term1 / jnp.sum(term1)

    #                 loss_sum_ctx += jnp.array([loss_ctx])
    #                 nb_steps_eph_ctx += nb_steps_ctx_

    #                 nb_batches_ctx += 1

    #             # if diff_ctx + diff_ctx_prev < inner_tol:
    #             diff_ctx = params_diff_norm_squared(contexts, contexts_prev) / params_norm_squared(contexts_prev)
    #             if diff_ctx < inner_tol_ctx or out_step==0:
    #                 break
    #             # diff_ctx_prev = diff_ctx
    #             contexts_prev = contexts

    #         loss_epoch_ctx = loss_sum_ctx/nb_batches_ctx



    #         losses_node.append(loss_epoch_node)
    #         losses_ctx.append(loss_epoch_ctx)
    #         nb_steps_node.append(nb_steps_eph_node)
    #         nb_steps_ctx.append(nb_steps_eph_ctx)

    #         if out_step%print_error_every==0 or out_step<=3 or out_step==nb_outer_steps_max-1:

    #             if val_dataloader is not None:
    #                 self.learner.neuralode = node
    #                 self.learner.contexts = contexts
    #                 ind_crit,_ = tester.test(val_dataloader, int_cutoff=1.0, criterion=val_criterion, verbose=False)
    #                 val_losses.append(np.array([out_step, ind_crit]))
    #                 print(f"    Outer Step: {out_step:-5d}      LossTrajs: {loss_epoch_node[0]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}     ValIndCrit: {ind_crit:-.8f}", flush=True)
    #                 ## Check if val loss is lowest to save the model
    #                 if ind_crit <= jnp.stack(val_losses)[:,1].min() and save_path:
    #                     print(f"        Saving best model so far ...")
    #                     self.learner.save_learner(save_path)
    #                 ## Restore the learner at the last evaluation step
    #                 if out_step == nb_outer_steps_max-1:
    #                     # print(f"        Restoring model to the best one found during training ...")
    #                     self.learner.load_learner(save_path)

    #             else:
    #                 print(f"    Epoch: {out_step:-5d}      LossTrajs: {loss_epoch_node[0]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}", flush=True)

    #             print(f"        -NbInnerStepsNode: {in_step_node+1:4d}\n        -NbInnerStepsCxt: {in_step_ctx+1:4d}\n        -InnerToleranceNode: {inner_tol_node:.2e}\n        -InnerToleranceCtx:  {inner_tol_ctx:.2e}\n        -DiffNode: {diff_node:.2e}\n        -DiffCxt:  {diff_ctx:.2e}", flush=True)

    #         if in_step_node < 1 and in_step_ctx < 1:     ## These numbers could be played with
    #             early_stopping_count += 1
    #         else:
    #             early_stopping_count = 0

    #         # term1 = term1 + 1e-16
    #         # weights = term1 / jnp.sum(term1)

    #         if (patience is not None) and (early_stopping_count >= patience):
    #             print(f"Stopping early after {patience} steps with no improvement in the loss. Consider increasing the tolerances for the inner minimizations.")
    #             break


    #     wall_time = time.time() - start_time
    #     time_in_hmsecs = seconds_to_hours(wall_time)
    #     print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)
    #     print("Environment weights at the end of the training:", weights)

    #     self.losses_node.append(jnp.vstack(losses_node))
    #     self.losses_ctx.append(jnp.vstack(losses_ctx))
    #     self.nb_steps_node.append(jnp.array(nb_steps_node) / (in_step_node+1))
    #     self.nb_steps_ctx.append(jnp.array(nb_steps_ctx)/ (in_step_ctx+1))

    #     if val_dataloader is not None:
    #         self.val_losses.append(np.vstack(val_losses))

    #     self.opt_node_state = opt_state_node
    #     self.opt_ctx_state = opt_state_ctx

    #     if val_dataloader is None:
    #         self.learner.neuralode = node
    #         self.learner.contexts = contexts

    #     # Save the model and results
    #     if save_path:
    #         self.save_trainer(save_path)



    def save_trainer(self, path):
        assert path[-1] == "/", "ERROR: The path must end with /"
        # print(f"\nSaving model and results into {path} folder ...\n")

        np.savez(path+"train_histories.npz",
                 losses_node=torch.vstack(self.losses_node), 
                 losses_ctx=torch.vstack(self.losses_ctx), 
                 nb_steps_node=np.concatenate(self.nb_steps_node), 
                 nb_steps_ctx=np.concatenate(self.nb_steps_ctx))
        
        if hasattr(self, 'val_losses'):
            np.save(path+"val_losses.npy", np.vstack(self.val_losses))

        pickle.dump(self.opt_node.state_dict(), open(path+"opt_state_node.pkl", "wb"))
        pickle.dump(self.opt_ctx.state_dict(), open(path+"opt_state_ctx.pkl", "wb"))

        if not hasattr(self, 'val_losses'):
            self.learner.save_learner(path)


    def restore_trainer(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"
        print(f"\nNo training, loading model and results from {path} folder ...\n")

        histories = np.load(path+"train_histories.npz")
        self.losses_node = [histories['losses_node']]
        self.losses_ctx = [histories['losses_ctx']]
        self.nb_steps_node = [histories['nb_steps_node']]
        self.nb_steps_ctx = [histories['nb_steps_ctx']]

        if os.path.exists(path+"val_losses.npy"):
            self.val_losses = [np.load(path+"val_losses.npy")]

        self.opt_node = pickle.load(open(path+"opt_state_node.pkl", "rb"))
        self.opt_ctx = pickle.load(open(path+"opt_state_ctx.pkl", "rb"))

        self.learner.load_learner(path)


    def adapt(self, data_loader, nb_epochs, print_error_every=100, save_path=False):
        """Adapt the model to new environments (in bulk) using the provided dataset. """

        loss_fn = self.learner.loss_fn
        node = self.learner.neuralode

        ## Freeze the shared neural ODE weights (quires_grad=False)
        for param in node.parameters():
            param.requires_grad = False
        
        ## Disable the Taylor expansion
        node.vectorfield.taylor_order = 0

        contexts = ContextParams(data_loader.nb_envs, self.learner.contexts.params.shape[1])
        contexts.to(data_loader.device)

        opt = optim.Adam(contexts.parameters(), lr=self.learning_rates[1])
        self.learner.init_ctx_params_adapt = contexts.params.clone().detach()
        self.losses_adapt = []
        self.nb_steps_adapt = []

        nb_train_steps_per_epoch = int(np.ceil(data_loader.nb_trajs_per_env / data_loader.batch_size))
        total_steps = nb_epochs * nb_train_steps_per_epoch

        print(f"\n\n=== Beginning adaptation ... ===")
        print(f"    Number of examples in a batch: {data_loader.batch_size}")
        print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
        print(f"    Number of training epochs: {nb_epochs}")
        print(f"    Total number of training steps: {total_steps}")

        start_time = time.time()

        losses = []
        nb_steps = []

        for epoch in range(nb_epochs):
            nb_batches = 0
            loss_sum = np.zeros(1)
            nb_steps_eph = 0

            for i, batch in enumerate(data_loader):

                opt.zero_grad()
                loss, (nb_steps_, term1, term2) = loss_fn(node, contexts, batch)
                loss.backward()
                opt.step()

                loss_sum += np.array([loss.detach().cpu().numpy()])
                nb_steps_eph += nb_steps_

                nb_batches += 1

            loss_epoch = loss_sum/nb_batches

            losses.append(loss_epoch)
            nb_steps.append(nb_steps_eph)

            if epoch%print_error_every==0 or epoch<=3 or epoch==nb_epochs-1:
                print(f"    Epoch: {epoch:-5d}     LossContext: {loss_epoch[0]:-.8f}", flush=True)

        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent adaptation time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_adapt.append(np.vstack(losses))
        self.nb_steps_adapt.append(np.array(nb_steps))

        self.opt_adapt = opt
        self.opt_state_adapt = opt.state_dict()

        self.learner.contexts_adapt = contexts

        if save_path:
            self.save_adapted_trainer(save_path)







    # def adapt_sequential(self, data_loader, nb_epochs, optimizer=None, print_error_every=100, save_path=False, key=None):
    #     """Adapt the model to a new environment using the provided dataset. """
    #     # key = key if key is not None else self.key

    #     loss_fn = self.learner.loss_fn
    #     node = self.learner.neuralode

    #     if optimizer is None:       ## You want to continue a previous adaptation !!!
    #         # if self.opt_adapt is not None:
    #         if hasattr(self, 'opt_adapt'):
    #             print("WARNING: No optimizer provided for adaptation, using any previrouly defined for adapation")
    #             opt = self.opt_adapt
    #             contexts = self.learner.contexts_adapt
    #             opt_state = self.opt_state_adapt
    #         else:
    #             raise ValueError("No optimizer provided for adaptation, and none previously defined")
    #     else:
    #         opt = optimizer
    #         self.losses_adapt = []
    #         self.nb_steps_adapt = []

    #     def train_step(node, contexts, batch, weights, key):
    #         print('\nCompiling function "train_step" for context ...')

    #         opt.zero_grad()
    #         loss, aux_data = loss_fn(node.freeze(), contexts, batch, weights, key)
    #         loss.backward()
    #         opt.step()

    #         return node, contexts, loss, aux_data

    #     nb_train_steps_per_epoch = int(np.ceil(data_loader.nb_trajs_per_env / data_loader.batch_size))
    #     total_steps = nb_epochs * nb_train_steps_per_epoch

    #     print(f"\n\n=== Beginning sequential adaptation ... ===")
    #     print(f"    Number of examples in a batch: {data_loader.batch_size}")
    #     print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
    #     print(f"    Number of training epochs: {nb_epochs}")
    #     print(f"    Total number of training steps: {total_steps}")

    #     nb_adapt_envs = data_loader.nb_envs

    #     contexts = []
    #     all_losses = []
    #     all_nb_steps = []
    #     inits_ctx = []
    #     for env_id in range(nb_adapt_envs):

    #         start_time = time.time()

    #         print(f"\nAdapting to environment {env_id} ...")

    #         new_dataset = data_loader.dataset[env_id:env_id+1,...]

    #         new_dataloader = DataLoader(new_dataset, t_eval=data_loader.t_eval, adaptation=True, data_id=data_loader.data_id+str(env_id), key=key)

    #         opt = optimizer
    #         context = ContextParams(1, self.learner.contexts.params.shape[1], key)
    #         inits_ctx.append(context.params)
    #         opt_state = opt.init(context)

    #         losses = []
    #         nb_steps = []

    #         weights = jnp.ones(1)
    #         loss_key = get_new_key(key)

    #         for epoch in range(nb_epochs):
    #             nb_batches = 0
    #             loss_sum = jnp.zeros(1)
    #             nb_steps_eph = 0

    #             for i, batch in enumerate(new_dataloader):
    #                 loss_key = get_new_key(loss_key)

    #                 node, context, opt_state, loss, (nb_steps_, term1, term2) = train_step(node, context, batch, weights, opt_state, loss_key)

    #                 loss_sum += jnp.array([loss])
    #                 nb_steps_eph += nb_steps_

    #                 nb_batches += 1

    #             loss_epoch = loss_sum/nb_batches

    #             losses.append(loss_epoch)
    #             nb_steps.append(nb_steps_eph)

    #             if epoch%print_error_every==0 or epoch<=3 or epoch==nb_epochs-1:
    #                 print(f"    Epoch: {epoch:-5d}     LossContext: {loss_epoch[0]:-.8f}", flush=True)

    #         wall_time = time.time() - start_time
    #         time_in_hmsecs = seconds_to_hours(wall_time)
    #         print("\nGradient descent adaptation time: %d hours %d mins %d secs" %time_in_hmsecs)

    #         contexts.append(context.params)
    #         all_losses.append(jnp.stack(losses))
    #         all_nb_steps.append(jnp.array(nb_steps))


    #     # data_loader.dataset = orig_dataset
    #     contexts = eqx.tree_at(lambda c: c.params, 
    #                             ContextParams(nb_adapt_envs, self.learner.contexts.params.shape[1], None), 
    #                             jnp.concatenate(contexts))

    #     self.losses_adapt.append(jnp.mean(jnp.stack(all_losses), axis=0))
    #     self.nb_steps_adapt.append(jnp.mean(jnp.stack(all_nb_steps), axis=0))

    #     self.opt_adapt = opt
    #     self.opt_state_adapt = opt.init(contexts)

    #     self.learner.contexts_adapt = contexts
    #     self.learner.init_ctx_params_adapt = jnp.concatenate(inits_ctx)

    #     if save_path:
    #         self.save_adapted_trainer(save_path, data_loader.data_id)










    def save_adapted_trainer(self, path):
        print(f"\nSaving adaptation parameters into {path} folder ...\n")

        np.savez(path+"adapt_histories_.npz", 
                 losses_adapt=np.vstack(self.losses_adapt), 
                 nb_steps_adapt=np.concatenate(self.nb_steps_adapt))

        pickle.dump(self.opt_state_adapt, open(path+"/opt_state_adapt.pkl", "wb"))

        torch.save(self.learner.contexts_adapt, path+"/adapted_contexts_.pt")

        np.save(path+"adapted_contexts_init_.npy", self.learner.init_ctx_params_adapt.cpu().numpy())


    def restore_adapted_trainer(self, path, data_loader=None):

        if data_loader is None:
            ValueError("ERROR: You must provide the dataset on which this system was adapted.")
        load_id = data_loader.data_id

        print(f"\nNo adaptation, loading adaptation parameters from {path} folder with ...\n")

        histories = np.load(path+"adapt_histories_.npz")
        self.losses_adapt = [histories['losses_adapt']]
        self.nb_steps_adapt = [histories['nb_steps_adapt']]

        self.opt_state_adapt = pickle.load(open(path+"/opt_state_adapt.pkl", "rb"))

        self.learner.contexts_adapt = ContextParams(data_loader.nb_envs, self.learner.contexts.params.shape[1], None)

        self.learner.contexts_adapt = torch.load(path+"/adapted_contexts_.pt")

        self.learner.init_ctx_params_adapt = np.load(path+"adapted_contexts_init_.npy")
