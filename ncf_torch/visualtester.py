from ._utils import *


class VisualTester:
    def __init__(self, trainer, key=None):

        self.trainer = trainer

    def test(self, data_loader, criterion=None, int_cutoff=1.0, verbose=True):
        """ Compute test metrics on the adaptation dataloader  """

        criterion = criterion if criterion else lambda x, x_hat: torch.mean((x-x_hat)**2)

        t_eval = data_loader.t_eval
        test_length = int(data_loader.nb_steps_per_traj*int_cutoff)
        X = data_loader.dataset[:, :, :test_length, :]
        t_test = t_eval[:test_length]

        if verbose == True:
            if data_loader.adaptation == False:
                print("==  Begining in-domain testing ... ==")
                print("    Number of training environments:", self.trainer.dataloader.nb_envs)
            else:
                print("==  Begining out-of-distribution testing ... ==")
                print("    Number of training environments:", self.trainer.dataloader.nb_envs)
                print("    Number of adaptation environments:", data_loader.nb_envs)
            print("    Final length of the training trajectories:", self.trainer.dataloader.int_cutoff)
            print("    Length of the testing trajectories:", test_length)

        if data_loader.adaptation == False:
            contexts = self.trainer.learner.contexts.params
        else:
            contexts = self.trainer.learner.contexts_adapt.params

        X_hat = []
        for e in range(data_loader.nb_envs):
            X_hat_e, _ = self.trainer.learner.neuralode(X[e, :, 0, :], 
                                                        t_test, 
                                                        contexts[e],
                                                        contexts[e])
            X_hat.append(X_hat_e)
        X_hat = torch.stack(X_hat, dim=0)

        crit_all = []
        for e in range(data_loader.nb_envs):
            crit_e = criterion(X[e], X_hat[e])
            crit_all.append(crit_e)
        crit_all = torch.stack(crit_all, dim=0).detach().cpu().numpy()

        crit = crit_all.mean(axis=0)

        if verbose == True:
            if data_loader.adaptation == False:
                print("Test Score (In-Domain):", crit)
            else:
                print("Test Score (OOD):", crit)
            print(flush=True)

        return crit, crit_all


    def visualize(self, data_loader, e=None, traj=None, dims=(0,1), context_dims=(0,1), int_cutoff=1.0, save_path=False, key=None):

        e = e if e is not None else np.random.randint(0, data_loader.nb_envs, (1,))[0]
        traj = traj is not None if traj else np.random.randint(0, data_loader.nb_trajs_per_env, (1,))[0]

        t_eval = data_loader.t_eval
        test_length = int(data_loader.nb_steps_per_traj*int_cutoff)
        X = data_loader.dataset[e, traj:traj+1, :test_length, :]
        t_test = t_eval[:test_length]

        if data_loader.adaptation == False:
            print("==  Begining in-domain visualisation ... ==")
        else:
            print("==  Begining out-of-distribution visualisation ... ==")
        print("    Environment id:", e)
        print("    Trajectory id:", traj)
        print("    Visualized dimensions:", dims)
        print("    Final length of the training trajectories:", self.trainer.dataloader.int_cutoff)
        print("    Length of the testing trajectories:", test_length)

        if data_loader.adaptation == False:
            contexts = self.trainer.learner.contexts.params
        else:
            contexts = self.trainer.learner.contexts_adapt.params
        X_hat, _ = self.trainer.learner.neuralode(X[:, 0, :],
                                            t_test, 
                                            contexts[e],
                                            contexts[e])

        X_hat = X_hat.squeeze().detach().cpu().numpy()
        X = X.squeeze().detach().cpu().numpy()
        t_plot = t_test.detach().cpu().numpy()

        fig, ax = plt.subplot_mosaic('AB;CC;DD;EF', figsize=(6*2, 3.5*4))

        mks = 2
        dim0, dim1 = dims

        ax['A'].plot(t_plot, X[:, 0], c="deepskyblue", label=f"$x_{{{dim0}}}$ (GT)")
        ax['A'].plot(t_plot, X_hat[:, 0], "o", c="royalblue", label=f"$\\hat{{x}}_{{{dim0}}}$ (NCF)", markersize=mks)

        ax['A'].plot(t_plot, X[:, 1], c="violet", label=f"$x_{{{dim1}}}$ (GT)")
        ax['A'].plot(t_plot, X_hat[:, 1], "x", c="purple", label=f"$\\hat{{x}}_{{{dim1}}}$ (NCF)", markersize=mks)

        ax['A'].set_xlabel("Time")
        ax['A'].set_ylabel("State")
        ax['A'].set_title("Trajectories")
        ax['A'].legend()

        ax['B'].plot(X[:, 0], X[:, 1], c="turquoise", label="GT")
        ax['B'].plot(X_hat[:, 0], X_hat[:, 1], ".", c="teal", label="NCF")
        ax['B'].set_xlabel(f"$x_{{{dim0}}}$")
        ax['B'].set_ylabel(f"$x_{{{dim1}}}$")
        ax['B'].set_title("Phase space")
        ax['B'].legend()

        nb_envs = data_loader.nb_envs

        nb_steps = np.concatenate(self.trainer.nb_steps_node)
        losses_node = np.vstack(self.trainer.losses_node)
        losses_ctx = np.vstack(self.trainer.losses_ctx)
        xis = self.trainer.learner.contexts.params.detach().cpu().numpy()
        init_xis = self.trainer.learner.init_ctx_params.detach().cpu().numpy()

        if data_loader.adaptation == True:  ## Overwrite the above if adaptation
            nb_steps = np.concatenate(self.trainer.nb_steps_adapt)
            losses_node = np.vstack(self.trainer.losses_adapt)      ## Replotting the label context !
            losses_ctx = np.vstack(self.trainer.losses_adapt)
            xis = self.trainer.learner.contexts_adapt.params.detach().cpu().numpy()
            init_xis = self.trainer.learner.init_ctx_params_adapt.detach().cpu().numpy()

        mke = np.ceil(losses_node.shape[0]/100).astype(int)

        label_node = "Train Loss" if data_loader.adaptation == False else "Adapt Loss"
        ax['C'].plot(losses_node[:,0], label=label_node, color="grey", linewidth=3, alpha=1.0)

        # label_ctx = "Context Loss" if data_loader.adaptation == False else "Context Loss Adapt"
        # ax['C'].plot(losses_ctx[:,0], "x-", markevery=mke, markersize=mks, label=label_ctx, color="grey", linewidth=1, alpha=0.5)

        if data_loader.adaptation==False and hasattr(self.trainer, 'val_losses') and len(self.trainer.val_losses)>0:
            val_losses = np.vstack(self.trainer.val_losses)
            ax['C'].plot(val_losses[:,0], val_losses[:,1], "y.", label="Validation Loss", linewidth=3, alpha=0.5)

        ax['C'].set_xlabel("Epochs")
        ax['C'].set_title("Loss Terms")
        ax['C'].set_yscale('log')
        ax['C'].legend()

        ax['D'].plot(nb_steps, c="brown")
        ax['D'].set_xlabel("Epochs")
        ax['D'].set_title("Total Number of Steps Taken (Proportional to NFEs)")
        if np.all(nb_steps>0):
            ax['D'].set_yscale('log')

        eps = 0.1
        colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
        colors = colors*(nb_envs)
        cdim0, cdim1 = context_dims

        ax['E'].scatter(init_xis[:,cdim0], init_xis[:,cdim1], s=30, c=colors[:nb_envs], marker='X')
        ax['F'].scatter(xis[:,cdim0], xis[:,cdim1], s=50, c=colors[:nb_envs], marker='o')
        for i, (x, y) in enumerate(init_xis[:, context_dims]):
            ax['E'].annotate(str(i), (x, y), fontsize=8)
        for i, (x, y) in enumerate(xis[:, context_dims]):
            ax['F'].annotate(str(i), (x, y), fontsize=8)
        ax['E'].set_title(r'Initial Contexts')
        ax['E'].set_xlabel(f'dim {cdim0}')
        ax['E'].set_ylabel(f'dim {cdim1}')

        ax['F'].set_title(r'Final Contexts')
        ax['F'].set_xlabel(f'dim {cdim0}')
        ax['F'].set_ylabel(f'dim {cdim1}')

        plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

        plt.tight_layout()
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Testing finished. Figure saved in:", save_path);

