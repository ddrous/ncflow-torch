from ._utils import *


class Learner:
    def __init__(self, vectorfield, contexts, loss_fn_env, ivp_args):
        self.nb_envs, self.context_size = contexts.params.shape

        self.neuralode = NeuralODE(vectorfield, ivp_args)
        self.contexts = contexts

        self.init_ctx_params = self.contexts.params.clone().detach()
        self.contexts_adapt = None

        # self.loss_fn = lambda model, contexts, batch: loss_fn(model, contexts, batch, loss_fn_env)
        self.loss_fn_env = loss_fn_env


    # @torch.compile()
    def loss_fn(self, batch, training=True):
        Xs, t_eval = batch

        if training:
            contexts = self.contexts
        else:   ## Adaptation
            contexts = self.contexts_adapt

        all_loss = []
        all_nb_steps = []
        all_term1 = []
        all_term2 = []
        for e in range(Xs.shape[0]):
            loss, (nb_steps, term1, term2) = self.loss_fn_env(self.neuralode, Xs[e, :, :, :], t_eval, contexts.params[e], contexts.params)
            all_loss.append(loss)
            all_nb_steps.append(nb_steps)
            all_term1.append(term1)
            all_term2.append(term2)

        all_loss = torch.stack(all_loss)
        all_nb_steps = torch.stack(all_nb_steps)
        all_term1 = torch.stack(all_term1)
        all_term2 = torch.stack(all_term2)

        recons = torch.mean(all_loss, dim=0)
        regul = 0.

        total_loss = recons + regul

        return total_loss, (torch.sum(all_nb_steps), all_term1, all_term2)



    def save_learner(self, path, suffix=None):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"
        
        if suffix is not None:
            torch.save(self.neuralode, path+f"model_{suffix}.pth")
            torch.save(self.contexts, path+f"contexts_{suffix}.pth")
            # torch.save(self.init_ctx_params, path+f"contexts_init_{suffix}.pth")
        else:
            torch.save(self.neuralode, path+"model.pth")
            torch.save(self.contexts, path+"contexts.pth")
            torch.save(self.init_ctx_params, path+"contexts_init.pth")

    def load_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"

        self.neuralode = torch.load(path+"model.pth", weights_only=False)
        self.contexts = torch.load(path+"contexts.pth", weights_only=False)

        if os.path.exists(path+"contexts_init.pth"):
            self.init_ctx_params = torch.load(path+"contexts_init.pth", weights_only=False)


class ContextParams(nn.Module):
    params: torch.nn.Parameter

    def __init__(self, nb_envs, context_size):
        super(ContextParams, self).__init__()
        self.params = nn.Parameter(torch.zeros((nb_envs, context_size)))

class NeuralODE(nn.Module):
    vectorfield: callable
    ivp_args: dict
    nfe: int

    def __init__(self, vectorfield, ivp_args):
        super(NeuralODE, self).__init__()
        self.ivp_args = ivp_args
        self.vectorfield = vectorfield
        self.nfe = 0

    def forward(self, x0s, t_eval, ctx, ctx_):

        def odefunc(t, x):
            self.nfe += 1
            return self.vectorfield(t, x, (ctx, ctx_.squeeze()))

        self.nfe = 0
        sol = odeint(
                odefunc,
                method=self.ivp_args.get("method", "dopri5"),
                t=t_eval,
                y0=x0s,
                rtol=self.ivp_args.get("rtol", 1e-3), 
                atol=self.ivp_args.get("atol", 1e-6),
            )

        ## Transpose the first two dimensions
        sol = sol.permute(1, 0, 2)

        return sol, self.nfe

