from nets import Discriminator, Learner
from aisgail.utils import GailConfig,make_env, TemporaryBuffer, PreparedBuffer
import torch
from torch.nn import BCELoss, MSELoss

class GAIL:
    def __init__(self, conf: GailConfig) -> None:
        
        self.conf = conf
        # Load gym Env from path provided in config file
        self.env = make_env(conf.env_id,conf.env_path)
        self.state_dim = self.env.observation_space.shape
        self.n_actions = self.env.action_space.shape

        # Init current learner Policy (π_theta)
        self.learner = Learner(self.state_dim, self.n_actions)

        # Init old learner policy (π_theta_old) as copy of learner
        self.learner_old = Learner(self.state_dim,self.n_actions)
        self.learner_old.load_state_dict(self.learner.state_dict())

        # Init the discriminator network
        self.discriminator = Discriminator(self.state_dim,self.n_actions)

        # Optimizer for the learner
        self.optimizer = torch.optim.Adam([
            {"params": self.learner.actor.parameters(), "lr": self.conf.lr_actor},
            {"params": self.learner.critic.parameters(), "lr": self.conf.lr_critic}
        ])

        # Optimizer for the discriminator
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.conf.lr_discriminator
        )

        # Mean-square error and binary-cross-entropy loss
        self.mse_loss = MSELoss()
        self.bce_loss = BCELoss()

        # TODO Load expert trajectories
        self.expert_state_actions = ...

        # Init buffer for saving trajectories
        self.buffer = TemporaryBuffer()
        
        # Check for cuda support and select device
        if torch.cuda.is_available():
            self.device = "cuda"
        else: self.device = "cpu"

    def step(self, state: torch.Tensor):
        """
        Act in the evironment for a sinlge step,
        and save information on 
            - state
            - action 
            - action log probabilities
            - reward
            and
            - termination sinal 
            to the temporaray trajectory buffer
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, action_log_prob = self.learner_old.select_action(state)
        action = action.detach().item()
        s2, r, d, _ = self.env.step(action)

        self.buffer.states.append(state.detach())
        self.buffer.actions.append(action)
        self.buffer.action_log_probs.append(action_log_prob.detach())

        self.buffer.rewards.append(r)
        self.buffer.is_terminal.append(d)

        return s2, r, d
    
    def update_discriminator(self,pbuffer: PreparedBuffer):
        """
        Perform a single update with the
        discriminator. 
        
        Aim for the discriminator is to 
        learn whether a provided trajectory 
        stems from the learner or an expert
        demostration.
        """
        # ------ Buffer Prep --------------------
        #prep = self.buffer.prepare(self.device,self.n_actions)
        assert pbuffer.learner_state_actions.size()==self.expert_state_actions.size(),\
             "Learner and expert trajectories must have equal size"
        
        # ------- Discriminator updating ---------
        expert_probs = self.discriminator(self.expert_state_actions)
        learner_probs = self.discriminator(pbuffer.learner_state_actions)
        learner_loss = self.bce_loss(
            learner_probs, torch.ones(
                (pbuffer.learner_state_actions.shape[0], 1), 
                device=self.device
            )
        )
        expert_loss = self.bce_loss(
            expert_probs, 
            torch.zeros(
                (self.expert_state_actions.shape[0], 1),
                device=self.device
            )
        )
        loss: torch.Tensor = learner_loss + expert_loss
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

    def update_policy(self, pbuffer: PreparedBuffer) -> None:
        """
        ake a policy step, using the TRPO rule 
        with cost function of the updated discriminator log(D(s, a))
        """
        # Get discounted rewards of learner, 
        # evaluated by the discriminator.
        # Input is a concatenation of states and one-hot-encoded
        # actions from the learners trajectory
        with torch.no_grad():
            dr = torch.log(self.discriminator(pbuffer.learner_state_actions))

        rewards = []
        # Cumulated discounted reward
        cdr = 0
        for i in reversed(range(len(dr))):
            cdr = dr[i] + self.conf.gamma * cdr
            rewards.append(cdr)
        
        # Cast rewards to tensor
        rewards = torch.tensor(rewards, dtype=torch.float32,device=self.device)

        # Normalize to mu=0, Var=1 | use small offset to circumvent division by zero
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        for _ in range(self.conf.num_epochs):
            values, log_prob_actions, entropy = self.learner.evaluate(
                pbuffer.state_hist,pbuffer.action_hist
            ) 
            # Andvantages
            avtgs = rewards - values.detach()

            # Get ratios for importance sampling
            imp_ratios = torch.exp(log_prob_actions,pbuffer.action_log_prob_hist)

            # Find the clipped surrogate objective
            lower, upper = 1-self.conf.clip_eps, 1+self.conf.clip_eps
            clamped_ratios = torch.clamp(imp_ratios,lower,upper)

            # Build loss
            l1 = -torch.min(imp_ratios,clamped_ratios) * avtgs
            l2 = 0.5 * self.mse_loss(values,rewards)
            l3 = -0.01 * entropy
            loss: torch.Tensor = l1+l2+l3
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.learner_old.load_state_dict(self.learner.state_dict())
        self.buffer.reset()


        

