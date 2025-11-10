import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from rsl_rl.storage import HistoryStorage 

# computes and returns the latent from the expert
class DaggerExpert(nn.Module):
    def __init__(self, policy, nenvs,):
        super().__init__()
        self.policy = policy

    def forward(self, privilege_obs):
        with torch.no_grad():
            expert_latent = self.policy.prop_encoder(privilege_obs)
        return expert_latent

class DaggerAgent:
    def __init__(self, expert_policy,
                 prop_latent_encoder, student_mlp,
                 T, history_size, num_obs,device):
        expert_policy.to(device)
        prop_latent_encoder.to(device)
        student_mlp.to(device)
        self.expert_policy = expert_policy
        self.prop_latent_encoder = prop_latent_encoder # The student environment latent encoder
        self.student_mlp = student_mlp
        self.history_size = history_size
        self.num_obs = num_obs
        self.device = device
        self.itr = 0
        self.current_prob = 0
        
        # copy expert weights for mlp policy
        self.student_mlp.architecture.load_state_dict(self.expert_policy.policy.action_mlp.state_dict())
        for net_i in [self.expert_policy.policy, self.student_mlp]:
            for param in net_i.parameters():
                param.requires_grad = False

    def set_itr(self, itr):
        self.itr = itr
        if (itr+1) % 100 == 0:
            self.current_prob += 0.1
            print(f"Probability set to {self.current_prob}")

    # ---- Get Latents ---- #
    def get_student_latent(self, history):
        '''Given proprioceptive history, return \hat{z_t}'''
        
        #TODO(student): Return the predicted latent 
        with torch.no_grad():
          student_latent = self.prop_latent_encoder(history)
        return student_latent
    
    def get_expert_latent(self, privilege_obs):
        '''Given privileged obs, return z_t'''
        
        # TODO(student): Return the expert latent
        # Note: don't compute gradients 
        
        exp_latent = self.expert_policy(privilege_obs)
        return exp_latent


    
    # ---- Get Actions ---- #
    def evaluate(self, obs, history):
        prop_latent = self.get_student_latent(history)
        output = torch.cat([obs, prop_latent], 1)
        
        output = self.student_mlp.architecture(output)
        return output

    def get_expert_action(self, obs, privilege_obs):
        expert_latent = self.get_expert_latent(privilege_obs)
        output = torch.cat([obs, expert_latent], 1)
        
        output = self.student_mlp.architecture(output)
        return output

    def get_student_action(self, obs, history):
        return self.evaluate(obs,history)



    def save_deterministic_graph(self, fname_prop_encoder,
                                 fname_mlp, example_input, device='cpu'):
        hlen = self.base_obs_acts_size * self.T

        prop_encoder_graph = torch.jit.trace(self.prop_latent_encoder.to(device), example_input[:, :hlen])
        torch.jit.save(prop_encoder_graph, fname_prop_encoder)

        mlp_graph = torch.jit.trace(self.student_mlp.architecture.to(device), example_input[:, hlen:])
        torch.jit.save(mlp_graph, fname_mlp)

        self.prop_latent_encoder.to(self.device)
        self.student_mlp.to(self.device)

class DaggerTrainer:
    def __init__(self,
            actor,
            num_envs, 
            num_transitions_per_env,
            obs_shape, 
            latent_shape,
            num_learning_epochs=4,
            num_mini_batches=4,
            device=None,
            learning_rate=5e-4,
            ):
        

        self.actor = actor
        self.storage = None
        self.optimizer = optim.Adam([*self.actor.prop_latent_encoder.parameters()],
                                    lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)
        self.device = device
        self.itr = 0

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.loss_fn = nn.MSELoss()

    def init_storage(self, num_envs, num_transitions_per_env, history_shape, latent_shape, obs_shape):
        self.storage = HistoryStorage(num_envs, num_transitions_per_env, history_shape, latent_shape, self.device)

    def observe(self, history, obs):
        '''
        Returns the student policy action given history and obs
        Note: We don't want to compute gradients here 
        '''
        
        #TODO(student): Return the student action
        #Note: don't compute gradients here
        with torch.no_grad():
          fin_obs = self.actor.get_student_action(obs=obs, history=history)
        return fin_obs


    def step(self, privilege_obs: torch.Tensor, history: torch.Tensor):
        '''Each step, store the expert_latent and proprioceptive history in HistoryStorage() stucture'''
        
        # TODO(student): Store the history and expert_latent in self.storage 
        exp_latent = self.actor.get_expert_latent(privilege_obs) 
        self.storage.add_inputs(history, exp_latent)

    def update(self):
        # return loss in the last epoch
        prop_mse = 0
        loss_counter = 0
        for history_batch, expert_latent_batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):

            # TODO(student): Calculate the loss using nn.MSELoss (self.loss_fn) to optimized the adaptaion encoder.
            pred_latent = self.actor.prop_latent_encoder(history_batch)
            loss_prop = self.loss_fn(pred_latent, expert_latent_batch)

            # Gradient step
            loss = loss_prop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            prop_mse += loss_prop.item()
            loss_counter += 1
        num_updates = self.num_learning_epochs * self.num_mini_batches
        avg_prop_loss = prop_mse / num_updates 
        self.storage.clear()
        self.scheduler.step()
        return avg_prop_loss, None

