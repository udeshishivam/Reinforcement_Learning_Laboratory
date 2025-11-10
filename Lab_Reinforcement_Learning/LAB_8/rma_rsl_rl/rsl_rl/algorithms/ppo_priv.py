import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.modules import ActorCriticLatent
from rsl_rl.storage import RolloutStorageRMA
from rsl_rl.algorithms import PPO

# This algorithm includes the mirror loss
# https://arxiv.org/pdf/1801.08093.pdf

class PPO_priv(PPO):
    actor_critic: ActorCriticLatent
    def __init__(self, 
                 actor_critic_latent, 
                 num_learning_epochs=1, 
                 num_mini_batches=1, 
                 clip_param=0.2, 
                 gamma=0.998, 
                 lam=0.95, 
                 value_loss_coef=1, 
                 entropy_coef=0, 
                 learning_rate=0.001, 
                 max_grad_norm=1, 
                 use_clipped_value_loss=True, 
                 schedule="fixed", 
                 desired_kl=0.01, 
                 device='cpu',
                 **kwargs):
        if kwargs:
            print("PPO_priv.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
            
        super().__init__(actor_critic_latent, num_learning_epochs, num_mini_batches, clip_param, gamma, lam, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, use_clipped_value_loss, schedule, desired_kl, device)
        
        self.transition = RolloutStorageRMA.Transition()
        self.mirror_weight = 0
        print("Priveleged PPO is loaded")

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, action_shape):
        self.storage = RolloutStorageRMA(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, action_shape, self.device)      

    def act(self, obs, privilege_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        total_obs=torch.cat((obs,privilege_obs),dim=-1)
        self.transition.actions = self.actor_critic.act(total_obs).detach()
        self.transition.values = self.actor_critic.evaluate(total_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = privilege_obs
        return self.transition.actions
          
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
    
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, privilege_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
                # The shape of an obs batch is : (minibatchsize, obs_shape)
                total_obs_batch = torch.cat((obs_batch,privilege_obs_batch),dim=-1)
                self.actor_critic.act(total_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(total_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate
                


                mirror_loss=0
                
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = self.mirror_weight * mirror_loss +surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss