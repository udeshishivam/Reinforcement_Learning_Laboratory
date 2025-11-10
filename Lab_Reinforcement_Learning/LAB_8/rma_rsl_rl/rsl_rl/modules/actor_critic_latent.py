import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal

class MLPEncode(nn.Module):
    def __init__(self, shape,
                 actionvation_fn, 
                 base_obdim, 
                 output_size, 
                 output_activation_fn = None, 
                 small_init= False, 
                 priv_dim = 261, ):
        super(MLPEncode, self).__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        self.base_obs_dim = base_obdim
        
        
      
        ## Encoder Architecture
        prop_latent_dim = 8
        self.prop_latent_dim = prop_latent_dim

        self.prop_encoder =  nn.Sequential(*[
                                    nn.Linear(priv_dim, 256), self.activation_fn,
                                    nn.Linear(256, 128), self.activation_fn,
                                    nn.Linear(128, prop_latent_dim), self.activation_fn,
                                    ])
        
        
        # creating the action encoder
        modules = [nn.Linear(self.base_obs_dim + prop_latent_dim, shape[0]), self.activation_fn]
    

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn)

        modules.append(nn.Linear(shape[-1], output_size))
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn)

        self.action_mlp = nn.Sequential(*modules)

        self.input_shape = [base_obdim]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self, x):
        base_obs = x[:,:self.base_obs_dim] # (b, 47)
        priv_obs = x[:, self.base_obs_dim:] # (b, 6)
        prop_latent = self.prop_encoder(priv_obs) # (b, 8)

        
        input = torch.cat([base_obs, prop_latent], dim=1) 
        return self.action_mlp(input)
    def only_obs(self,x):
        return self.action_mlp(x)
    
    def only_latent(self, x):
        '''Given x obs, only returns the prop latent (8 dim)'''
        priv_obs = x[:, self.base_obs_dim:] # (b, 6)
        prop_latent = self.prop_encoder(priv_obs) # (b, 8)
        
        return prop_latent
        

class MLPEncode_wrap(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None,
                 small_init= False, priv_dim = 261):
        super(MLPEncode_wrap, self).__init__()
        
        self.architecture = MLPEncode(shape, actionvation_fn, input_size, output_size, output_activation_fn, small_init, priv_dim)

        self.input_shape = self.architecture.input_shape
        self.output_shape = self.architecture.output_shape

class ActorCriticLatent(nn.Module):
    is_recurrent = False
    def __init__(self,  num_obs,
                        privDim, 
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        output_activation='tanh',
                        init_noise_std=1.0, 
                        **kwargs):
        if kwargs:
            print("ActorCriticLatent.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticLatent, self).__init__()

        activation = get_activation(activation)
        output_activation = get_activation(output_activation)

        self.actor = MLPEncode_wrap(actor_hidden_dims,
                                     activation,
                                     num_obs,
                                     num_actions,
                                     priv_dim = privDim,
                                     output_activation_fn=output_activation)
        


        self.critic = MLPEncode_wrap(critic_hidden_dims,
                                     activation,
                                     num_obs,
                                     1,
                                     priv_dim = privDim,
                                     output_activation_fn=None)
        # Value function

        print(f"Actor MLP: {self.actor.architecture}")
        print(f"Critic MLP: {self.critic.architecture}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor.architecture(observations)
        std = self.std.expand_as(mean)
        
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor.architecture(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic.architecture(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
