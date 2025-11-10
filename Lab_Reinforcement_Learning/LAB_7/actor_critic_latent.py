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
        # super(MLPEncode, self).__init__()
        super().__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        self.base_obs_dim = base_obdim 
        self.priv_obs_dim = priv_dim
        
        self.env_latent_dim = 4 
        
        
        ## TODO(student): Create the Environment Encoder Architecture
        # The environment encoder should be a neural network that maps from a vector of 
        # priviliged environment observations to a latent dimension defined in self.env_latent_dim. 
        # Use hard-coded hidden layer sizes of [256, 128] and the activation function defined in actionvation_fn
        
        self.priv_encoder = nn.Sequential(*[
            nn.Linear(self.priv_obs_dim, 256),
            actionvation_fn,
            nn.Linear(256, 128),
            actionvation_fn,
            nn.Linear(128, self.env_latent_dim)
        ])
        
        ## END TODO
        
        ## TODO(student): Create the Actor / Critic Policy Network
        # Recall that the policy takes as input the base observation and the computed environment latent 
        # from the priv_encoder. Use the "shape" array to define the hidden layer sizes. 
        # Also remember that the output size/activation of this network is dependent on if it's the actor or the critic
        # so update based on output_size and output_activation_fn accordingly
        # creating the action encoder

        enc_layers = []
        in_dim = self.base_obs_dim + self.env_latent_dim
        
        for hidden_dim in shape:
            enc_layers.append(nn.Linear(in_dim, hidden_dim))
            enc_layers.append(actionvation_fn)
            in_dim = hidden_dim
        
        enc_layers.append(nn.Linear(in_dim, output_size))
        if output_activation_fn is not None:
            enc_layers.append(output_activation_fn)
        self.action_mlp = nn.Sequential(*enc_layers)
        
        # END TODO

        self.input_shape = [base_obdim]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self, x):
        '''
        Forward pass of the ActorCriticLatent. 
        First computes the latent environment feature using the priviliged_obs
        Then uses the base_obs and computed feature to compute the policy output
        
        @param x: batched total observation (calculated by x = torch.cat([base_obs, priviliged_obs], dim=1))
        '''
        
        #TODO(student): Implement the forward pass.
        # HINT: self.base_obs_dim may be useful
        
        base_obs = x[:, :self.base_obs_dim]
        priv_obs = x[:, self.base_obs_dim:]
        
        latent = self.priv_encoder(priv_obs)
        combined_input = torch.cat([base_obs, latent], dim=1)
        
        out = self.action_mlp(combined_input)
        
        # END TODO
        return out 
    
    def only_obs(self,x):
        return self.action_mlp(x)
    
    def only_latent(self, x):
        '''Similar to forward(), except just return the computed z_t
        
        @param x: batched total observation (calculated by x = torch.cat([obs, priviliged_obs], dim=1))
        '''
        
        # TODO(student): Implement this function to compute the latent
        
        priv_obs = x[:, self.base_obs_dim:] #(b,6)
        prop_latent = self.priv_encoder(priv_obs) #(b,8)
        
        # END TODO
        
        return prop_latent
        

class MLPEncode_wrap(nn.Module):
    def __init__(self, shape, actionvation_fn, base_obs_dim, output_size, output_activation_fn = None,
                 small_init= False, priv_dim = 261):
        # super(MLPEncode_wrap, self).__init__()
        super().__init__()
        
        self.architecture = MLPEncode(shape, actionvation_fn, base_obs_dim, output_size, output_activation_fn, small_init, priv_dim)

        self.input_shape = self.architecture.input_shape
        self.output_shape = self.architecture.output_shape

class ActorCriticLatent(nn.Module):
    is_recurrent = False
    def __init__(self,  num_obs, # The size of the base observation
                        num_priv_obs, # The size of the privileged environment observation
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        output_activation='tanh',
                        init_noise_std=1.0, 
                        **kwargs):
        if kwargs:
            print("ActorCriticLatent.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        # super(ActorCriticLatent, self).__init__()
        super().__init__()

        activation = get_activation(activation)
        output_activation = get_activation(output_activation)

        # TODO(student): This is where the actor and critic networks are defined
        self.actor = MLPEncode_wrap(actor_hidden_dims,
                                     activation,
                                     base_obs_dim=num_obs,
                                     output_size=num_actions,
                                     priv_dim = num_priv_obs,
                                     output_activation_fn=output_activation)
        
        self.critic = MLPEncode_wrap(critic_hidden_dims,
                                     activation,
                                     base_obs_dim=num_obs,
                                     output_size=1,
                                     priv_dim = num_priv_obs,
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
