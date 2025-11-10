from rsl_rl.env import VecEnv
import torch.utils.dlpack as tpack
from jax.dlpack import from_dlpack
from mujoco_playground._src import wrapper
import torch 
from collections import deque
import jax
import functools
import numpy as np

def _jax_to_torch(tensor):
    '''Convert a jax array to a torch tensor'''
    tensor = tpack.from_dlpack(tensor)
    return tensor

def _torch_to_jax(tensor):
    '''Convert a torch tensor to a jax array'''
    tensor = from_dlpack(tensor)
    return tensor

class CSE598RSLWrapper(VecEnv):
  """Wrapper for Brax environments that interop with torch."""

  def __init__(
      self,
      env,
      num_actors,
      seed,
      episode_length,
      action_repeat,
      randomization_fn=None,
      render_callback=None,
  ):
    

    self.seed = seed
    self.batch_size = num_actors
    self.num_envs = num_actors
    
    self.device="cuda:0"

    self.key = jax.random.PRNGKey(self.seed)

    # split key into two for reset and randomization
    key_reset, key_randomization = jax.random.split(self.key)

    self.key_reset = jax.random.split(key_reset, self.batch_size)

    if randomization_fn is not None:
      randomization_rng = jax.random.split(key_randomization, self.batch_size)
      v_randomization_fn = functools.partial(
          randomization_fn, rng=randomization_rng
      )
    else:
      v_randomization_fn = None

    self.env = wrapper.wrap_for_brax_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    self.render_callback = render_callback

    self.asymmetric_obs = False
    obs_shape = self.env.env.unwrapped.observation_size
    print(f"obs_shape: {obs_shape}")

    if isinstance(obs_shape, dict):
      print("Asymmetric observation space")
      self.asymmetric_obs = True
      self.num_obs = obs_shape["state"]
      self.num_privileged_obs = obs_shape["privileged_state"]
    else:
      self.num_obs = obs_shape
      self.num_privileged_obs = None

    self.num_actions = self.env.env.unwrapped.action_size

    self.max_episode_length = episode_length

    # todo -- specific to leap environment
    self.success_queue = deque(maxlen=100)

    print("JITing reset and step")
    self.reset_fn = jax.jit(self.env.reset)
    self.step_fn = jax.jit(self.env.step)
    print("Done JITing reset and step")
    self.env_state = None

  def step(self, action: torch.Tensor):
    '''Wrapper of the step function, taking a torch.Tensor action input'''
    
    # TODO(student): The implementation of the below step() logic is incomplete (there are many type errors). 
    # Consider at each step whether a variable should be stored as a torch.Tensor or a jax.Array
    # HINT: the functions "_jax_to_torch" and "_torch_to_jax" to convert between the two types.
    
    # Clip action (torch operation)
    action = torch.clip(action, -1.0, 1.0)
    
    # Convert torch action to JAX for environment step
    action_jax = _torch_to_jax(action)
    
    # Step environment (JAX operation)
    self.env_state = self.step_fn(self.env_state, action_jax)
    
    # Extract and convert observations
    critic_obs = None
    if self.asymmetric_obs:
      obs = _jax_to_torch(self.env_state.obs["state"])
      critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
    else:
      obs = _jax_to_torch(self.env_state.obs)
    
    # Convert reward and done to torch
    reward = _jax_to_torch(self.env_state.reward)
    done = _jax_to_torch(self.env_state.done)
    
    # Info stays as JAX dict for now
    info = self.env_state.info
    truncation = _jax_to_torch(info["truncation"])
    # END TODO(student)

    info_ret = {
        "time_outs": truncation,
        "observations": {"critic": critic_obs},
        "log": {},
    }

    if "last_episode_success_count" in info:
      last_episode_success_count = (
          _jax_to_torch(info["last_episode_success_count"])[done > 0]
          .float()
          .tolist()
      )
      if len(last_episode_success_count) > 0:
        self.success_queue.extend(last_episode_success_count)
      info_ret["log"]["last_episode_success_count"] = np.mean(
          self.success_queue
      )

    for k, v in self.env_state.metrics.items():
      if k not in info_ret["log"]:
        info_ret["log"][k] = _jax_to_torch(v).float().mean().item()

    return obs, reward, done, info_ret

  def reset(self):
    # todo add random init like in collab examples?
    self.env_state = self.reset_fn(self.key_reset)

    if self.asymmetric_obs:
      obs = _jax_to_torch(self.env_state.obs["state"])
      # critic_obs = jax_to_torch(self.env_state.obs["privileged_state"])
    else:
      obs = _jax_to_torch(self.env_state.obs)
    return obs

  def reset_with_critic_obs(self):
    self.env_state = self.reset_fn(self.key_reset)
    obs = _jax_to_torch(self.env_state.obs["state"])
    critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
    return obs, critic_obs

  def get_observations(self):
    if self.asymmetric_obs:
      obs, critic_obs = self.reset_with_critic_obs()
      return obs, {"observations": {"critic": critic_obs}}
    else:
      return self.reset(), {"observations": {}}

  def render(self, mode="human"):  # pylint: disable=unused-argument
    if self.render_callback is not None:
      self.render_callback(self.env.env.env, self.env_state)
    else:
      raise ValueError("No render callback specified")

  def get_number_of_agents(self):
    return 1

  def get_env_info(self):
    info = {}
    info["action_space"] = self.action_space  # pytype: disable=attribute-error
    info["observation_space"] = (
        self.observation_space  # pytype: disable=attribute-error
    )
    return info