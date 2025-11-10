#@title Barkour vb Quadruped Env
from etils import epath
from typing import Any, Dict, Sequence, Tuple, Union, List
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from custom_render import render_array_with_foot_trace 

UNITREEGO2_ROOT_PATH = epath.Path('unitree_go2')

class UnitreeGo2Env(PipelineEnv):
  """Custom Environment for UnitreeGo2 in MJX."""

  def __init__(
      self,
      obs_noise: float = 0.05,
      action_scale: float = 0.3,
      kick_vel: float = 0.0,
      scene_file: str = 'scene_mjx.xml',
      keyframe_name="home",
      randomize_initial_pos=False,
      render_foot_cursor=False,
      **kwargs,
  ):
    self.randomize_initial_pos = randomize_initial_pos
    path = UNITREEGO2_ROOT_PATH / scene_file
    sys = mjcf.load(path.as_posix())
    self._dt = 0.02  # this environment is 50 fps
    sys = sys.tree_replace({'opt.timestep': 0.004})

    # override menagerie params for smoother policy
    sys = sys.replace(
        dof_damping=sys.dof_damping.at[6:].set(0.5239),
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
        actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
    )

    n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
    super().__init__(sys, backend='mjx', n_frames=n_frames)

    self._torso_idx = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base'
    )
    
    self.render_foot_cursor = render_foot_cursor
    
    
    self._action_scale = action_scale
    self._obs_noise = obs_noise
    self._kick_vel = kick_vel
    
    
    
    self._init_q = jp.array(sys.mj_model.keyframe(keyframe_name).qpos)
    self._default_pose = sys.mj_model.keyframe(keyframe_name).qpos[7:]
    
    self.lowers = jp.array([-0.9472, -1.4, -2.6227] * 4)
    self.uppers = jp.array([0.9472, 2.5, -0.84776] * 4)
    
    self._foot_radius = 0.0175
    self._nv = sys.nv

  def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
    rng, key = jax.random.split(rng)
    
    
    # TODO(student): Modify the below so that when self.randomize_initial_pos the environment randomizes the position
    # of the robot to some degree. Do some sleuthing to figure out exactly what quantities need 
    # to be randomized and at which indices. The degree to which it is randomized is up to you. 
    # Note: take a peak at the unitree_go2/go2_mjx.xml file to see what these "keyframes" are refering to!
    # Note: use jax.random functions to generate random numbers
    reset_q = self._init_q #the qposition of keyframe with key=home
    if self.randomize_initial_pos:
      rng, subkey1, subkey2 = jax.random.split(rng, 3)
      noise_xy = 0.05 * jax.random.normal(subkey1, (2,))
      updated_q = reset_q.at[0:2].add(noise_xy)
      reset_q = updated_q

      yaw_noise = jax.random.uniform(key, (1,), minval=-jp.pi/4, maxval=jp.pi/4)
      current_quat = reset_q[3:7]
      yaw_quat = jp.array([jp.cos(yaw_noise[0]/2), 0, 0, jp.sin(yaw_noise[0]/2)])

      # Multiply quaternions to combine rotations
      new_quat = math.quat_mul(current_quat, yaw_quat)
      reset_q = reset_q.at[3:7].set(new_quat)
      pass

    pipeline_state = self.pipeline_init(reset_q, jp.zeros(self._nv))

    state_info = {
        'rng': rng,
        'last_act': jp.zeros(12),
        'last_vel': jp.zeros(12),
        'kick': jp.array([0.0, 0.0]),
        'step': 0,
    }

    obs_history = jp.zeros(15 * 31)  # store 15 steps of history
    obs = self._get_obs(pipeline_state, state_info, obs_history)
    reward, done = jp.zeros(2)
    metrics = {'total_dist': 0.0}
    state = State(pipeline_state, obs, reward, done, metrics, state_info)  # pytype: disable=wrong-arg-types
    return state



  def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
    rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)



    # kick
    push_interval = 100
    kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
    # TODO(student): Impliment a "kick" to the robot every push_interval steps along the xy-plane.
    # you may use kick_theta to determine the direction of the kick.
    # Be sure to also add the final "kick" to the "state" dictionary. 
    # HINTS: 
    #   1. At a high level, a common way to impliment a 'kick" is 
    #       by instantaneously setting the velocity of the base to some non-zero 
    #       quantity.
    #   2. You can use the "pipeline_state" variable to get the velcoity of the robot (and motors)
    #   3. use self._kick_vel to determine velocity
    #   4. state.info['step'] gives the step of the sim, 
    #   5. Use state = state.tree_replace({'pipeline_state.qvel': qvel})
    kick = jp.array([0, 0]) # kick vel in x and y direction
    kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
    # impliment here
    kick = self._kick_vel * kick
    
    qvel = state.pipeline_state.qvel
    qvel_kick = qvel.at[0].set(kick[0]).at[1].set(kick[1])

    # condition: apply kick only at multiples of push_interval
    qvel = jax.lax.cond(
        state.info['step'] % push_interval == 0,
        lambda _: qvel_kick,   # true branch
        lambda _: qvel,        # false branch
        operand=None
    )
    state = state.tree_replace({'pipeline_state.qvel': qvel})

    # physics step
    motor_targets = self._default_pose + action * self._action_scale
    motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
    pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
    x, xd = pipeline_state.x, pipeline_state.xd

    # observation data
    obs = self._get_obs(pipeline_state, state.info, state.obs)
    joint_angles = pipeline_state.q[7:]
    joint_vel = pipeline_state.qd[6:]

    # done if joint limits are reached or robot is falling
    up = jp.array([0.0, 0.0, 1.0])
    done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
    done |= jp.any(joint_angles < self.lowers)
    done |= jp.any(joint_angles > self.uppers)
    done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

    
    # state management
    # TODO(student): Add the "kick" to the state
    state.info['last_act'] = action
    state.info['last_vel'] = joint_vel
    state.info['step'] += 1
    state.info['rng'] = rng
    state.info['kick'] = kick

    # reset the step counter when done
    state.info['step'] = jp.where(
        done | (state.info['step'] > 500), 0, state.info['step']
    )

    # log total displacement as a proxy metric
    state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
    # state.metrics.update(state.info['rewards'])

    done = jp.float32(done)
    reward = jp.float32(0)
    state = state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
    return state

  def _get_obs(
      self,
      pipeline_state: base.State,
      state_info: dict[str, Any],
      obs_history: jax.Array,
  ) -> jax.Array:
    inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
    local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

    obs = jp.concatenate([
        jp.array([local_rpyrate[2]]) * 0.25,                 # yaw rate
        math.rotate(jp.array([0, 0, -1]), inv_torso_rot),    # projected gravity
        pipeline_state.q[7:] - self._default_pose,           # motor angles
        state_info['last_act'],                              # last action
    ])

    # clip, noise
    obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
        state_info['rng'], obs.shape, minval=-1, maxval=1
    )
    # stack observations through time
    obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)

    return obs

  def render(
      self, trajectory: List[base.State], camera: str | None = None,
      width: int = 240, height: int = 320,
  ) -> Sequence[np.ndarray]:
    camera = camera or 'track'
        
    if self.render_foot_cursor:
        # Overwritten render function that traces a path where the FR foot has been
        return render_array_with_foot_trace(self.sys, trajectory, height, width, camera)
    else:
        return super().render(trajectory, camera=camera, width=width, height=height)


