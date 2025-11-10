# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Go1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts

from brax import envs

# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain randomization for the Go1 environment."""

import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def rma_domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
      
    # Randomize mass to torso: +U(0, 2.0). (1 dim)
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=0, maxval=2.0)
    new_torso_mass = model.body_mass[TORSO_BODY_ID] + dmass
    body_mass = model.body_mass.at[TORSO_BODY_ID].set(new_torso_mass)
    
    # Randomize COM_xy positiion: +U(-0.05, 0.05). (2 dim)
    rng, key = jax.random.split(rng)
    dpos = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
    dpos = dpos.at[2].set(0.0) # don't randomize z COM
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + dpos
    )
    
    # Randomize Motor stength(joint stiffness) *U(0.9, 1.1) (12 dim)
    rng, key = jax.random.split(rng)
    kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
        key, (model.nu,), minval=0.9, maxval=1.1
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)
      
    # Randomize Floor friction: =U(0.4, 1.0).
    rng, key = jax.random.split(rng)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        jax.random.uniform(key, minval=0.4, maxval=1.0)
    )

    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(12,), minval=-0.05, maxval=0.05)
    )

    return (
        body_mass,
        body_ipos,
        actuator_gainprm,
        actuator_biasprm,
        geom_friction,
        qpos0,
    )

  (
    body_mass,
    body_ipos,
    actuator_gainprm,
    actuator_biasprm,
    geom_friction,
    qpos0,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "body_mass": 0,
      "body_ipos": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
      "geom_friction": 0,
      "qpos0": 0,
  })

  model = model.tree_replace({
      "body_mass": body_mass,
      "body_ipos": body_ipos,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
      "geom_friction": geom_friction,
      "qpos0": qpos0,
  })

  return model, in_axes



def go1_rma_default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.75, # used to be 0.5
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              track_forward_vel_rma=3,
              
              alive=1,
              
              # Penalties 
              lateral_mov_rot_rma=-1,
              work_rma=-0.005,
              gnd_impact_rma= 0, # -0.02, (way too high right now I think)
              smoothness_rma=-0.001,
              action_rma=0,#-0.07,
              joint_speed_rma=-0.002,
              orientation_rma=-10.5,
              z_vel=0,#-2.0,
              feet_slip_rma=0,#-0.8,
          ),
      ),
      impl="jax",
      nconmax=4 * 8192,
      njmax=40,
  )
  
  
from etils import epath
ROOT = epath.Path(__file__).parent


class LocomotionRMAEnv(go1_base.Go1Env):
  """RMA. Track a forward command"""

  def __init__(
      self,
      task: str = "flat_terrain", # can be "rough_terrain"
      config: config_dict.ConfigDict = go1_rma_default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    xml_path = consts.task_to_xml(task).as_posix()
    if task == "rough_terrain":
        my_xml_path = ROOT / "scene_mjx_feetonly_rough_terrain.xml"
        xml_path = my_xml_path
    super().__init__(
        xml_path=xml_path,
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )
    self._feet_body_id = np.array( #TODO(drewskis): This might not work
        [self._mj_model.geom_bodyid[gid] for gid in self._feet_geom_id]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.0, maxval=0.0)
    )
    

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])
    
    f0 = data.cfrc_ext[self._feet_body_id, :3]

    info = {
        "rng": rng,
        "training_step": -1,
        "last_f_gnd": f0,
        "last_torque": jp.zeros(self.mjx_model.nu),
        "last_last_torque": jp.zeros(self.mjx_model.nu),
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "last_contact": jp.zeros(4, dtype=bool),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    # assert(state.info["training_step"] != -1)
    k_t = (0.03) ** jp.power(0.997, state.info["training_step"])
    # k_t = 1
    
    # jax.debug.print("This is the training step, {}", state.info["training_step"])
    
    rewards = self._get_reward(data, action, state.info, state.metrics, done, contact)
    scaled_rewards = {}
    for key, val in rewards.items():
        if key == "track_forward_vel_rma" or key == "lateral_mov_rot_rma" or key == "alive" or key=="orientation_rma": #or key == "work_rma":
            scaled_rewards[key] = val*self._config.reward_config.scales[key]
        else:
            # apply the curriculum
            scaled_rewards[key] = k_t*val*self._config.reward_config.scales[key]
            
    
    # rewards = {
    #     k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    # }
    reward = sum(scaled_rewards.values()) * self.dt
    # reward = jp.clip(reward, 0.0, 10000.0) # Done incentivise killing yourself immediately
    # reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0) #TODO(drewskis): should this be here? 
    
    
    state.info["last_f_gnd"] = data.cfrc_ext[self._feet_body_id, :3]
    state.info["last_last_torque"] = state.info["last_torque"]
    state.info["last_torque"] = data.actuator_force
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_contact"] = contact
    
    for k, v in scaled_rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    return fall_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    

    gravity = self.get_gravity(data)
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]    
    roll = jp.arctan2(gravity[1], gravity[2])
    pitch = jp.arctan2(-gravity[0], jp.sqrt(gravity[1]**2 + gravity[2]**2))
    
    state = jp.hstack([
        joint_angles - self._default_pose, # (12, )
        joint_vel, #(12, )
        roll,  # (1, )
        pitch, # (1, )
        info["last_contact"],  # (4, )
        self.get_gyro(data),
        self.get_gravity(data),
        self.get_local_linvel(data),
        # vvv Also append last action for convenience vvv #
        info["last_act"] # (12, )
    ])

    FLOOR_GEOM_ID = 0
    TORSO_BODY_ID = 1
    # local_terrain_height = self.get_feet_pos(data) 
    # print("This is the shape of the local_terrain_height", local_terrain_height.shape)
    # exit(0)
    privileged_state = jp.hstack([
        self.mjx_model.body_mass.at[TORSO_BODY_ID].get(), # 1 (Body mass) 
        self.mjx_model.body_ipos.at[TORSO_BODY_ID].get(), # 3 (Body COM)
        self.mjx_model.actuator_gainprm.at[:, 0].get(), # 12 (approx "motor stength")
        self.mjx_model.geom_friction.at[FLOOR_GEOM_ID, 0].get(),  # 1 (Surface friction)
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        # Tracking
        "track_forward_vel_rma": self._reward_forward_vel_rma(self.get_local_linvel(data)),
        
        "alive": self._reward_alive(done),
        
        # Penalties
        "lateral_mov_rot_rma": self._cost_lateral_mov_rot_rma(self.get_local_linvel(data), self.get_gyro(data)),
        "work_rma": self._cost_work_rma(data.qvel[6:], data.actuator_force),
        "gnd_impact_rma": self._cost_gnd_impact_rma(data.cfrc_ext[self._feet_body_id, :3], info["last_f_gnd"]),
        "smoothness_rma": self._cost_smoothness_rma(data.actuator_force, info["last_torque"], info["last_last_torque"]),
        "action_rma": self._cost_action_rma(action),
        "joint_speed_rma": self._cost_joint_speed_rma(data.qvel[6:]),
        "orientation_rma": self._cost_orientation_rma(self.get_upvector(data)),
        "z_vel": self._cost_z_vel_rma(self.get_local_linvel(data)),
        "feet_slip_rma": self._cost_feet_slip_rma(data, contact),
        # "termination": self._cost_termination(done), # TODO(drewskis): Should I care about this?
    }

  # -------- RMA Tracking Rewards -------- #
  def _reward_forward_vel_rma(self, 
                              local_vel: jax.Array) -> jax.Array:
      alive_bonus = 0
      return jp.minimum(0.35, local_vel[0]) + alive_bonus
  
  # --------- RMA Penalty Costs --------- #
  def _cost_lateral_mov_rot_rma(self,
                                local_vel: jax.Array,
                                ang_vel: jax.Array) -> jax.Array:
      return jp.square(local_vel[1]) + jp.square(ang_vel[2])
  
  def _cost_work_rma(self,
                     qvel: jax.Array,
                     torques: jax.Array,) -> jax.Array:
    #   return jp.sum(jp.abs(qvel) * jp.abs(torques))
      return jp.abs(jp.sum(qvel * torques))
  
  def _cost_gnd_impact_rma(self,
                           f_gnd: jax.Array,
                           last_f_gnd: jax.Array) -> jax.Array:
      f_gnd_per_foot = jp.linalg.norm(f_gnd, axis=1, keepdims=True)
      f_gnd_per_foot_last = jp.linalg.norm(last_f_gnd, axis=1, keepdims=True)
    
      gnd_impact_cost = jp.sum(jp.square(f_gnd_per_foot-f_gnd_per_foot_last))
      
      return gnd_impact_cost
  
  def _cost_smoothness_rma(self, 
                           torque: jax.Array, 
                           last_torque: jax.Array, 
                           last_last_torque: jax.Array) -> jax.Array:
    del last_last_torque  # Unused.
    return jp.sum(jp.square(torque - last_torque))
  
  def _cost_action_rma(self, 
                       act: jax.Array) -> jax.Array:
      return jp.sum(jp.square(act))
  
  def _cost_joint_speed_rma(self,
                          qvel: jax.Array) -> jax.Array:
      return jp.sum(jp.square(qvel))
  
  
  def _cost_orientation_rma(self, 
                            torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))
  
  def _cost_z_vel_rma(self,
                        local_vel: jax.Array) -> jax.Array:
      return jp.square(local_vel[2])
  
  def _cost_feet_slip_rma(self, 
                          data: mjx.Data, 
                          contact: jax.Array) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact)

  def _reward_alive(self, done):
     return 1.0 - done.astype(jp.float32)


envs.register_environment('rma_go1_locomote', LocomotionRMAEnv)