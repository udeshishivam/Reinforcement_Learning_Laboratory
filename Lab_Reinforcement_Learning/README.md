# Action and Perception Labs: Reinforcement Learning with MuJoCo and MJX

Welcome to the **Action and Perception Labs** repository! This repository showcases a series of labs designed to explore reinforcement learning (RL) concepts using MuJoCo and MJX. Each lab focuses on different aspects of environment simulation, agent control, and RL techniques, highlighting the skills and programming expertise developed throughout the course.

---

## Table of Contents

- [Overview](#overview)
- [Labs and Skills](#labs-and-skills)
- [Acknowledgments](#acknowledgments)

---

## Overview

This repository is part of the **EECS 598: Reinforcement Learning** course. The labs are designed to provide hands-on experience with:
- Simulating environments using MuJoCo and MJX.
- Implementing reinforcement learning algorithms.
- Exploring state and action spaces.
- Modifying environments and agents for custom tasks.

---

## Labs and Skills

### **Lab 2-3: Exploring Environments and Action Spaces**
- **Skills Developed:**
  - Understanding MuJoCo XML-based environment definitions.
  - Manipulating state and action spaces.
  - Rendering simple environments and rollouts on both CPU and GPU.
- **Highlights:**
  - Created a simple MuJoCo world with lights, joints, and geometries.
  - Explored `mjModel`, `mjData`, and `mjRenderer` structures for simulation.

---

### **Lab 4: Terrain Generation and Custom Environments**
- **Skills Developed:**
  - Procedural terrain generation using Python.
  - Customizing environments with obstacles and challenges.
  - Integrating external assets (e.g., URDF, MJX files) into simulations.
- **Highlights:**
  - Designed terrains with varying difficulty levels.
  - Implemented a terrain generator to test agent adaptability.

---

### **Lab 5: Reinforcement Learning with PPO**
- **Skills Developed:**
  - Implementing Proximal Policy Optimization (PPO) for training agents.
  - Visualizing training progress and evaluating policies.
  - Debugging RL algorithms and optimizing hyperparameters.
- **Highlights:**
  - Trained agents to navigate complex terrains.
  - Analyzed training curves and policy performance.

---

### **Lab 6: Domain Randomization**
- **Skills Developed:**
  - Randomizing environment parameters to improve agent robustness.
  - Training agents to adapt to varying conditions.
  - Using checkpoints to save and load trained models.
- **Highlights:**
  - Simulated diverse environments with randomized friction and gravity.
  - Evaluated agent performance under unseen conditions.

---

### **Lab 7: Latent Space Exploration with Actor-Critic Models**
- **Skills Developed:**
  - Implementing actor-critic models for RL.
  - Visualizing latent spaces using t-SNE.
  - Debugging and optimizing neural network architectures.
- **Highlights:**
  - Explored latent representations of policies.
  - Analyzed agent behavior in rough terrains.

---

### **Lab 8: Imitation Learning with DAgger**
- **Skills Developed:**
  - Implementing the Dataset Aggregation (DAgger) algorithm.
  - Training agents using expert demonstrations.
  - Debugging imitation learning pipelines.
- **Highlights:**
  - Trained agents to mimic expert trajectories.
  - Evaluated the trade-offs between exploration and exploitation.

---

### **Lab 9: Introduction to HabitatSim**
- **Skills Developed:**
  - Familiarization with HabitatSim framework.
  - Exploring basic functionalities and configurations.
  - Understanding the potential of HabitatSim for embodied AI tasks.
- **Highlights:**
  - Set up HabitatSim environments.
  - Experimented with simple navigation tasks.

---

### **Lab 10: HabitatLab and Navigation**
- **Skills Developed:**
  - Setting up and configuring HabitatSim environments.
  - Implementing sensor configurations (RGB, Depth) and agent configurations.
  - Navigating environments using NavMesh and shortest path algorithms.
  - Visualizing NavMesh top-down maps and agent trajectories.
  - Implementing trajectory following using GreedyGeodesicFollower.
  - Computing SPL (Success weighted by Path Length) for navigation tasks.
- **Highlights:**
  - Explored HabitatSim for embodied AI tasks.
  - Configured sensors and agents for navigation in 3D environments.
  - Visualized and analyzed agent paths and navigation performance.

---

## Acknowledgments

This repository is part of the **EECS 598: Reinforcement Learning** course. Special thanks to the course instructors and teaching assistants for their guidance and support.