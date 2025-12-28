# DQN Algorithms Comparison on Inverted Pendulum-v4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m7KI03UVxwEN3F8xSwbS2f4NPsaimrWS?usp=sharing)

This repository implements and compares three Deep Q-Network (DQN) variants—**Vanilla DQN**, **Double DQN**, and **Dueling DQN**—on the classic `InvertedPendulum-v4` control problem. The project analyzes learning efficiency, stability, and overall performance in a continuous control task adapted for discrete actions.

## 📖 Project Overview

The Inverted Pendulum is a foundational problem in control theory where an agent must apply horizontal forces to a cart to keep a hinged pole upright. The system is inherently unstable; without active control, the pendulum will quickly fall.

### The Environment
* **Environment:** `InvertedPendulum-v4` (Mujoco)
* **State Space:** 4 (Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity)
* **Action Space:** Continuous inherently, but discretized into **3 actions** for DQN compatibility.
* **Goal:** Maximize cumulative reward (+1 for every timestep the pole remains upright).

---

## 🧠 Algorithms Implemented

1.  **Vanilla DQN:** The foundational algorithm using a Q-network for action-value approximation and a separate target network.
2.  **Double DQN (DDQN):** Mitigates overestimation bias by decoupling action selection from action evaluation.
3.  **Dueling DQN:** Separates the estimation of state-value $V(s)$ and advantage $A(s,a)$ functions to better learn which states are valuable.

---

## ⚙️ Hyperparameters

The following configuration was used consistently across all three agents to ensure a fair comparison:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `LEARNING_RATE` | **1e-3** | Adam Optimizer learning rate |
| `BUFFER_SIZE` | **100,000** | Experience Replay buffer capacity |
| `BATCH_SIZE` | **64** | Minibatch size for training |
| `GAMMA` | **0.99** | Discount factor for future rewards |
| `EPSILON_START` | **1.0** | Initial exploration rate (100% random) |
| `EPSILON_END` | **0.01** | Minimum exploration rate (1% random) |
| `EPSILON_DECAY` | **10,000** | Rate of exponential decay for epsilon |
| `TARGET_UPDATE_FREQ`| **500** | Steps between target network updates |
| `NUM_EPISODES` | **500** | Total training episodes per agent |
| `MAX_TIMESTEPS` | **500** | Maximum steps allowed per episode |

---

## 📊 Results

The agents were trained for 500 episodes each. Performance was measured by total reward per episode, visualized using a 100-episode Simple Moving Average (SMA).

### Performance Summary (Final 50 Episodes)

| Algorithm | Avg. Reward |
| :--- | :--- |
| **Vanilla DQN** | **7.56** |
| **Double DQN** | 7.00 |
| **Dueling DQN** | 6.70 |

### Learning Curves
The plot below shows the 100-episode moving average of rewards. All agents show learning progress, with Vanilla DQN showing slightly higher final performance in this specific run.

![DQN Comparison Plot](dqn_comparison.png)
*(Note: Ensure you upload the 'dqn_comparison.png' file to your repository for this image to appear. You can generate this by running the last cell of the notebook.)*

---

## 🚀 Installation & Usage

### 1. Install Dependencies
```bash
pip install "gymnasium[mujoco]"
pip install torch numpy matplotlib
