# DQN Algorithms Comparison on Inverted Pendulum-v4 🤖

This repository implements and compares three Deep Q-Network (DQN) variants—**Vanilla DQN**, **Double DQN**, and **Dueling DQN**—on the classic `InvertedPendulum-v4` control problem from the Gymnasium library. The project analyzes learning efficiency, stability, and overall performance in a continuous control task adapted for discrete actions.

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

*(Note: In this specific short-duration experiment, Vanilla DQN performed surprisingly well, suggesting overestimation bias was not a primary hindrance for this task configuration.)*

### Learning Curves
![Performance Graph](performance_plot.png)
<img width="1156" height="701" alt="image" src="https://github.com/user-attachments/assets/92f95e94-b4be-4f0e-938e-404380da10db" />


---

## 🚀 Installation & Usage

### 1. Install Dependencies
```bash
pip install "gymnasium[mujoco]"
pip install torch numpy matplotlib
