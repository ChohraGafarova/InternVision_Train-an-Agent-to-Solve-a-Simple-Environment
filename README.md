# Reinforcement Learning - Q-Learning & DQN

Trained two RL agents from scratch to solve CartPole. One uses tabular Q-Learning, the other uses Deep Q-Networks with neural networks.

## What it does

Teaches an AI to balance a pole on a moving cart by learning from trial and error. No explicit programming of how to balance - the agent figures it out through rewards and penalties.

## Results

- **Q-Learning**: ~150-180 average reward (decent)
- **DQN**: 200+ average reward (solves environment)
- **Target**: 195 average over 100 episodes

DQN wins. Neural networks handle continuous states better than discretization.

## Tech Stack

```
PyTorch
OpenAI Gym / Gymnasium
NumPy
Matplotlib
```

## Installation

```bash
pip install torch gymnasium numpy matplotlib seaborn
```

Or if you have older gym:
```bash
pip install torch gym numpy matplotlib seaborn
```

## Quick Start

```bash
jupyter notebook rl_agent_training.ipynb
```

Run all cells. Takes about 10 minutes total.

## How Reinforcement Learning Works

Traditional ML: "Here's data, learn to predict"
Reinforcement Learning: "Here's an environment, figure it out"

The agent:
1. Observes current state
2. Takes an action
3. Gets a reward (or penalty)
4. Learns which actions lead to better rewards
5. Repeat thousands of times

## CartPole Environment

**Goal**: Keep pole balanced on cart for as long as possible

**State**: 4 values
- Cart position
- Cart velocity  
- Pole angle
- Pole angular velocity

**Actions**: 2 choices
- Push cart left
- Push cart right

**Reward**: +1 for each timestep pole stays up

**Done**: When pole falls past 15° or cart moves off screen

## Q-Learning (Tabular Method)

Maintains a table mapping (state, action) → expected reward

**Update rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**Challenge**: CartPole has continuous states. Had to discretize into bins.

**Result**: Works but loses information. Got ~150-180 average reward.

## Deep Q-Network (DQN)

Neural network approximates Q-values instead of table.

**Key innovations:**
- Experience replay: Store and reuse past experiences
- Target network: Separate network for stability
- Handles continuous states naturally

**Result**: Solves CartPole. Consistent 200+ reward.

## What I learned

**Exploration vs Exploitation is everything**

Started with epsilon = 1.0 (100% random actions). Agent explores like crazy, learns nothing useful.

Slowly decay to epsilon = 0.01 (99% learned actions). Agent exploits what it learned.

Decay too fast? Agent gets stuck in local optimum. Too slow? Takes forever to learn.

**Discretization sucks for continuous spaces**

Q-Learning needs discrete states. I used bins: (6, 6, 12, 12) for the 4 state variables.

Problem: Loses precision. Two very different states might map to same bin.

DQN's neural network handles continuous values naturally. Big advantage.

**Experience replay is genius**

Normal RL: Learn from current experience, throw it away.

DQN: Store experiences in buffer, sample randomly for training.

Why it works:
- Breaks correlation between consecutive samples
- Reuses data (sample efficient)
- Smooths out learning

Added 10x stability to training.

**Target networks prevent moving targets**

Problem: Using same network for prediction and targets creates instability.

Solution: Copy network every N episodes. Use copy for target Q-values.

Stopped the wild oscillations I was seeing initially.

## Training Details

### Q-Learning
- Episodes: 500
- Learning rate: 0.1
- Gamma (discount): 0.99
- Epsilon decay: 0.995
- State bins: (6, 6, 12, 12)
- Time: ~2 minutes

### DQN
- Episodes: 500  
- Learning rate: 0.001
- Gamma: 0.99
- Epsilon decay: 0.995
- Batch size: 64
- Buffer size: 10,000
- Hidden layers: [128, 128]
- Target update: Every 10 episodes
- Time: ~5 minutes

## Common Issues

**"ImportError: cannot import name 'AdamW'"**
Already fixed. Imports from torch.optim now.

**Agent not learning**
- Check epsilon decay (might be too fast/slow)
- Try different learning rate
- Make sure rewards are being collected

**Training unstable**
- Reduce learning rate
- Increase target network update frequency
- Check gradient clipping is working

**Slow convergence**
- Increase learning rate slightly
- Adjust epsilon decay
- Try different network architecture

## Performance Comparison

| Method | Avg Reward | Training Time | Memory | Best For |
|--------|-----------|---------------|---------|----------|
| Q-Learning | 150-180 | 2 min | Low | Discrete states |
| DQN | 200+ | 5 min | Medium | Continuous states |

## When to Use Each

**Q-Learning:**
- Discrete state/action spaces
- Simple grid worlds
- Want to inspect Q-table
- Limited compute

**DQN:**
- Continuous states
- Complex problems
- Visual inputs (Atari)
- Have GPU

## Improvements to Try

Already implemented:
-  Experience replay
-  Target network  
-  Gradient clipping
-  Epsilon decay

Could add:
- Double DQN (reduces overestimation)
- Dueling DQN (separate value/advantage streams)
- Prioritized replay (sample important transitions more)
- Rainbow (combines all improvements)
- Try policy gradient methods (PPO, A3C)

## Other Environments to Try

**Easier:**
- FrozenLake (discrete, simpler)
- Taxi (discrete navigation)

**Harder:**
- MountainCar (sparse rewards, tricky)
- LunarLander (continuous control)
- Atari games (visual inputs)

## Debugging Tips

**Plot everything:**
- Reward curves show if agent is learning
- Epsilon curve shows exploration schedule
- Loss curve (DQN) shows training stability

**Check edge cases:**
- What happens at epsilon=0? (Pure exploitation)
- What's the Q-table/network predicting?
- Are experiences being stored?

**Common mistakes I made:**
- Forgot to decay epsilon (agent never exploited)
- Learning rate too high (unstable training)
- Batch size too small (noisy updates)
- Didn't clip gradients (NaN losses)

## The "Aha" Moment

Watched my DQN fail for 200 episodes. Cart moving randomly, pole falling immediately.

Episode 250: Suddenly it clicked. Pole stayed up for 100 steps.

Episode 350: Consistently getting 200+ steps.

That's reinforcement learning. Looks like it's not working, then suddenly it does. Patience is key.

## Resources That Helped

- Sutton & Barto's RL book (free online)
- DeepMind's DQN paper
- OpenAI Spinning Up guide
- Lots of trial and error

## Next Steps

Planning to:
1. Implement Double DQN
2. Try MountainCar (harder problem)
3. Add dueling architecture
4. Test on Atari games
5. Compare with policy gradient methods
