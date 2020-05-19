import gym
import random
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('NChain-v0')

# Hyperparameters
episodes = 100000               # Number of episodes to run
alpha = 0.01                    # Learning rate
gamma = 0.99                    # Discount (Determines how important future rewards are.)
epsilon = 1                     # Ratio between exploration and exploitation
min_epsilon = 0.00              # Minimum epsilon
epsilon_decaying_start = 1      # Episode to start epsilon decaying
epsilon_decaying_end = 90000    # Episode to end epsilon decaying
epsilon_decay_value = epsilon / (epsilon_decaying_end - epsilon_decaying_start) # Epsilon decay value per episode

# Statistics
episode_rewards = []
statistics = {'episode': [], 'avg': [], 'max': [], 'min': []}

q_table = np.random.uniform(low=0, high=1, size=(env.observation_space.n, env.action_space.n))


for episode in range(episodes):
    episode_reward = 0
    # Reset the environment state
    state = env.reset()
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            # Explore the environment
            action = env.action_space.sample()
        else:
            # Exploit the environment
            action = np.argmax(q_table[state])

        # Performe the selected action
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # Maximum possible Q value in the next state
        future_max_q = np.max(q_table[new_state])
        # Current Q value (for current state and performed action)
        current_q = q_table[state, action]
        # Calculate the new Q value for the current state and action
        new_q_value = (1 - alpha) * current_q + alpha * (reward + gamma * future_max_q)
        # Update the Q table with the new Q value
        q_table[state, action] = new_q_value

        # Update the state with the new state
        state = new_state

    episode_rewards.append(episode_reward)

    # Perform epsilon decaying
    if epsilon_decaying_end >= episode >= epsilon_decaying_start:
        if epsilon - epsilon_decay_value < 0:
            epsilon = min_epsilon
        else:
            epsilon -= epsilon_decay_value

    if episode % 100 == 0:
        average_reward = sum(episode_rewards[-100:]) / 100
        statistics['episode'].append(episode)
        statistics['avg'].append(average_reward)
        statistics['max'].append(max(episode_rewards[-100:]))
        statistics['min'].append(min(episode_rewards[-100:]))
        # Print some information so we know the program is alive
        print('Episode: {}, Average reward: {}, Current epsilon: {}'.format(episode, average_reward, epsilon))

print('Training finished.')


# Plot training statistics
plt.plot(statistics['episode'], statistics['avg'], label='Average rewards')
plt.plot(statistics['episode'], statistics['max'], label='Max rewards')
plt.plot(statistics['episode'], statistics['min'], label='Min rewards')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# Get a final 100 episodes average reward
state = env.reset()
rewards = 0
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _ = env.step(action)
        rewards += reward

print('100 episodes average reward: {}'.format(rewards / 100))