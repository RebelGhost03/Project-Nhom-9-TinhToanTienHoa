import numpy as np
from collections import defaultdict
from environment import CliffEnvironment
import matplotlib.pyplot as plt


def epsilon_greedy_policy(Q, state, nA, epsilon):
    action = np.argmax(Q[state])
    A = np.ones(nA, dtype=np.float32) * epsilon / nA
    A[action] += 1 - epsilon
    return A


def greedy_policy(Q, state):
    best_action = np.argmax(Q[state])
    return best_action


def print_policy(Q):
    env = CliffEnvironment()
    result = ""
    for i in range(env.height):
        line = ""
        for j in range(env.width):
            action = np.argmax(Q[(j, i)])  # find the action to max Q value
            if action == 0:
                line += "↑ "
            elif action == 1:
                line += "↓ "
            elif action == 2:
                line += "← "
            else:
                line += "→ "
        result = line + "\n" + result
    print(result)


def Qlearning(alpha=0.1, episode_num=1000, discount_factor=1.0, epsilon=0.3):
    env = CliffEnvironment()
    Q = defaultdict(lambda: np.zeros(env.nA))
    rewards = []
    for i in range(episode_num):
        env._reset()
        cur_state, done = env.observation()
        sum_reward = 0.0

        while not done:
            prob = epsilon_greedy_policy(Q, cur_state, env.nA,epsilon)
            action = np.random.choice(np.arange(env.nA), p=prob)  # re-select action per round
            next_state, reward, done = env._step(action)  # apply action to find next state
            if done:
                Q[cur_state][action] = Q[cur_state][action] + alpha * (
                        reward + discount_factor * 0.0 - Q[cur_state][action])
                break
            else:
                # use next_action only to calculate Q，but the actual next action to perform is not that
                next_action = greedy_policy(Q, next_state)
                Q[cur_state][action] = Q[cur_state][action] + alpha * (
                        reward + discount_factor * Q[next_state][next_action] - Q[cur_state][action])
                cur_state = next_state
            sum_reward += reward
        rewards.append(sum_reward)
    return Q, rewards


def plot(x, y,labels):
    size = len(x)
    x = [x[i] for i in range(size) if i % 50 == 0]
    y = [y[i] for i in range(size) if i % 50 == 0]
    plt.plot(x, y,label=labels)    

candidate_epsilon=[0,0.01,0.1,0.2,0.3,0.5]
for epsilon in candidate_epsilon:
    print('epsilon: ',epsilon)
    Q, rewards = Qlearning(episode_num=1000,epsilon=epsilon)
    average_rewards = []
    for i in range(10):
        _, rewards = Qlearning(episode_num=1000,epsilon=epsilon)
        if len(average_rewards) == 0:
            average_rewards = np.array(rewards)
        else:
            average_rewards = average_rewards + np.array(rewards)
    average_rewards = average_rewards * 1.0 / 10
    plot(range(1000), average_rewards,labels='epsilon='+str(epsilon))
    print_policy(Q)
    
plt.xlabel('episode')
plt.ylabel('reward')
plt.ylim(-800,0)
plt.legend()
plt.show()