import gym
from numpy import zeros, argmax, max as npmax, loadtxt, savetxt
from random import uniform
from time import sleep


class QlearningAgent:
    def __init__(self, alpha=0.95, gamma=0.5, epochs=1000):
        self.env = gym.make("Taxi-v3")
        # Hyperparams
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = 1.0  # Start with exploration rate of 1.0
        self.epsilon_min = 0.0001  # Set a minimum exploration rate
        self.epsilon_decay = 0.999  # Set the decay rate
        self.max_steps = 100
        self.epochs = epochs
        # Initialize Q-Table
        self.q_table = zeros([self.env.observation_space.n, self.env.action_space.n])

    def choose_action(self, state):
        if uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Explore action space
        else:
            action = argmax(self.q_table[state])  # Exploit learned values
        return action

    def step(self, action):
        new_state, reward, done, _, __ = self.env.step(action)
        return new_state, reward, done

    def update_q(self, state, new_state, reward, action):
        old_value = self.q_table[state, action]
        next_max = npmax(self.q_table[new_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def load_agent(self):
        self.q_table = loadtxt('qtable.txt', dtype=float)

    def save_agent(self):
        savetxt('qtable.txt', self.q_table)

    def train(self):
        print('Training for {} epochs...'.format(self.epochs))
        for epoch in range(1, self.epochs + 1):
            state = self.env.reset()[0]
            done = False
            steps = 0

            while not done and steps < self.max_steps:
                # choose action
                action = self.choose_action(state)

                # make a step
                new_state, reward, done = self.step(action)

                # update q table
                self.update_q(state, new_state, reward, action)

                # Update exploration rate
                self.epsilon = npmax(self.epsilon_min, int(self.epsilon_decay * self.epsilon))

                # Update state, steps.
                state = new_state
                steps += 1

        self.save_agent()
        self.env.close()

    def test(self):
        self.env = gym.make("Taxi-v3", render_mode="human")

        # Load Q-table - best agent.
        self.load_agent()

        print('Testing Q-learning...')
        for epoch in range(1, 4):
            state = self.env.reset()[0]
            done = False
            steps, penalties, avg_reward = 0, 0, 0

            while not done and steps < self.max_steps:
                sleep(0.3)
                action = argmax(self.q_table[state])  # choose best action due to qtable values
                new_state, reward, done, _, __ = self.env.step(action)  # do step

                compass = {0: "move(↓)", 1: "move(↑)", 2: "move(->)", 3: "move(<-)", 4: "PickUp()", 5: "DropOff()"}
                print("{} - {} -> {}; reward: {}".format(
                    state, compass[action], new_state, reward))

                if reward == -10:
                    penalties += 1
                state = new_state
                avg_reward += reward
                steps += 1

            print('Epoch {} is {}: steps={}, penalties={}, avg_reward={:.2f} \n\n'.format(epoch, done, steps,
                                                                                          penalties,
                                                                                          avg_reward / steps))
        self.env.close()


class SarsaAgent(QlearningAgent):
    def __init__(self, alpha=0.65, gamma=0.6, epochs=1000):
        super().__init__(alpha, gamma, epochs)

    def update_q(self, state, new_state, reward, action, next_action=None):
        old_value = self.q_table[state, action]
        next_value = self.q_table[new_state, next_action]
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_value)
        self.q_table[state, action] = new_value

    def load_agent(self):
        self.q_table = loadtxt('sarsa_qtable.txt', dtype=float)

    def save_agent(self):
        savetxt('sarsa_qtable.txt', self.q_table)

    def train(self):
        print('Training for {} epochs...'.format(self.epochs))
        for epoch in range(1, self.epochs + 1):
            state = self.env.reset()[0]
            done = False
            steps = 0

            # choose action
            action = self.choose_action(state)

            while not done and steps < self.max_steps:
                # make a step
                new_state, reward, done = self.step(action)

                # Choose next action
                if uniform(0, 1) < self.epsilon:
                    next_action = self.env.action_space.sample()  # Explore action space
                else:
                    next_action = argmax(self.q_table[new_state])  # Exploit learned values

                # Updating Q-table
                self.update_q(state, new_state, reward, action, next_action)

                # Update exploration rate
                self.epsilon = npmax(self.epsilon_min, int(self.epsilon_decay * self.epsilon))

                # Update state, steps. Avg_steps, avg_reward for reference table
                state = new_state
                action = next_action
                steps += 1

        self.save_agent()
        self.env.close()


class ValueAgent(QlearningAgent):
    def __init__(self, alpha=0.85, gamma=0.75, epochs=10000):
        super().__init__(alpha, gamma, epochs)
        self.value_table = zeros(self.env.observation_space.n)

    def load_agent(self):
        self.q_table = loadtxt('value_qtable.txt', dtype=float)

    def save_agent(self):
        savetxt('value_qtable.txt', self.q_table)

    def train(self):
        print('Training...')
        for epoch in range(self.epochs):
            delta = 0
            for state in range(self.env.observation_space.n):
                values = []
                for action in range(self.env.action_space.n):
                    prob, next_state, reward, done = self.env.P[state][action][0]
                    value = prob * (reward + self.gamma * self.value_table[next_state])
                    values.append(value)

                max_value = max(values)
                delta = max(delta, abs(self.value_table[state] - max_value))
                self.value_table[state] = max_value

            if epoch % 100 == 0:
                print('Epoch = {}, delta = {}'.format(epoch, delta))

            # Stop iteration if convergence is achieved
            if delta < self.alpha:
                print("Converged after %d iterations." % epoch)
                break

        # output the deterministic policy
        for state in range(self.env.observation_space.n):
            values = []
            for action in range(self.env.action_space.n):
                prob, next_state, reward, done = self.env.P[state][action][0]
                value = prob * (reward + self.gamma * self.value_table[next_state])
                self.q_table[state][action]=value

        self.save_agent()
        self.env.close()
