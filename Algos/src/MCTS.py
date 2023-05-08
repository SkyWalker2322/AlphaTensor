from typing import *
from tqdm import tqdm
import numpy as np

from NeuralNet import NeuralNet
from Environment import State, Environment


class MCTS:

    def __init__(
            self,
            environment: Environment,
            nnet: NeuralNet,
            num_simulations: int,
            max_num_steps: int,
            cpuct: float = 1.0,
    ):
        self.environment = environment
        self.nnet = nnet
        self.num_simulations = num_simulations
        self.max_num_steps = max_num_steps
        self.cpuct = cpuct

        # transposition tables
        self.N: Dict[str, np.ndarray] = {}  # No of time child with action a visited N(s,a)
        self.W: Dict[str, np.ndarray] = {}  # 
        self.Q: Dict[str, np.ndarray] = {}  # Q(s,a)
        self.P: Dict[str, np.ndarray] = {}  # initial policy for starting state s (returned by NN)

        self.states = []
        self.actions = []
        self.cumulative_rewards = []

    def policy(self, state: State) -> np.ndarray:
        for _ in range(self.num_simulations):
            self.Run_Expansion(state)

        s = state.to_str()

        return self.N[s] / np.sum(self.N[s]) 
    
    def back_propagate(self,cumulative_reward):
        for s, a, r in zip(self.states, self.actions, self.cumulative_rewards):
            self.N[s][a] = self.N[s][a] + 1
            self.W[s][a] = self.W[s][a] + cumulative_reward - r
            self.Q[s][a] = self.W[s][a] / self.N[s][a]

    def Run_Expansion(self, state):
        assert not self.environment.is_terminal(state)

        self.states.clear()
        self.actions.clear()
        self.cumulative_rewards.clear()
        cumulative_reward = 0
        num_steps = 0
        while True:
            s = state.to_str()

            # Simulation using NN 
            if s not in self.P:
                # leaf state
                p, v = self.nnet.predict(state)
                cumulative_reward += v  #adding the value of selected child
                self.P[s] = p
                #initialising the values of N(s), Q(s), W(s) of unexplored state s
                self.N[s] = np.zeros(len(p), dtype=int)     
                self.W[s] = np.zeros(len(p), dtype=float)
                self.Q[s] = np.zeros(len(p), dtype=float)
                break

            #Selection
            # pick the action with the highest upper confidence bound
            a = np.argmax(self.Q[s] + self.cpuct * self.P[s] * np.sqrt(np.sum(self.N[s])) / (1 + self.N[s]))

            self.states.append(s)
            self.actions.append(a)
            self.cumulative_rewards.append(cumulative_reward)  # current cumulative reward, before applying action a

            cumulative_reward += self.environment.get_intermediate_reward(state, a)
            state = self.environment.get_next_state(state, a) #expansion
            num_steps += 1

            #expanding till max_num_step since primitive model can span very large unwanted tree
            if num_steps >= self.max_num_steps or self.environment.is_terminal(state):
                cumulative_reward += self.environment.get_final_reward(state)
                break

        # backup
        self.back_propagate(cumulative_reward)