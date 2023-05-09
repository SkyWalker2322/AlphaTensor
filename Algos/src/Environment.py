import numpy as np
import random
import itertools



class State:
    def __init__(self, tensor: np.ndarray):
        self.tensor = np.array(tensor, dtype=int)

    def to_str(self) -> str:
        return str(self.tensor)

    @property
    def rank_upper_bound(self) -> int:
        return float(np.sum(np.linalg.matrix_rank(self.tensor)))

class Action:
    def __init__(self, u, v, w):
        u = np.array(u, dtype=int)
        v = np.array(v, dtype=int)
        w = np.array(w, dtype=int)

        self.u = u
        self.v = v
        self.w = w
        self.tensor = u[:, None, None] * v[None, :, None] * w[None, None]

u_values = [-1, 0, 1]
u = list(itertools.product(u_values, repeat=4))
v_values = [-1, 0, 1]
v = list(itertools.product(v_values, repeat=4))
w_values = [-1, 0, 1]
w = list(itertools.product(w_values, repeat=4))
all_tuples = list(itertools.product(u, v, w))

Actions = []    #appending all possible actions
for tup in all_tuples:
    Actions.append(Action(tup[0], tup[1], tup[2]))

ACTIONS = Actions      # Action space


INIT_STATE = State([
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0 ,0]],
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [1, 0, 0, 0],
     [0, 1, 0, 0]],
    [[0, 0, 1, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
])


class Environment:

    @property
    def num_actions(self):
        return len(ACTIONS)

    def is_terminal(self, state) -> bool:
        return np.all(state.tensor == 0)

    def get_init_state(self) -> State:
        return INIT_STATE

    def get_next_state(self, state: State, action_idx: int) -> State:
        return State(state.tensor - ACTIONS[action_idx].tensor)


    #reward functions
    def get_intermediate_reward(self, state: State, action_idx: int) -> float:
        return -1.0

    def get_final_reward(self, state: State) -> float:
        if self.is_terminal(state):
            return 0.0

        return -state.rank_upper_bound

    #generates Synthetic examples for better fitting of neural networks
    def generate_synthetic_examples(self, max_num_steps: int):
        n = min(self.num_actions, max_num_steps)
        indices = np.random.choice(self.num_actions, size=n)
        state = State(np.sum([ACTIONS[i].tensor for i in indices], axis=0))
        examples = []
        cumulative_rewards = []
        cumulative_reward = 0
        for action_idx in indices:
            examples.append([state, np.eye(self.num_actions)[action_idx], None])
            print(examples)
            cumulative_rewards.append(cumulative_reward)

            cumulative_reward += self.get_intermediate_reward(state, action_idx)
            state = self.get_next_state(state, action_idx)

        assert self.is_terminal(state)
        cumulative_reward += self.get_final_reward(state)

        return [(s, pi, cumulative_reward - r) for (s, pi, _), r in zip(examples, cumulative_rewards)]
