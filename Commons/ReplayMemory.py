import numpy as np
import random

from config import *

class ReplayMemory:

    def __init__(self):
        self.count = 0
        self.current = 0

        self.actions = np.empty((mem_size, min_history + states_to_update), dtype=np.unicode)
        self.rewards = np.empty((mem_size, min_history + states_to_update), dtype=np.int32)
        self.terminals = np.empty((mem_size, min_history + states_to_update), dtype=np.float16)

        if data_format == 'NHWC':
            self.states = np.empty((mem_size, (min_history + states_to_update), MAP_HEIGHT if is_full_observable else VIEW_HEIGHT, MAP_WIDTH if is_full_observable else VIEW_WIDTH, n_layer), dtype=np.uint8)
            self.states_n = np.empty((mem_size, (min_history + states_to_update), MAP_HEIGHT if is_full_observable else VIEW_HEIGHT, MAP_WIDTH if is_full_observable else VIEW_WIDTH, n_layer), dtype=np.uint8)
            self.states_out = np.empty((batch_size, min_history + states_to_update, MAP_HEIGHT if is_full_observable else VIEW_HEIGHT, MAP_WIDTH if is_full_observable else VIEW_WIDTH, n_layer), dtype=np.uint8)
            self.states_n_out = np.empty((batch_size, min_history + states_to_update, MAP_HEIGHT if is_full_observable else VIEW_HEIGHT, MAP_WIDTH if is_full_observable else VIEW_WIDTH, n_layer), dtype=np.uint8)
        elif data_format == 'NCHW':
            self.states = np.empty((mem_size, (min_history + states_to_update), n_layer, MAP_HEIGHT if is_full_observable else VIEW_HEIGHT, MAP_WIDTH if is_full_observable else VIEW_WIDTH), dtype=np.uint8)
            self.states_n = np.empty((mem_size, (min_history + states_to_update), n_layer, MAP_HEIGHT if is_full_observable else VIEW_HEIGHT, MAP_WIDTH if is_full_observable else VIEW_WIDTH), dtype=np.uint8)
            self.states_out = np.empty((batch_size, min_history + states_to_update, n_layer, MAP_HEIGHT if is_full_observable else VIEW_HEIGHT, MAP_WIDTH if is_full_observable else VIEW_WIDTH), dtype=np.uint8)
            self.states_n_out = np.empty((batch_size, min_history + states_to_update, n_layer, MAP_HEIGHT if is_full_observable else VIEW_HEIGHT, MAP_WIDTH if is_full_observable else VIEW_WIDTH), dtype=np.uint8)

        
        self.actions_out = np.empty((batch_size, min_history + states_to_update), dtype=np.unicode)
        self.rewards_out = np.empty((batch_size, min_history + states_to_update))
        self.terminals_out = np.empty((batch_size, min_history + states_to_update))

    def add(self, state, action, reward, state_n, terminal):

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states[self.current] = state
        self.states_n[self.current] = state_n
        self.terminals[self.current] = float(terminal)
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % mem_size


    def getState(self, index):
        s = self.states[index]
        a = self.actions[index]
        r = self.rewards[index]
        n = self.states_n[index]
        t = self.terminals[index]

        return s, a, r, n, t

    def sample_batch(self):
        assert self.count > min_history + states_to_update

        indices = []
        while len(indices) < batch_size:

            index = random.randint(0, self.count - 1)

            self.states_out[len(indices)], self.actions_out[len(indices)], self.rewards_out[len(indices)], self.states_n_out[len(indices)], self.terminals_out[len(indices)] = self.getState(index)
            indices.append(index)


        state_out = np.reshape(self.states_out, [batch_size * (min_history + states_to_update), n_layer, MAP_HEIGHT if is_full_observable else VIEW_HEIGHT, MAP_WIDTH if is_full_observable else VIEW_WIDTH])
        states_n_out = np.reshape(self.states_n_out, [batch_size * (min_history + states_to_update), n_layer, MAP_HEIGHT if is_full_observable else VIEW_HEIGHT, MAP_WIDTH if is_full_observable else VIEW_WIDTH])
        rewards_out = np.reshape(self.rewards_out, [batch_size * (min_history + states_to_update)])
        actions_out = np.reshape(self.actions_out, [batch_size * (min_history + states_to_update)])
        terminals_out = np.reshape(self.terminals_out, [batch_size * (min_history + states_to_update)])
        return state_out, actions_out, rewards_out, states_n_out, terminals_out