from __future__ import print_function
from functools import reduce

import numpy as np
import random
import util
import sys
from copy import copy
import os
# import arcade

# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf
from Agents.Agent import Agent
from Commons.Utils import linear, conv2d, clipped_error, conv2dv2
from Commons.ReplayMemory import ReplayMemory

from config import *

np.random.seed(1)
tf.set_random_seed(1)

params = {
    'model_dir': 'saves/DRQN/',
    'save_file': 'DRQN'
}

class DRQNAgent2(Agent):
    # def __init__(self, coordinates, name, agent_respawn, food_respawn, direction, preference):
    def __init__(self, coordinates, name, agent_respawn, food_respawn, direction, preference):
        super(DRQNAgent2, self).__init__(coordinates, name, direction)

        self.params = params
        self.params['width'] = MAP_WIDTH
        self.params['height'] = MAP_HEIGHT
        self.params['train_width'] = MAP_WIDTH if is_full_observable else VIEW_WIDTH
        self.params['train_height'] = MAP_HEIGHT if is_full_observable else VIEW_HEIGHT
        self.params['agent_respawn'] = agent_respawn
        self.params['food_respawn'] = food_respawn

        # self.sess = sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.sess = tf.InteractiveSession(graph = tf.Graph(), config = tf.ConfigProto(gpu_options = gpu_options))

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.build_dqn()

        # self.replay_mem = deque()
        self.replay_mem = ReplayMemory()

        self.update_count, self.ep_reward, self.collect_ep_reward = 0, 0., 0.
        self.total_loss, self.total_q = 0., 0.
        # self.max_avg_ep_reward = 0
        # self.max_collect_ep_reward = 0
        self.ep_rewards, self.collect_ep_rewards = [], []
        self.countA, self.countO=0, 0

        self.numeps = 0
        self.best_reward_history = 0.

        # initial zero state to be used when starting episode
        self.initial_zero_state_batch = np.zeros((batch_size, lstm_size))
        self.initial_zero_state_single = np.zeros((1, lstm_size))

        # initialize for only evaluation
        self.eval_ep_rewards = []
        self.beam_rates = []
        self.moved_outs = []

        self.preference = preference

    def initialize(self, env, info): # inspects the starting state

        # Reset state
        self.last_state = None
        self.current_state = None
        self.sequence_state = []
        self.sequence_action = []
        self.sequence_reward = []

        # Reset actions
        self.last_action = None
        self.last_status = self.is_moveout()
        # Num of episode
        self.numeps += 1

        self.x = info[0]
        self.y = info[1]
        self.direction = info[2]

        self.train_count = 0

        if is_full_observable:
            self.current_state = self.get_fullstate_matrix(env)
            # init_state = self.get_fullstate_matrix(env)
        else:
            self.current_state = self.get_state_matrix(env)

        for _ in range(min_history + states_to_update):
            self.sequence_state.append(self.current_state)
            self.sequence_reward.extend([0])
            self.sequence_action.extend([0])

        # self.lstm_state_c, self.lstm_state_h = self.initial_zero_state_single, self.initial_zero_state_single
        self.lstm_state_single = (self.initial_zero_state_single, self.initial_zero_state_single)

    def reset_observation(self, env):
        self.sequence_state = []
        self.sequence_action = []
        self.sequence_reward = []

        if is_full_observable:
            self.current_state = self.get_fullstate_matrix(env)
        else:
            self.current_state = self.get_state_matrix(env)

        for _ in range(min_history + states_to_update):
            self.sequence_state.append(self.current_state)
            self.sequence_reward.extend([0])
            self.sequence_action.extend([0])

        self.lstm_state_single = (self.initial_zero_state_single, self.initial_zero_state_single)
    
    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((batch_size, 8))
        for i in range(len(actions)):                                           
            actions_onehot[i][self.get_action_value(actions[i])] = 1
        return actions_onehot

    def process_actions(self, actions):
        for i in range(len(actions)):      
            actions[i]=self.get_action_value(actions[i])
        return actions

    def opponent_in_view(self, env):
        for agent in env.agents:
            if agent.name != self.name:
                if agent.is_moveout():
                    return False
                else:
                    r_1, r_2, c_1, c_2, gap_h_r, gap_h_l, gap_v_u, gap_v_d = self.get_vision_matrix(env)
                    if agent.y >= r_1 and agent.y <= r_2 and agent.x >= c_1 and agent.x <= c_2:
                        return True
                    else:
                        return False

    def get_fullstate_matrix(self, env):
        if data_format == 'NHWC':
            matrix = np.zeros((self.params['height'], self.params['width'], n_layer), dtype=np.uint8)
        elif data_format == 'NCHW':
            matrix = np.zeros((n_layer, self.params['height'], self.params['width']), dtype=np.uint8)

        # beam : yellow
        for i1, value in enumerate(env.beams_set):
            for i2, r_loc in enumerate(value):
                matrix[0][r_loc[1]][r_loc[0]] = 255
                matrix[1][r_loc[1]][r_loc[0]] = 255
                matrix[2][r_loc[1]][r_loc[0]] = 0
                # matrix[0][r_loc[1]][r_loc[0]] = 200

        # food : green
        for food in env.food_objects:
            matrix[0][food.y][food.x] = 0 if food.food_type == 'A' else 255
            matrix[1][food.y][food.x] = 255 if food.food_type == 'A' else 165
            matrix[2][food.y][food.x] = 0 if food.food_type == 'A' else 0
            # matrix[0][food.y][food.x] = 150
        
        # opponent : red
        for agent in env.agents:
            if agent.name != self.name and not agent.is_moveout():
                matrix[0][agent.y][agent.x] = 255
                matrix[1][agent.y][agent.x] = 0
                matrix[2][agent.y][agent.x] = 0

                # agent direction mark
                posX, posY = agent.direction_mark()
                if posX != -1 and posY != -1 and matrix[0][posY][posX] == 0 and matrix[1][posY][posX] == 0 and matrix[2][posY][posX] == 0:
                    matrix[0][posY][posX] = 220
                    matrix[1][posY][posX] = 220
                    matrix[2][posY][posX] = 220
                    # matrix[0][agent.y][agent.x] = 100

        # actor : blue
        for agent in env.agents:
            if agent.name == self.name and not agent.is_moveout():
                matrix[0][agent.y][agent.x] = 0
                matrix[1][agent.y][agent.x] = 0
                matrix[2][agent.y][agent.x] = 255
                # matrix[0][agent.y][agent.x] = 50

                # agent direction mark
                posX, posY = agent.direction_mark()
                if posX != -1 and posY != -1 and matrix[0][posY][posX] == 0 and matrix[1][posY][posX] == 0 and matrix[2][posY][posX] == 0:
                    matrix[0][posY][posX] = 220
                    matrix[1][posY][posX] = 220
                    matrix[2][posY][posX] = 220
                    # matrix[0][agent.y][agent.x] = 100

                break

        return matrix


    def get_state_matrix(self, env):
        '''Get partially observed view matrix'''

        r_1 = 0; r_2 = 0; c_1 = 0; c_2 = 0
        gap_h_r = 0; gap_h_l = 0; gap_v_u = 0; gap_v_d = 0

        r_1, r_2, c_1, c_2, gap_h_r, gap_h_l, gap_v_u, gap_v_d = self.get_vision_matrix(env)

        matrix = np.zeros((n_layer, self.params['height'], self.params['width']), dtype=np.uint8)

        # beam : yellow
        for i1, value in enumerate(env.beams_set):
            for i2, r_loc in enumerate(value):
                matrix[0][r_loc[1]][r_loc[0]] = 255
                matrix[1][r_loc[1]][r_loc[0]] = 255
                matrix[2][r_loc[1]][r_loc[0]] = 0
                # matrix[0][r_loc[1]][r_loc[0]] = 200

        # food : green
        for food in env.food_objects:
            matrix[0][food.y][food.x] = 0
            matrix[1][food.y][food.x] = 255
            matrix[2][food.y][food.x] = 0
            # matrix[0][food.y][food.x] = 150
        
        # opponent : red
        for agent in env.agents:
            if agent.name != self.name and not agent.is_moveout():
                matrix[0][agent.y][agent.x] = 255
                matrix[1][agent.y][agent.x] = 0
                matrix[2][agent.y][agent.x] = 0
                # matrix[0][agent.y][agent.x] = 100

                # agent direction mark
                posX, posY = self.direction_mark()
                if posX != -1 and posY != -1 and (matrix[0][posY][posX] == 0 and matrix[1][posY][posX] == 0 and matrix[2][posY][posX] == 0):
                # if posX != -1 and posY != -1 and (matrix[0][posY][posX] == 0):
                    matrix[0][posY][posX] = 220
                    matrix[1][posY][posX] = 220
                    matrix[2][posY][posX] = 220
                    # matrix[0][posY][posX] = 220

        # actor : blue
        for agent in env.agents:
            if agent.name == self.name and not agent.is_moveout():
                matrix[0][agent.y][agent.x] = 0
                matrix[1][agent.y][agent.x] = 0
                matrix[2][agent.y][agent.x] = 255
                # matrix[0][agent.y][agent.x] = 50

                # agent direction mark
                posX, posY = self.direction_mark()
                if posX != -1 and posY != -1 and (matrix[0][posY][posX] == 0 and matrix[1][posY][posX] == 0 and matrix[2][posY][posX] == 0):
                # if posX != -1 and posY != -1 and (matrix[0][posY][posX] == 0):
                    matrix[0][posY][posX] = 220
                    matrix[1][posY][posX] = 220
                    matrix[2][posY][posX] = 220
                    # matrix[0][posY][posX] = 220
                break

        return_matrix = matrix[:, r_1 : r_2 + 1, c_1 : c_2 + 1]
        for i in range(gap_h_l, 0):
            return_matrix = np.insert(return_matrix, 0, values=0, axis=2)

        for i in range(0, gap_h_r):
            return_matrix = np.insert(return_matrix, len(return_matrix[0][0]), values=0, axis=2)

        for i in range(gap_v_u, 0):
            return_matrix = np.insert(return_matrix, 0, values=0, axis=1)

        for i in range(0, gap_v_d):
            return_matrix = np.insert(return_matrix, len(return_matrix[0]), values=0, axis=1)

        observation = self.rotate_matrix(return_matrix)
        # observation = np.swapaxes(observation, 0, 2)

        if data_format == 'NHWC':
            observation = np.transpose(observation, [1, 2, 0])

        return observation

    def get_vision_matrix(self, env):
        '''Get partialy observation information'''
        '''Specifically boundaries and possible gaps of 4 sides'''
        
        cache_x, cache_y = self.forward_direction_cache()
        # cache_x, cache_y = cache_x * 15, cache_y * 15
        cache_x, cache_y = cache_x * (self.params['train_height'] - 1), cache_y * (self.params['train_height'] - 1)
        r_1 = 0; r_2 = 0; c_1 = 0; c_2 = 0 # rows and columns to label the rectangle
        gap_h_r = 0; gap_h_l = 0; gap_v_u = 0; gap_v_d = 0

        side_length = self.params['height'] - 1

        if self.direction == 'U':
            r_1 = self.y + cache_y if self.y + cache_y >= 0 else 0
            r_2 = self.y

            c_1 = self.x - side_length if self.x - side_length >= 0 else 0
            c_2 = self.x + side_length if self.x + side_length <= self.env_x_size - 1 else self.env_x_size - 1

            if self.x - side_length < 0:
                gap_h_l = self.x - side_length
            if self.x + side_length > self.env_x_size - 1:
                gap_h_r = self.x + side_length - (self.env_x_size - 1)

            if self.y + cache_y < 0:
                gap_v_u = self.y + cache_y
            
            
        elif self.direction == 'D':
            r_1 = self.y
            r_2 = self.y + cache_y if self.y + cache_y <= self.env_y_size - 1 else self.env_y_size - 1
            c_1 = self.x - side_length if self.x - side_length >= 0 else 0
            c_2 = self.x + side_length if self.x + side_length <= self.env_x_size - 1 else self.env_x_size - 1

            if self.x - side_length < 0:
                gap_h_l = self.x - side_length
            if self.x + side_length > self.env_x_size - 1:
                gap_h_r = self.x + side_length - (self.env_x_size - 1)

            if self.y + cache_y > self.env_y_size - 1:
                gap_v_d = self.y + cache_y - (self.env_y_size - 1)

        elif self.direction == 'R':
            r_1 = self.y - side_length if self.y - side_length >= 0 else 0
            r_2 = self.y + side_length if self.y + side_length <= self.env_y_size - 1 else self.env_y_size - 1
            c_1 = self.x
            c_2 = self.x + cache_x if self.x + cache_x <= self.env_x_size - 1 else self.env_x_size - 1

            if self.x + cache_x > self.env_x_size - 1:
                gap_h_r = self.x + cache_x - (self.env_x_size - 1)

            if self.y - side_length < 0:
                gap_v_u = self.y - side_length
            if self.y + side_length > (self.env_y_size - 1):
                gap_v_d = self.y + side_length - (self.env_y_size - 1)

        elif self.direction == 'L':
            r_1 = self.y - side_length if self.y - side_length >= 0 else 0
            r_2 = self.y + side_length if self.y + side_length <= self.env_y_size - 1 else self.env_y_size - 1
            c_1 = self.x + cache_x if self.x + cache_x >= 0 else 0
            c_2 = self.x
            

            if self.x + cache_x < 0:
                gap_h_l = self.x + cache_x

            if self.y - side_length < 0:
                gap_v_u = self.y - side_length
            if self.y + side_length > (self.env_y_size - 1):
                gap_v_d = self.y + side_length - (self.env_y_size - 1)

        return r_1, r_2, c_1, c_2, gap_h_r, gap_h_l, gap_v_u, gap_v_d

    def rotate_matrix(self, matrix):
        '''Rotate matrix according to agent orientation'''
        if self.direction == 'D':
            matrix[0] = np.flip(matrix[0])
            matrix[1] = np.flip(matrix[1])
            matrix[2] = np.flip(matrix[2])
        elif self.direction == 'R':
            return_mat = []
            return_mat.append(np.transpose(np.flip(matrix[0], axis=1)))
            return_mat.append(np.transpose(np.flip(matrix[1], axis=1)))
            return_mat.append(np.transpose(np.flip(matrix[2], axis=1)))
            matrix = return_mat
        elif self.direction == 'L':
            return_mat = []
            return_mat.append(np.flip(np.transpose(matrix[0]), axis=1))
            return_mat.append(np.flip(np.transpose(matrix[1]), axis=1))
            return_mat.append(np.flip(np.transpose(matrix[2]), axis=1))
            matrix = return_mat
        return matrix

    def get_action(self, env, test = False):

        eps = 0.05 if test else (0.1 if self.load_existing_model else eps_final + max(0, (eps_initial - eps_final) * (eps_step - max(0, float(self.num_steps) - train_start)) / eps_step))
        # if np.random.rand() > eps:
        if np.random.rand() > eps:
            # self.lstm_state_c, self.lstm_state_h = self.initial_zero_state_single, self.initial_zero_state_single

            # Exploit / Explore
            action, self.lstm_state_single = self.sess.run(
                [self.prediction, self.rnn_state_train], feed_dict=
                {
                    self.s : np.reshape(self.current_state,
                    (1, n_layer, self.params['train_height'], self.params['train_width']) if data_format == 'NCHW' else (1, self.params['train_height'], self.params['train_width'], n_layer)),
                    self.batch_size: 1,
                    self.train_length: 1,
                    self.lstm_state_train_in: self.lstm_state_single
                }
            )

            move = self.get_action_keyword(action[0])
        else:

            self.lstm_state_single = self.sess.run(
                [self.rnn_state_train],
                {
                    self.s: np.reshape(self.current_state,
                        (1, n_layer, self.params['train_height'], self.params['train_width']) if data_format == 'NCHW' else (1, self.params['train_height'], self.params['train_width'], n_layer)),
                    self.batch_size: 1,
                    self.train_length: 1,
                    self.lstm_state_train_in: self.lstm_state_single
                }
            )
            

            move = random.choice(list(self.legal_moves(env)))

        # Save last_action
        self.last_action = move
        if not test:
            self.num_steps += 1

        return move

    # def observe(self, env, reward, action):
    def observe(self, env, reward, action, beam_rate, moved_out_count, opponent_reward=0, food_type=''):
        if self.num_steps == train_start:
            self.update_count, self.ep_reward, self.collect_ep_reward = 0, 0., 0.
            self.total_loss, self.total_q = 0., 0.
            self.ep_rewards, self.collect_ep_rewards = [], []

        self.ep_reward += reward
        self.collect_ep_reward += ((1 if reward > 0 else 0) + (1 if opponent_reward > 0 else 0))
        if food_type == 'A':
            self.countA += 1
        elif food_type == 'O':
            self.countO += 1

        # before_state = self.current_state
        before_state = self.sequence_state[:]

        if is_full_observable:
            self.current_state = self.get_fullstate_matrix(env)
        else:
            self.current_state = self.get_state_matrix(env)

        # svo_reward = (opponent_reward * svo_value if self.opponent_in_view(env) else 0)

        self.sequence_state.pop(0)
        self.sequence_action.pop(0)
        self.sequence_reward.pop(0)
        self.sequence_state.append(self.current_state)
        self.sequence_action.extend([self.get_action_value(action)])
        # self.sequence_reward.extend([reward + svo_reward])
        # self.sequence_reward.extend([reward])
        self.sequence_reward.extend([reward * 0 if food_type == '' else self.preference[food_type]])

        current_status = self.is_moveout()

        if not current_status and not self.last_status:
            self.replay_mem.add(before_state, self.sequence_action, self.sequence_reward, self.sequence_state, self.is_moveout())
        # self.replay_mem.add(before_state, self.sequence_action, self.sequence_reward, self.sequence_state, self.is_moveout())

        # Train
        if self.num_steps > train_start:
            if self.num_steps % Q_update_freq == 0:
                self.update_target_q_network()
            
            if self.num_steps % train_freq == 0:
                loss, q = self.train()
                self.ep_loss.append(loss)
                self.ep_q.append(q)

        self.last_status = current_status

        if self.num_steps > train_start:
            if self.num_steps % GAME_LENGTH == 0:
                # avg_reward = self.ep_reward / self.update_count
                avg_reward = self.ep_reward
                avg_loss = self.total_loss / self.update_count
                avg_q = self.total_q / self.update_count

                collect_reward = self.collect_ep_reward
                
                self.ep_rewards.append(self.ep_reward)
                self.ep_reward = 0.
                self.collect_ep_rewards.append(self.collect_ep_reward)
                self.collect_ep_reward = 0.

                try:
                    max_ep_reward = np.max(self.ep_rewards)
                    min_ep_reward = np.min(self.ep_rewards)
                    avg_ep_reward = np.mean(self.ep_rewards)
                except:
                    max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0
                    max_collect_ep_reward, min_collect_ep_reward, avg_collect_ep_reward = 0, 0, 0

                print('\nreward: %d, avg_l: %.6f, avg_q: %3.6f, beam_rate: %.2f, moved out: %d, collective rewards: %d' \
                    % (avg_reward, avg_loss, avg_q, beam_rate, moved_out_count, collect_reward))
  
                # self.max_avg_ep_reward = max(self.max_avg_ep_reward, avg_ep_reward)
                # self.max_collect_ep_reward = max(self.max_collect_ep_reward, avg_ep_reward)

                self.inject_train_summary({
                    'beam rate' : beam_rate,
                    'reward': avg_reward,
                    'collective reward': collect_reward,
                    'average.loss': avg_loss,
                    'average.q': avg_q,
                    'moved out': moved_out_count,
                    'apples': self.countA,
                    'oranges': self.countO,
                    
                }, self.num_steps)

                file_path = self.params['model_dir'] + "a" + str(self.params['agent_respawn']) + "f" + str(self.params['food_respawn']) + "/" + self.name + "/"
                log_file = open(file_path + 'train.log','a')
                log_file.write('# Steps: ' + str(self.num_steps) + '\n')
                log_file.write('# reward: ' + str(avg_reward) + '\n')
                log_file.write('# collective reward: ' + str(collect_reward) + '\n')
                log_file.write('# average.loss: ' + str(avg_loss) + '\n')
                log_file.write('# average.q: ' + str(avg_q) + '\n')
                log_file.write('# beam rate: ' + str(beam_rate) + '\n')
                log_file.write('# moved out: ' + str(moved_out_count) + '\n')
                
                log_file.write('=============================================================\n')
                sys.stdout.flush()

                self.total_loss = 0.
                self.total_q = 0.
                self.update_count = 0
                self.ep_reward, self.collect_ep_reward = 0., 0.
                self.countA, self.countO = 0, 0
                # for evaluation
                self.eval_ep_rewards = []
                self.beam_rates = []
                self.moved_outs = []
        else:
            if self.num_steps % GAME_LENGTH == 0:
                self.countA, self.countO = 0, 0

    def evaluate(self, env, reward, beam_count, final_step, final_ep, moved_out_count, opponent_reward=0, food_type=''):

        self.ep_reward += reward
        self.collect_ep_reward += ((1 if reward > 0 else 0) + (1 if opponent_reward>0 else 0))

        # svo_reward = (opponent_reward * svo_value if self.opponent_in_view(env) else 0)

        before_state = self.sequence_state[:]

        if is_full_observable:
            self.current_state = self.get_fullstate_matrix(env)
        else:
            self.current_state = self.get_state_matrix(env)
        
        self.sequence_state.pop(0)
        # self.sequence_state.extend(self.current_state)
        self.sequence_state.append(self.current_state)

        if self.num_steps > train_start:
            if final_step:
                avg_reward = self.ep_reward / GAME_LENGTH
                # avg_collect_reward = self.collect_ep_reward / GAME_LENGTH
                
                self.eval_ep_rewards.append(self.ep_reward)
                self.ep_reward, self.collect_ep_reward = 0., 0.

                self.beam_rates.append(beam_count)
                self.moved_outs.append(moved_out_count)

                if final_ep:

                    try:
                        max_ep_reward = np.max(self.eval_ep_rewards)
                        min_ep_reward = np.min(self.eval_ep_rewards)
                        avg_ep_reward = np.mean(self.eval_ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    # self.max_avg_ep_reward = max(self.max_avg_ep_reward, avg_ep_reward)

                    beam_rate = np.mean(self.beam_rates)
                    moved_out = np.mean(self.moved_outs)

                    self.inject_eval_summary({
                        'eval.ep.max reward' : max_ep_reward, 
                        'eval.ep.min reward' : min_ep_reward, 
                        'eval.ep.avg reward' : avg_ep_reward,
                        'eval.moved out' : moved_out,
                        'eval.beam rate' : beam_rate
                    }, self.num_steps)

                    file_path = self.params['model_dir'] + "a" + str(self.params['agent_respawn']) + "f" + str(self.params['food_respawn']) + "/" + self.name + "/"
                    log_file = open(file_path + 'eval.log','a')
                    log_file.write('# Steps: ' + str(self.num_steps) + '\n')
                    log_file.write('# reward: ' + str(avg_reward) + '\n')
                    log_file.write('# moved out: ' + str(moved_out) + '\n')
                    log_file.write('# beam rate: ' + str(beam_rate) + '\n')
                    log_file.write('=============================================================\n')
                    sys.stdout.flush()

                    self.eval_ep_rewards = []

                    # if save_model and avg_ep_reward > self.best_reward_history:
                    if save_model and self.num_steps % save_freq == 0:
                        self.save_model(file_path, "model.ckpt", self.num_steps)
                        print(self.name, str(self.num_steps))

                        self.best_reward_history = avg_ep_reward



    def inject_train_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_train_ops[tag] for tag in tag_dict.keys()], {
            self.summary_train_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, self.num_steps/GAME_LENGTH)

    def inject_eval_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_eval_ops[tag] for tag in tag_dict.keys()], {
            self.summary_eval_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, self.num_steps/GAME_LENGTH)

    def train(self):
        q_t, loss = np.zeros((batch_size, NUM_ACTION)), 0

        batch_s, batch_a, batch_r, batch_n, batch_t = self.replay_mem.sample_batch()

        train_batch = (self.initial_zero_state_batch, self.initial_zero_state_batch)

        target_q_values, q_next_action = self.sess.run(
            [self.targetQout, self.prediction],
            {
                self.s: batch_n,
                self.target_s: batch_n,
                self.batch_size: batch_size,
                self.train_length: min_history + states_to_update,
                self.lstm_state_target_in: train_batch,
                self.lstm_state_train_in: train_batch
            }
        )

        q_eval = self.sess.run(self.Qout, feed_dict = {
            self.s: batch_s,
            self.batch_size: batch_size,
            self.train_length: min_history + states_to_update,
            self.lstm_state_train_in: train_batch
        })
        q_target = q_eval.copy()

        max_target = target_q_values[range(batch_size * (min_history + states_to_update)), q_next_action]

        q_target = batch_r + (1 - batch_t) * discount * max_target

        _, q_t, loss, summary_str = self.sess.run(
            [self.optim, self.Qout, self.loss, self.q_summary], feed_dict = 
            {
                self.s: batch_s,
                self.nextQ: q_target,
                self.action: batch_a,
                self.batch_size: batch_size,
                self.train_length: min_history + states_to_update,
                self.lstm_state_train_in: train_batch
            }
        )


        self.writer.add_summary(summary_str, self.num_steps)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

        return loss, q_t.mean()

    def initiate_train(self):
        self.episode_reward = 0
        self.ep_loss = []
        self.ep_q = []

    def get_info(self):
        return self.name, self.x, self.y, self.direction

    def clone(self):
        agent  = copy(self)
        return agent

    def build_dqn(self):
        self.w = {}
        self.t_w = {}

        # initializer = tf.truncated_normal_initializer(0, 0.02)
        initializer = tf.truncated_normal_initializer(0, 0.01)
        activation_fn = tf.nn.relu

        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=True)

        # training network
        if data_format == 'NHWC':
            self.s = tf.placeholder(dtype = tf.float32, shape = [None, self.params['train_height'], self.params['train_width'], n_layer], name='s')
            self.target_s = tf.placeholder(dtype = tf.float32, shape = [None, self.params['train_height'], self.params['train_width'], n_layer], name='target_s')
        elif data_format == 'NCHW':
            self.s = tf.placeholder(dtype = tf.float32, shape = [None, n_layer, self.params['train_height'], self.params['train_width']], name='s')
            self.target_s = tf.placeholder(dtype = tf.float32, shape = [None, n_layer, self.params['train_height'], self.params['train_width']], name='target_s')

        self.action = tf.placeholder(tf.int32, shape=[None], name='action')
        self.terminal = tf.placeholder(dtype=tf.float32, shape=[None], name="terminal")

        self.actions_onehot = tf.one_hot(self.action, NUM_ACTION, 1.0, 0.0, name='actions_onehot')

        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
        self.train_length = tf.placeholder(tf.int32, name='train_length')

        self.lstm_state_train_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.lstm_state_target_in = rnn_cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            if is_full_observable:
                self.l1, self.w['l1_w'], self.w['l1_b'] = conv2dv2(self.s, 32, [3, 3], [1, 1], c_names, initializer, activation_fn, name='l1')
                self.l2, self.w['l2_w'], self.w['l2_b'] = conv2dv2(self.l1, 32, [1, 1], [1, 1], c_names, initializer, activation_fn, name='l2')
            else:
                self.l1, self.w['l1_w'], self.w['l1_b'] = conv2dv2(self.s, 32, [4, 4], [2, 2], c_names, initializer, activation_fn, name='l1')
                self.l2, self.w['l2_w'], self.w['l2_b'] = conv2dv2(self.l1, 32, [2, 2], [1, 1], c_names, initializer, activation_fn, name='l2')

            shape = self.l2.get_shape().as_list()
            final_dims = shape[1] * shape[2] * shape[3]
            self.l2_flat = tf.reshape(self.l2, [-1, final_dims])
            rnn_input = tf.reshape(self.l2_flat, [self.batch_size, self.train_length, final_dims])

            out, self.rnn_state_train = tf.nn.dynamic_rnn(inputs=rnn_input, cell=rnn_cell, initial_state=self.lstm_state_train_in, scope='lstm_train', dtype=tf.float32)
            
            rnn_output = tf.reshape(out, [-1, lstm_size])

            self.l3, self.w['l3_w'], self.w['l3_b'] = linear(rnn_output, 512, c_names, activation_fn=activation_fn, name='l3')
            self.Qout , self.w['q_w'], self.w['q_b'] = linear(self.l3, NUM_ACTION, c_names, name='q')

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis = 1)

            self.prediction = tf.argmax(self.Qout, dimension=1)

        # target network
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            if is_full_observable:
                self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2dv2(self.target_s, 32, [3, 3], [1, 1], c_names, initializer, activation_fn, name='target_l1')
                self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2dv2(self.target_l1, 32, [1, 1], [1, 1], c_names, initializer, activation_fn, name='target_l2')
            else:
                self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2dv2(self.target_s, 32, [4, 4], [2, 2], c_names, initializer, activation_fn, name='target_l1')
                self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2dv2(self.target_l1, 32, [2, 2], [1, 1], c_names, initializer, activation_fn, name='target_l2')

            shape = self.target_l2.get_shape().as_list()
            final_dims = shape[1] * shape[2] * shape[3]
            self.target_l2_flat = tf.reshape(self.target_l2, [-1, final_dims])
            rnn_input = tf.reshape(self.target_l2_flat, [self.batch_size, self.train_length, final_dims])

            out, self.rnn_state_target = tf.nn.dynamic_rnn(inputs=rnn_input, cell=rnn_cell, initial_state=self.lstm_state_target_in, scope='lstm_train', dtype=tf.float32)
            
            rnn_output = tf.reshape(out, [-1, lstm_size])

            self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = linear(rnn_output, 512, c_names, activation_fn=activation_fn, name='target_l3')
            self.targetQout , self.t_w['q_w'], self.t_w['q_b'] = linear(self.target_l3, NUM_ACTION, c_names, name='target_q')

            self.targetQ_prediction = tf.argmax(self.targetQout, dimension=1)
            self.targetQ = tf.reduce_sum(tf.multiply(self.targetQout, self.actions_onehot), axis = 1)
        

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder(dtype = tf.float32, shape = self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        self.global_step = tf.Variable(0, trainable=False)

        with tf.variable_scope('loss'):
            # self.target_q_t = tf.placeholder(dtype = tf.float32, shape = [None, NUM_ACTION], name='target_q_t')
            self.nextQ = tf.placeholder(dtype = tf.float32, shape = [None], name='nextQ')

            # self.maskA = tf.zeros([self.batch_size, min_history])
            # self.maskB = tf.ones([self.batch_size, states_to_update])
            # self.mask = tf.concat([self.maskA, self.maskB], 1)
            # self.mask = tf.reshape(self.mask, [-1])

            self.maskA = tf.zeros([self.batch_size, min_history])
            self.maskB = tf.ones([self.batch_size, states_to_update])
            self.mask = tf.concat([self.maskA, self.maskB], 1)
            self.mask = tf.reshape(self.mask, [-1])

            # huber loss
            hubers = tf.losses.huber_loss(self.nextQ, self.Q)
            self.loss = tf.reduce_mean(hubers)

        with tf.variable_scope('optimizer'):

            self.optim = tf.train.AdamOptimizer(lr).minimize(self.loss)

        with tf.variable_scope('train_summary'):
            scalar_train_summary_tags = ['beam rate', 'reward', 'collective reward', 'average.loss', 'average.q', 'moved out', 'apples', 'oranges']

            self.summary_train_placeholders = {}
            self.summary_train_ops = {}

            for tag in scalar_train_summary_tags:
                self.summary_train_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_train_ops[tag]  = tf.summary.scalar(tag, self.summary_train_placeholders[tag])

            histogram_summary_tags = []

            for tag in histogram_summary_tags:
                self.summary_train_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_train_ops[tag]  = tf.summary.histogram(tag, self.summary_train_placeholders[tag])


            self.single_q = tf.placeholder(dtype = tf.float32, shape = None, name='single_q')
            q_record = tf.summary.scalar('q_record', self.single_q)
            self.record_q_op = tf.summary.merge([q_record])

        with tf.variable_scope('eval_summary'):
            scalar_eval_summary_tags = ['eval.ep.max reward', 'eval.ep.min reward', 'eval.ep.avg reward', 'eval.moved out', 'eval.beam rate']

            self.summary_eval_placeholders = {}
            self.summary_eval_ops = {}

            for tag in scalar_eval_summary_tags:
                self.summary_eval_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_eval_ops[tag]  = tf.summary.scalar(tag, self.summary_eval_placeholders[tag])

        with tf.name_scope('performance'):
            q_summary = []
            avg_q = tf.reduce_mean(self.Qout, 0)
            for idx in range(NUM_ACTION):
               q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))

            self.q_summary = tf.summary.merge([tf.summary.histogram("nextQ", self.nextQ),
                tf.summary.histogram("Q", self.Q)],
                q_summary)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)

        path = (os.path.expanduser("~") if save_under_user_home else '') + '/TensorBoard/DRQN/' + self.name

        self.writer = tf.summary.FileWriter(path, self.sess.graph)

        self.load_existing_model = self.load_model()
        self.update_target_q_network()

    def update_target_q_network(self):

        for name in self.w.keys():
            self.sess.run(self.t_w_assign_op[name], feed_dict = {
                self.t_w_input[name]: (self.w[name].eval(session=self.sess) * TAU) + (self.t_w[name].eval(session=self.sess) * (1 - TAU))
                # self.t_w_input[name]: self.w[name].eval(session=self.sess)
            })

    def save_model(self, file_path, file_name, steps):
        '''save model to assigned dir'''
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.saver.save(self.sess, file_path + file_name, global_step=steps)
        print('Model saved: ', file_path + file_name)

    def load_model(self):
        print('Loaded model...')

        file_path = self.params['model_dir'] + "a" + str(self.params['agent_respawn']) + "f" + str(self.params['food_respawn']) + "/" + self.name+ "/"

        ckpt = tf.train.get_checkpoint_state(file_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(file_path, ckpt_name)
            
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % file_path)
            return False