import random
import numpy as np

from Agents.Agent import Agent
from Commons.Utils import get_action_with_target
from config import *

class HardcodedAgent(Agent):
    def __init__(self, coordinates, name, agent_respawn, food_respawn, direction):
        super(HardcodedAgent, self).__init__(coordinates, name, direction)

        self.agent_idx = int(name.split('agent')[-1]) - 1

    def get_action(self, env, test = False):
        self.num_steps += 1

        target_food = self.get_nearest_food(env, self.x, self.y)
        attack_target = []
        for idx, agent in enumerate(env.agents):
            if idx != self.agent_idx:
                attack_target.append((agent.x, agent.y))
                # opponent_target = self.get_nearest_food(env, agent.x, agent.y)
                # if opponent_target != None:
                #     attack_target.append((opponent_target.x, opponent_target.y))

        if self.is_moveout():
            return 'W'
        elif target_food != None:
            # return get_action_with_target(self.y, self.x, self.direction, target)
            if self.if_opponent_in_view(env):
                return 'B'
            else:
                return get_action_with_target(self.y, self.x, self.direction, target_food)
        else:
            return random.choice(list(self.legal_moves(env)))

    def get_estimated_reward_recursive(self, env, depth, if_beam):
        if depth == 0:
            return 0.0

        # if if_beam:

    def if_opponent_in_view(self, coordinates):
        matrix = self.get_state_matrix(coordinates)
        if self.direction == 'R':
            view_arr = matrix[self.y][self.x:]
        elif self.direction == 'L':
            view_arr = matrix[self.y][:self.x]
        elif self.direction == 'D':
            view_arr = matrix[:,self.y][self.x:]
        elif self.direction == 'U':
            view_arr = matrix[:,self.y][:self.x]

        return True if 150 in view_arr else False

    def get_state_matrix(self, coordinates):
        '''Get partially observed view matrix'''

        matrix = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)

        for i,cordnt in enumerate(coordinates):
            matrix[cordnt[1]][cordnt[0]] = 50

        return matrix


    def get_state_matrix(self, env):
        '''Get partially observed view matrix'''

        matrix = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)

        # beam : yellow
        for i1, value in enumerate(env.beams_set):
            for i2, r_loc in enumerate(value):
                matrix[r_loc[1]][r_loc[0]] = 50

        # food : green
        for food in env.food_objects:
            matrix[food.y][food.x] = 100
        
        # opponent : red
        for agent in env.agents:
            if agent.name != self.name and not agent.is_moveout():
                matrix[agent.y][agent.x] = 150

                # agent direction mark
                posX, posY = agent.direction_mark()
                if posX != -1 and posY != -1 and (matrix[posY][posX] == 0):
                    matrix[posY][posX] = 250

        # actor : blue
        for agent in env.agents:
            if agent.name == self.name and not agent.is_moveout():
                matrix[agent.y][agent.x] = 200

                # agent direction mark
                posX, posY = agent.direction_mark()
                if posX != -1 and posY != -1 and (matrix[posY][posX] == 0):
                    matrix[posY][posX] = 250
                break

        return matrix


