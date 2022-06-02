import random
import numpy as np

from Agents.Agent import Agent
from Commons.Utils import get_action_with_target

from config import *

np.random.seed(1)

class FixedAgent(Agent):
    def __init__(self, coordinates, name, agent_respawn, food_respawn, direction):
        super(FixedAgent, self).__init__(coordinates, name, direction)

        self.agent_idx = int(name.split('agent')[-1]) - 1

    def get_action(self, env, test = False):
        self.num_steps += 1

        ratio = np.random.rand()

        if ratio > action_reference:
            target = self.get_nearest_enemy(env)
        else:
            target = self.get_nearest_food(env, self.x, self.y)

        if self.is_moveout():
            return 'W'
        elif target != None:
            if ratio > action_reference:
                return self.get_action_with_enemy(target)
            else:
                return get_action_with_target(self.y, self.x, self.direction, target)
        else:
            return random.choice(list(self.legal_moves(env)))

    def insight(self, ):
        return 0


    def get_action_with_enemy(self, target):
        x_bias = target.x - self.x
        y_bias = target.y - self.y
        if x_bias < 2:
            if y_bias < 0:
                if self.direction == 'U':
                    return 'B'
                else:
                    if self.direction == 'D':
                        return 'RR'
                    elif self.direction == 'L':
                        return 'RR'
                    elif self.direction == 'R':
                        return 'RL'

            else:
                if self.direction == 'D':
                    return 'B'
                else:
                    if self.direction == 'U':
                        return 'RR'
                    elif self.direction == 'L':
                        return 'RL'
                    elif self.direction == 'R':
                        return 'RR'

        elif y_bias < 2:
            if x_bias < 0:
                if self.direction == 'L':
                    return 'B'
                else:
                    if self.direction == 'U':
                        return 'RL'
                    elif self.direction == 'D':
                        return 'RR'
                    elif self.direction == 'R':
                        return 'RR'

            else:
                if self.direction == 'R':
                    return 'B'
                else:
                    if self.direction == 'U':
                        return 'RR'
                    elif self.direction == 'D':
                        return 'RL'
                    elif self.direction == 'L':
                        return 'RR'
        else:

            if x_bias > 0:
                if self.direction == 'U':
                    return 'MR'
                elif self.direction == 'D':
                    return 'ML'
                elif self.direction == 'L':
                    return 'MB'
                elif self.direction == 'R':
                    return 'MF'
            elif x_bias < 0:
                if self.direction == 'U':
                    return 'ML'
                elif self.direction == 'D':
                    return 'MR'
                elif self.direction == 'R':
                    return 'MB'
                elif self.direction == 'L':
                    return 'MF'
            elif y_bias > 0:
                if direction == 'U':
                    return 'MB'
                elif self.direction == 'D':
                    return 'MF'
                elif self.direction == 'L':
                    return 'ML'
                elif self.direction == 'R':
                    return 'MR'
            elif y_bias < 0:
                if self.direction == 'U':
                    return 'MF'
                elif self.direction == 'D':
                    return 'MB'
                elif self.direction == 'L':
                    return 'MR'
                elif self.direction == 'R':
                    return 'ML'
            else:
                return 'W'

    def get_nearest_enemy(self, env):
        '''Get the nearest agent from an agent'''
        agents = [a for a in env.agents if not a.is_moveout() and a.name != self.name]
        distance = 99999
        temp = 0
        target = None

        for agent in agents:
            temp = pow(self.y - agent.y, 2) + pow(self.x - agent.x, 2)
            if temp < distance:
                target = agent
                distance = temp

        return target