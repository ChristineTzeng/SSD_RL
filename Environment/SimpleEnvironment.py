from Agents.Agent import Agent
from Agents.Food import Food

from config import MAP_WIDTH, MAP_HEIGHT, GAME_LENGTH
from Commons.Utils import legal_moves

import numpy as np
import scipy.misc
import random

import sys
import time
from copy import copy, deepcopy

map_size = 10

class SimpleEnvironment(object):
    def __init__(self):
        self.size_x = map_size
        self.size_y = map_size
        self.numSteps = 0
        self.agents = []

        self.numeps = 0

    def terminated(self):
        # return any([self.food_objects.is_hidden()])
        return any([len(self.food_objects) < 1])

    def validActions(self):
        return legal_moves()

    def reset(self):
        '''reset game'''
        self.numSteps = 0

        # self.food = Food(coordinates=(4, 4))
        self.food_objects = []
        self.food_objects.append(Food(coordinates=(4, 4)))

        self.numeps += 1

    def execute(self, actions = None):
        '''Run agents simultaneously'''
        old_loc = [[actor.x, actor.y] for actor in self.agents]
        rewards = [0 for actor in self.agents]
        self.beams_set = []

        for idx, agent in enumerate(self.agents):
            self.agents[idx].move(actions[idx])

        # Count rewards for agents according to its position
        for food in self.food_objects:
            if not food.is_hidden():
                for idx, agent in enumerate(self.agents):
                    if food.x == self.agents[idx].x and food.y == self.agents[idx].y:
                        # rewards[idx] = food.collect(5)
                        self.food_objects.remove(food)

        self.numSteps += 1

        return rewards, self.terminated()

    def fullstate(self):
        ''' Perfect information: Returns a tuple of the player and foods locations as a proxy for complete state information. '''
        return tuple([(agent.x, agent.y, agent.direction) for agent in self.agents] + [(r.x, r.y) for r in self.food_objects if not r.is_hidden()])