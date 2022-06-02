from Agents.Agent import Agent
from Agents.Food import Food

from config import MAP_WIDTH, MAP_HEIGHT, GAME_LENGTH

import numpy as np
import scipy.misc
import random

import sys
import time
from copy import copy, deepcopy

class SimulationEnvironment(object):
    def __init__(self, agent_hidden = 5, food_hidden = 4):
        self.size_x = MAP_WIDTH
        self.size_y = MAP_HEIGHT
        self.food_objects = []
        self.agent_hidden = agent_hidden
        self.food_hidden = food_hidden
        self.numSteps = 0
        self.agents = []

        self.beam_records = []

        self.numeps = 0

    def terminated(self):
        return any([self.numSteps >= GAME_LENGTH])

    def validActions(self, agent):
        return agent.legal_moves(self)

    def reset(self):
        '''reset game'''
        self.numSteps = 0
        self.beams_set = []

        self.food_objects = []

        # for x in range(14, 19):
        #     delta = x - 14 if x - 14 < 18 - x else 18 - x
        #     self.food_objects.append(Food(coordinates=(x, 5)))
        #     for i in range(delta):
        #         self.food_objects.append(Food(coordinates=(x, 4 - i)))
                # self.food_objects.append(Food(coordinates=(x, 6 + i)))

        for x in range(6, 9):
            delta = x - 6 if x - 6 < 8 - x else 8 - x
            self.food_objects.append(Food(coordinates=(x, 2)))
            for i in range(delta):
                self.food_objects.append(Food(coordinates=(x, 1 - i)))
                self.food_objects.append(Food(coordinates=(x, 3 + i)))

        # assign food location
        # self.food_objects = [Food(coordinates=(4, 1)), Food(coordinates=(4, 3)), Food(coordinates=(7, 2)), Food(coordinates=(10, 1)), Food(coordinates=(10, 3))]

        self.beam_records = []
        for x in range(len(self.agents)):
            self.beam_records.append(0)

        self.numeps += 1

    def execute(self, actions = None):
        '''Run agents simultaneously'''
        old_loc = [[actor.x, actor.y] for actor in self.agents]
        rewards = [0 for actor in self.agents]
        self.beams_set = []

        for idx, agent in enumerate(self.agents):
            agent.check_respawn(self)

            if actions[idx] == 'B' and not self.agents[idx].is_moveout():
                self.beams_set.append(self.agents[idx].beam())
                self.beam_records[idx] += 1
            elif not self.agents[idx].is_moveout():
                self.beams_set.append([])
                self.agents[idx].move(actions[idx])

        # If there's a conflict after execution, move back one agent
        if len(self.agents) > 1:
            if not self.agents[0].is_moveout() and not self.agents[1].is_moveout() and\
                (self.agents[0].x == self.agents[1].x and self.agents[0].y == self.agents[1].y):

                if self.agents[0].x == old_loc[0][0] and self.agents[0].y == old_loc[0][1]:
                    self.agents[1].x = old_loc[1][0]
                    self.agents[1].y = old_loc[1][1]
                elif self.agents[1].x == old_loc[1][0] and self.agents[1].y == old_loc[1][1]:
                    self.agents[0].x = old_loc[0][0]
                    self.agents[0].y = old_loc[0][1]
                else:
                    resolved_one = random.randint(0, len(self.agents) - 1)

                    for idx, agent in enumerate(self.agents):
                        if idx != resolved_one:
                            self.agents[idx].x = old_loc[idx][0]
                            self.agents[idx].y = old_loc[idx][1]

        # Move out agents tagged by beam
        for i, value in enumerate(self.beams_set):
            for idx, agent in enumerate(self.agents):
                if i == idx:
                    pass
                elif (agent.x, agent.y) in value and not agent.is_moveout():
                    agent.tag_mark(self.agent_hidden)

        # Count rewards for agents according to its position
        for food in self.food_objects:
            food.run_hidden()
            if not food.is_hidden():
                for idx, agent in enumerate(self.agents):
                    if not self.agents[idx].is_moveout() and food.x == self.agents[idx].x and food.y == self.agents[idx].y:
                        rewards[idx] = food.collect(self.food_hidden)

        self.numSteps += 1

        return rewards

    def update_hidden_parm(self, agent_hidden, food_hidden):
        '''Update respawn setting'''
        self.agent_hidden = agent_hidden
        self.food_hidden = food_hidden

    def clone(self):
        ''' Clones game instance. Manual copying to prevent deep copying of agent game tree.'''
        g = copy(self)

        g.agents = [copy(a) for a in self.agents]
        g.food_objects = [copy(f) for f in self.food_objects]

        return g

    def state(self, actor_name):
        ''' Returns the point of view of a player as a proxy for imcomplete state information. '''
        for agent in self.agents:
            if agent.name == actor_name:
                return agent.get_state_matrix

    def fullstate(self):
        ''' Perfect information: Returns a tuple of the player and foods locations as a proxy for complete state information. '''
        return tuple([(agent.x, agent.y, agent.direction, agent.mark) for agent in self.agents if not agent.is_moveout()] + [(r.x, r.y) for r in self.food_objects if not r.is_hidden()])

    def get_agent(self, actor_name):
        '''Get agent by name instead if direct reference'''
        for agent in self.agents:
            if agent.name == actor_name:
                return agent