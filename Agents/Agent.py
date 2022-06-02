import random
from config import MAP_WIDTH, MAP_HEIGHT, random_initial
from Commons.Utils import legal_moves

# import arcade

class Agent(object):
    # def __init__(self, coordinates, type, name, width, height, direction='L', is_tag=False, tic_time=0):
    # def __init__(self, coordinates, name, direction='L', color, mark=0, tic_time=0):
    # def __init__(self, coordinates, name, direction, color, mark=0, tic_time=0):
    def __init__(self, coordinates, name, direction, mark=0, tic_time=0):
        self.name = name
        self.tic_time = tic_time
        self.mark = mark

        self.env_x_size = MAP_WIDTH
        self.env_y_size = MAP_HEIGHT

        self.num_steps = 0
        # self.color = color

        # self.respawn_loc = [[0, 2, 'R'], [14, 2, 'L']]
        self.respawn_loc = [[14, 1, 'L'], [14, 2, 'L']]

        if random_initial:
            selected = random.choice(self.respawn_loc)
            self.x = selected[0]
            self.y = selected[1]
            self.direction = selected[2]
        else:
            self.x = coordinates[0]
            self.y = coordinates[1]
            self.direction = direction


    def initialize(self, env, coordinates):
        pass

    def legal_moves(self, env):
        return legal_moves()

    def get_action(self, env):
        self.num_steps += 1

        ''' random action '''
        return random.choice(list(self.legal_moves(env)))

    def move(self, action):
        '''Operation that change the agent's state/location'''

        if action == 'W':
            return
        elif action == 'MF':
            cache_x, cache_y = self.forward_direction_cache()
            self.x = self.x + cache_x if self.x + cache_x >=0 and self.x + cache_x <= self.env_x_size - 1 else self.x
            self.y = self.y + cache_y if self.y + cache_y >=0 and self.y + cache_y <= self.env_y_size - 1 else self.y
        elif action == 'MB':
            cache_x, cache_y = self.forward_direction_cache()
            self.x = self.x - cache_x if self.x - cache_x >=0 and self.x - cache_x <= self.env_x_size - 1 else self.x
            self.y = self.y - cache_y if self.y - cache_y >=0 and self.y - cache_y <= self.env_y_size - 1 else self.y
        elif action == 'ML':
            cache_x, cache_y = self.move_left_cache()
            self.x = self.x + cache_x if self.x + cache_x >=0 and self.x + cache_x <= self.env_x_size - 1 else self.x
            self.y = self.y + cache_y if self.y + cache_y >=0 and self.y + cache_y <= self.env_y_size - 1 else self.y
        elif action == 'MR':
            cache_x, cache_y = self.move_left_cache()
            self.x = self.x - cache_x if self.x - cache_x >=0 and self.x - cache_x <= self.env_x_size - 1 else self.x
            self.y = self.y - cache_y if self.y - cache_y >=0 and self.y - cache_y <= self.env_y_size - 1 else self.y
        elif action == 'RL':
            self.turn_left()
        elif action == 'RR':
            self.turn_right()
            
        return self.x, self.y

    def reset_observation(self, env):
        return 0

    def is_moveout(self):
        '''If agent is moved out from the game'''
        return True if self.tic_time > 0 else False

    def check_respawn(self, env):
        '''Countdown the respawn time'''

        if self.tic_time == 1:
            loc = random.choice(self.respawn_loc)
            self.x = loc[0]
            self.y = loc[1]
            self.direction = loc[2]
            self.reset_observation(env)

        self.tic_time -= 1
        self.tic_time = 0 if self.tic_time <= 0 else self.tic_time

        return self.tic_time

    def tag_mark(self, agent_hidden):
        '''Mute or move out the agent while being tagged twice'''
        self.mark += 1
        if self.mark >= 2:
            self.mark = 0
            self.tic_time = agent_hidden
        return self.mark

    def direction_mark(self):
        if self.direction == 'R':
            x = self.x + 1 if self.x + 1 <= self.env_x_size - 1 else -1
            y = self.y
        elif self.direction == 'U':
            x = self.x
            y = self.y - 1 if self.y - 1 >= 0 else -1
        elif self.direction == 'L':
            x = self.x - 1 if self.x - 1 >= 0 else -1
            y = self.y
        elif self.direction == 'D':
            x = self.x
            y = self.y + 1 if self.y + 1 <= self.env_y_size - 1 else -1
        return x, y

    def forward_direction_cache(self):
        '''Information for move'''
        if self.direction == 'R':
            cache_x, cache_y = 1, 0
        elif self.direction == 'U':
            cache_x, cache_y = 0, -1
        elif self.direction == 'L':
            cache_x, cache_y = -1, 0
        elif self.direction == 'D':
            cache_x, cache_y = 0, 1
        else:
            print('wrong direction')

        return cache_x, cache_y

    def move_left_cache(self):
        '''Information for move'''
        if self.direction == 'R':
            cache_x, cache_y = 0, -1
        elif self.direction == 'U':
            cache_x, cache_y = -1, 0
        elif self.direction == 'L':
            cache_x, cache_y = 0, 1
        elif self.direction == 'D':
            cache_x, cache_y = 1, 0
        else:
            print('wrong direction')

        return cache_x, cache_y

    def turn_left(self):
        '''Information for move'''
        if self.direction == 'R':
            self.direction = 'U'
        elif self.direction == 'U':
            self.direction = 'L'
        elif self.direction == 'L':
            self.direction = 'D'
        elif self.direction == 'D':
            self.direction = 'R'

    def turn_right(self):
        '''Information for move'''
        if self.direction == 'R':
            self.direction = 'D'
        elif self.direction == 'U':
            self.direction = 'R'
        elif self.direction == 'L':
            self.direction = 'U'
        elif self.direction == 'D':
            self.direction = 'L'

    def beam(self):
        '''Return the matrix of the beam'''
        if self.direction == 'R':
            beam_set = [(i + 1, self.y) for i in range(self.x, self.env_x_size - 1)]
        elif self.direction == 'U':
            beam_set = [(self.x, i - 1) for i in range(self.y, 0, -1)]
        elif self.direction == 'L':
            beam_set = [(i - 1, self.y) for i in range(self.x, 0, -1)]
        elif self.direction == 'D':
            beam_set = [(self.x, i + 1) for i in range(self.y, self.env_y_size - 1)]
        else:
            print('wrong direction')
        return beam_set

    def get_action_value(self, action):
        if action == 'W':
            return 0
        elif action == 'MF':
            return 1
        elif action == 'MB':
            return 2
        elif action == 'ML':
            return 3
        elif action == 'MR':
            return 4
        elif action == 'RL':
            return 5
        elif action == 'RR':
            return 6
        elif action == 'B':
            return 7
        else:
            print('wrong key')
            return -1
    
    def get_action_keyword(self, action):
        if action == 0:
            return 'W'
        elif action == 1:
            return 'MF'
        elif action == 2:
            return 'MB'
        elif action == 3:
            return 'ML'
        elif action == 4:
            return 'MR'
        elif action == 5:
            return 'RL'
        elif action == 6:
            return 'RR'
        elif action == 7:
            return 'B'
        else:
            print('wrong value')
            return ''

    def increase_step(self):
        '''increase step still when agent was tagged and removed'''
        self.num_steps += 1

    def get_nearest_food(self, env, x, y):
        '''Get the nearest food from an agent'''
        foods = [f for f in env.food_objects if not f.is_hidden()]
        distance = 99999
        temp = 0
        target = None

        for food in foods:
            temp = pow(y - food.y, 2) + pow(x - food.x, 2)
            if temp < distance:
                target = food
                distance = temp

        return target

    def teleport(self, x, y):
        self.x = x
        self.y = y