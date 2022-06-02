# basic experiment setting the the Gathering
# Reset at each run
import sys
import numpy as np
import time

# from Agents.DDQNAgent import DDQNAgent
from Agents.DRQNAgent2 import DRQNAgent2
# from Agents.DRQNAgentV2 import DRQNAgentV2
from Agents.FixedAgent import FixedAgent
from Environment.PreferenceEnvironment import PreferenceEnvironment

from config import *
from Commons.Utils import get_time, spawn_rotation

from csv import writer
import numpy as np
import random

#simulation parameters
visualize = False

DQN_class = DRQNAgent2

def log_games():

    env = PreferenceEnvironment()
    # set initial position and direction
    agents = [(14, 1, "L", "agent1", DRQNAgent2, {"A": 10.0, "O": 1.0}), (14, 2, "L", "agent2", DRQNAgent2, {"A": 1.0, "O": 1.0})]

    ## respawn time setting for simulation- format: (agent, food)
    # sim_setting = [(20, 1), (20, 20), (1, 1), (1, 20)]
    sim_setting = [(10, 10)]

    for i,r_setting in enumerate(sim_setting):
        env = PreferenceEnvironment()
        env.update_hidden_parm(r_setting[0], r_setting[1])

        print('agent spawn: ' + str(r_setting[0]) + 'food spawn: ' + str(r_setting[1]))

        env.agents = []
        
        for i,r_loc in enumerate(agents):
            pos=env.spawn_point()
            # env.agents.append(r_loc[4](coordinates=(r_loc[0], r_loc[1]), name=r_loc[3], agent_respawn = r_setting[0], food_respawn = r_setting[1], direction = r_loc[2], preference=r_loc[5]))
            env.agents.append(r_loc[4](coordinates=(pos[0], pos[1]), name=r_loc[3], agent_respawn = r_setting[0], food_respawn = r_setting[1], direction = spawn_rotation(), preference=r_loc[5]))

        for num in range(num_train_episodes):
            env.reset()
            print(num + 1, 'episode in training')

            for i, value in enumerate(env.agents):
                env.agents[i].initialize(env, agents[i])
                if isinstance(value, DQN_class):
                    env.agents[i].initiate_train()
        
            run_simulation(env, num + 1, r_setting[0], r_setting[1])

            # evaluate model
            if eval_freq_in_episodes and (num + 1) % eval_freq_in_episodes == 0 and num > num_eval_episodes - 1 and (num + 1) > (train_start/GAME_LENGTH):
                print('evaluation...')

                for num2 in range(num_eval_episodes):
                    env.reset()

                    for i, value in enumerate(env.agents):
                        pos=env.spawn_point()
                        # env.agents[i].initialize(env, agents[i])
                        env.agents[i].initialize(env, [pos[0], pos[1], spawn_rotation()])
        
                    run_simulation(env, num2 + 1, r_setting[0], r_setting[1], is_eval = True, is_test = True)


def run_simulation(env, num_simulation, agent_hidden, food_hidden, is_eval = False, is_test = False):
    steps = 0
    count = 0

    try:
        i = 0
        sum_rewards = [0 for a in env.agents]
        count_beam = [0 for a in env.agents]
        count_movedout = [0 for a in env.agents]
        while not env.terminated():

            #simultaneous move
            actions = []
            for idx, actor in enumerate(env.agents):
                
                if not actor.is_moveout():
                    # action = actor.get_action(env, is_test)
                    action = actor.get_action(env, is_test)
                    actions.append(action)

                    count += 0 if action != 'B' else 1

                    if action == 'B':
                        count_beam[idx] += 1

                else:
                    actions.append('W')
                    if not is_test:
                        actor.increase_step()
                    count_movedout[idx] += 1

            rewards, food_types = env.execute(actions)

            if visualize:
                print('steps:', str(steps), actions, rewards)
                print(env.fullstate())

            sum_rewards = [x + y for x, y in zip(sum_rewards, rewards)]

            if steps == GAME_LENGTH - 1:
                for idx, val in enumerate(count_beam):
                    # print(idx, val)
                    count_beam[idx] = val / (GAME_LENGTH - count_movedout[idx])

            for idx, actor in enumerate(env.agents):
                other_reward = rewards[1] if idx == 0 else rewards[0]
                if isinstance(actor, DQN_class):
                    if is_test:
                        actor.evaluate(env, rewards[idx], count_beam[idx], steps == GAME_LENGTH - 1, num_simulation == num_eval_episodes, count_movedout[idx], other_reward, food_types[idx])
                    else:
                        actor.observe(env, rewards[idx], actions[idx], count_beam[idx], count_movedout[idx], other_reward, food_types[idx])




            steps += 1

        # count_beam = [number / 1000 for number in count_beam]

        write_file = True
        general_record_time = get_time()
        log_file = open('./logs/'+ str(general_record_time)+ ('-eval.log' if is_test else '-train.log'),'a')
        log_file.write('# Simulation: ' + str(num_simulation) + '\n')
        log_file.write('# Indivisual aggressiveness: ' + str(count_beam)[1:-1] + '\n')
        log_file.write('# Indivisual movedout: ' + str(count_movedout)[1:-1] + '\n')
        log_file.write('# Rewards: ' + str(sum_rewards)[1:-1] + '\n')
        log_file.write("# Setting:   agent_respawn: " + str(agent_hidden) + " | food_respawn: " + str(food_hidden) + "\n")

        sys.stdout.flush()

    except (KeyboardInterrupt, SystemExit):
        print('KEYBOARD INTERRUPT')
        raise


if __name__ == '__main__':
    if not visualize:
        import timeit
        print(timeit.timeit("log_games()", setup='from __main__ import log_games', number=1))
    else:
        log_games()