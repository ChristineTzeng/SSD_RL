# basic experiment setting the the Gathering
import sys
import numpy as np
import time
import tensorflow as tf
import math

from Agents.Agent import Agent
from Agents.DQNAgent4 import DQNAgent4
from Environment.SimulationEnvironment import SimulationEnvironment

from config import *
from Commons.Utils import get_time

from csv import writer
import numpy as np
import random

#simulation parameters
visualize = False

DQN_class = DRQNAgent3

# set initial position and direction
agents = [(0, 2, "R", "agent1", TargetAgent), (14, 2, "L", "agent2", DRQNAgent3)]
sim_setting = [(20, 20)]

def log_games():

    for i,r_setting in enumerate(sim_setting):
        env = SimulationEnvironment()
        env.update_hidden_parm(r_setting[0], r_setting[1])

        print('agent spawn: ' + str(r_setting[0]) + 'food spawn: ' + str(r_setting[1]))

        env.agents = []
        for i,r_loc in enumerate(agents):
            env.agents.append(r_loc[4](coordinates=(r_loc[0], r_loc[1]), name=r_loc[3], agent_respawn = r_setting[0], food_respawn = r_setting[1], direction = r_loc[2]))

        env.reset()

        for i, value in enumerate(env.agents):
            env.agents[i].initialize(env, agents[i])
            env.agents[i].initiate_train()
        
        run_simulation(env, r_setting[0], r_setting[1])


def run_simulation(env, agent_hidden, food_hidden):
    steps = 0
    count = 0

    try:
        i = 0
        sum_rewards = [0 for a in env.agents]
        count_beam = [0 for a in env.agents]
        count_movedout = [0 for a in env.agents]
        while step < num_train_steps:
            step = step + 1

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

            rewards = env.execute(actions)

            if visualize:
                print('steps:', str(steps), actions, rewards)
                print(env.fullstate())

            sum_rewards = [x + y for x, y in zip(sum_rewards, rewards)]

            if (steps % GAME_LENGTH) == GAME_LENGTH - 1:
                for idx, val in enumerate(count_beam):
                    # print(idx, val)
                    count_beam[idx] = val / (GAME_LENGTH - count_movedout[idx])

            # for idx, actor in enumerate(env.agents):
            #     actor.plot_summary(rewards[idx], count_beam[idx], steps == GAME_LENGTH - 1)

            for idx, actor in enumerate(env.agents):
                if isinstance(actor, DQN_class):
                    if is_test:
                        actor.evaluate(env, rewards[idx], count_beam[idx], steps == GAME_LENGTH - 1, num_simulation == num_eval_episodes)
                    else:
                        actor.observe(env, rewards[idx], actions[idx], count_beam[idx])


            steps += 1


            # statistics
            if step % GAME_LENGTH == 0 or step == num_train_steps:
                print('train step: ', step)
                for idx, val in enumerate(count_beam):
                    count_beam[idx] = val / (GAME_LENGTH - count_movedout[idx])

                for idx, actor in enumerate(env.agents):
                    actor.feedback(env, rewards[idx], count_beam[idx], step % GAME_LENGTH == 0)

                general_record_time = get_time()
               log_file = open('./logs/'+ str(general_record_time)+'-experiment2.log','a')
               log_file.write('# Simulation: ' + str(num_simulation) + '\n')
               log_file.write('# Indivisual aggressiveness: ' + str(count_beam)[1:-1] + '\n')
               log_file.write('# Indivisual movedout: ' + str(count_movedout)[1:-1] + '\n')
               log_file.write('# Rewards: ' + str(sum_rewards)[1:-1] + '\n')
               log_file.write("# Setting:   agent_respawn: " + str(agent_hidden) + " | food_respawn: " + str(food_hidden) + "\n")

                sys.stdout.flush()

                sum_rewards = [0 for a in env.agents]
                count_beam = [0 for a in env.agents]
                count_movedout = [0 for a in env.agents]

                for i, agent in enumerate(env.agents):
                    agent.initiate_train()

            # else:

            #     for idx, actor in enumerate(env.agents):
            #         actor.feedback(env, rewards[idx], count_beam[idx], step % GAME_LENGTH == 0)

            # pos = [(a.x, a.y) for a in env.agents]

                

            # evaluate model
            if step % eval_freq == 0:
                print('evaluating...')
                # eval_env = SimulationEnvironment()
                eval_env = env.clone()
                for i,r_setting in enumerate(sim_setting):
                    eval_env.update_hidden_parm(r_setting[0], r_setting[1])

                eval_env.reset()

                for i, agent in enumerate(eval_env.agents):
                    agent.initialize(eval_env, agents[i])
                    agent.initiate_eval(True)

                eval_rewards = [0 for a in eval_env.agents]
                eval_beam = [0 for a in eval_env.agents]
                eval_movedout = [0 for a in eval_env.agents]

                for eval_step in range(1, num_eval_steps + 1):
                    #simultaneous move
                    actions = []
                    for idx, actor in enumerate(eval_env.agents):
                        actor.observe(eval_env)
                        if not actor.is_moveout():
                            action = actor.get_action(eval_env, True)
                            actions.append(action)

                            if action == 'B':
                                eval_beam[idx] += 1

                        else:
                            actions.append('W')
                            eval_movedout[idx] += 1

                    if visualize:
                        print('steps:', str(eval_step + 1), actions)

                    rewards = eval_env.execute(actions)

                    eval_rewards = [x + y for x, y in zip(eval_rewards, rewards)]

                    # statistics
                    if eval_step % GAME_LENGTH == 0 or eval_step == num_eval_steps:
                        # print('eval step: ', eval_step)
                        for idx, val in enumerate(eval_beam):
                            eval_beam[idx] = val / (GAME_LENGTH - eval_movedout[idx])

                        for idx, actor in enumerate(eval_env.agents):
                            score = actor.eval_model(eval_env, rewards[idx], eval_beam[idx], num_eval_episodes, eval_step % GAME_LENGTH == 0, math.ceil(eval_step/GAME_LENGTH) == num_eval_episodes, best_score_history[idx], True)
                            best_score_history[idx] = score

                        general_record_time = get_time()
                        log_file = open('./logs/'+ str(general_record_time)+'-eval.log','a')
                        log_file.write('# Simulation step: ' + str(step) + ' | eval step: ' + str(eval_step) + '\n')
                        log_file.write('# Indivisual aggressiveness: ' + str(eval_beam)[1:-1] + '\n')
                        log_file.write('# Indivisual movedout: ' + str(eval_movedout)[1:-1] + '\n')
                        log_file.write('# Rewards: ' + str(eval_rewards)[1:-1] + '\n')

                        sys.stdout.flush()

                        eval_rewards = [0 for a in eval_env.agents]
                        eval_beam = [0 for a in eval_env.agents]
                        eval_movedout = [0 for a in eval_env.agents]

                        for i, agent in enumerate(eval_env.agents):
                            agent.initiate_eval(False)

                    else:
                        for idx, actor in enumerate(eval_env.agents):
                            actor.eval_model(eval_env, rewards[idx], eval_beam[idx], num_eval_episodes, eval_step == num_eval_steps, math.ceil(eval_step/GAME_LENGTH) == num_eval_episodes, best_score_history[idx], False)

                    # print(eval_step, step, num_eval_steps)


    except (KeyboardInterrupt, SystemExit):
        print('KEYBOARD INTERRUPT')
        tf.InteractiveSession.close()
        raise


if __name__ == '__main__':
    if not visualize:
        import timeit
        print(timeit.timeit("log_games()", setup='from __main__ import log_games', number=1))
    else:
        log_games()