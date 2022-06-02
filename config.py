# Simulation parameters
random_initial = False
# MAP_WIDTH = 33
# MAP_HEIGHT = 11
MAP_WIDTH = 15
MAP_HEIGHT = 5
# VIEW_WIDTH = 21
# VIEW_HEIGHT = 16
VIEW_WIDTH = 9
VIEW_HEIGHT = 7
GAME_LENGTH = 500
NUM_ACTION = 8

is_full_observable = False

svo_value = 1

######### DQN parameters ###########
double_dqn = True
num_eval_episodes = 20
num_train_episodes = 2500#40000
num_test_episodes = 0
eval_freq_in_episodes = 50#5
save_freq_in_episodes = 100
save_model = True

num_train_steps = num_train_episodes * GAME_LENGTH # No. of iterations of training. default: 40000000
num_eval_steps = num_eval_episodes * GAME_LENGTH
eval_freq = eval_freq_in_episodes * GAME_LENGTH           # Episodes before training starts: 10000
save_freq = save_freq_in_episodes * GAME_LENGTH

# Training parameters
train_start = 50000     
batch_size = 64           # Replay memory batch size; DEEP MIND USES 10000
mem_size = 100000           # Replay memory size; recommended 1,000,000
discount = 0.99             # Discount rate (gamma value)  0.99
lr = 0.00025                 # Learning rate: 0.00025  .00005?
lr_actor = 0.00025                 # Learning rate for actor
lr_critic = 0.00025                 # Learning rate for crtic
TAU = 1e-3

eps = 1.0                  # Epsilon start value
eps_initial = 1.0          # Epsilon initial value
eps_final = 0.1            # Epsilon end value
eps_step = mem_size           # Epsilon steps between start and end (linear) for explitation

train_freq = 1
Q_update_freq = 4#10000
recent_frames = 4
n_layer = 3
min_history = 4#4
states_to_update = 1#4

num_lstm_layers = 1
lstm_size = 512

data_format = 'NCHW'#'NHWC'
save_under_user_home = False

########### DPW parameters #############
best_by_visits = False
enable_action_pw = True
DPW_DEPTH = 20
SIMULATION_DEPTH = 20
num_iterations = 10000
exploration_constant = 1.0
action_K_value = 10#3
action_alpha_value = 0.4 #0.5
state_K_value = 10
state_alpha_value = 0.4 #0.5
reward_discount = 1#0.95
max_conflict_counts = 3

########################################
action_reference = 1