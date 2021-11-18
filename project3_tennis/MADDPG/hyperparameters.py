
BUFFER_SIZE = int(1e4)         # Replay buffer size
BATCH_SIZE = 256               # Minibatch size
GAMMA = 0.95                   # Discount factor
TAU = 0.02                     # Soft update of target parameters
ACTOR_FC1_UNITS = 256          # Number of hidden units in layer 1 in the actor model
ACTOR_FC2_UNITS = 128          # Number of hidden units in layer 2 of the actor model
CRITIC_FC1_UNITS = 256         # Number of hidden units in layer 1 of the critic model
CRITIC_FC2_UNITS = 128         # Number of hidden units in layer 2 of the critic model
ACTOR_LR = 1e-3                # learning rate of the actor 
CRITIC_LR = 1e-3               # learning rate of the critic
WEIGHT_DECAY = 0               # L2 weight decay
DO_GRADIENT_CLIP_CRITIC = True # Gradient clipping of critic parameters
DO_APPLY_BATCH_NORM = True     # Add batch normalisation after first hidden layer of both Actor and Critic NN
LEARN_EVERY = 1                # Learn every N timesteps
N_LEARN = 1                    # When learning, learn N times (must be >= 1)