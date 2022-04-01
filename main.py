import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import time
from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from wolp import WOLPAgent
from util import *
from ContinuousCartPole import ContinuousCartPoleEnv
def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...        
        
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup:
            agent.update_policy()
        
        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))
class Arguments(object):
    def __init__(self):
        self.mode = 'train'
        self.env = "InvertedPendulum-v2"
        self.hidden1 = 400
        self.hidden2 = 300
        self.rate = 0.001
        self.prate = 0.0001
        self.warmup = 100
        self.discount = 0.99
        self.bsize = 64
        self.rmsize = 6000000
        self.window_length = 1
        self.tau = 0.001
        self.ou_theta = 0.15
        self.ou_sigma = 0.2
        self.ou_mu = 0.0
        self.validate_episodes = 20
        self.max_episode_length = 500
        self.validate_steps = 2000
        self.output = 'output'
        self.debug='debug'
        self.init_w = 0.003
        self.train_iter=20000
        self.epsilon=50000
        self.seed=-1
        self.max_actions=1e6
        self.resume='default'
        self.k_ratio = 1e-6

args = Arguments()
args.output = get_output_folder(args.output, args.env)
if args.resume == 'default':
    args.resume = 'output/{}-run0'.format(args.env)

# env = NormalizedEnv(gym.make(args.env))
# env = gym.make(args.env)
env = ContinuousCartPoleEnv()
#################################### Our Code ##############################
args.low = env.action_space.low
args.high = env.action_space.high
#################################### Our Code ##############################

if args.seed > 0:
    np.random.seed(args.seed)
    env.seed(args.seed)

nb_states = env.observation_space.shape[0]
nb_actions = env.action_space.shape[0]


agent = WOLPAgent(nb_states, nb_actions, args)
evaluate = Evaluator(args.validate_episodes, 
    args.validate_steps, args.output, max_episode_length=args.max_episode_length)

start_time = time.time()

if args.mode == 'train':
    train(args.train_iter, agent, env, evaluate, 
        args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)
    end_time = time.time()

elif args.mode == 'test':
    test(args.validate_episodes, agent, env, evaluate, args.resume,
        visualize=True, debug=args.debug)

else:
    raise RuntimeError('undefined mode {}'.format(args.mode))
