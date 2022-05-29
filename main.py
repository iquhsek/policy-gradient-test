import imp


import gym
import matplotlib.pyplot as plt
import numpy as np
from src.agent import Agent
from utils.plot_learning_curve import plot_learning_curve


if __name__ == '__main__':
    agent = Agent(alpha=0.0005, input_dims=8, gamma=0.99, n_actions=4, layer1_size=64, layer2_size=64)
    env = gym.make('LunarLander-v2')
    score_hist = []
    
    n_episodes = 2000
    
    for i in range(n_episodes):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward)
            obs = obs_
            score += reward
        score_hist.append(score)
        
        agent.learn()
        
        print('episode ', i, 'score %.1f' % score,
              'average_score %.1f' % np.mean(score_hist[-100:]))

file_name = 'img/lunar_lander.png'
plot_learning_curve(score_hist, filename=file_name, window=100)