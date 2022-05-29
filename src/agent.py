from tensorflow.python.framework.ops import disable_eager_execution
from argparse import Action
from hashlib import sha1
import imp
from keras.layers import Dense, Activation,Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np


class Agent(object):
    def __init__(self, alpha, gamma=0.99, n_actions=4,
                 layer1_size=16, layer2_size=16, input_dims=128,
                 file_name='model/reinforce.h5') -> None:
        self.lr = alpha
        self.gamma = gamma
        self.G = 0
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
        self.policy, self.predict = self.build_policy_network()
        self.action_space = [i for i in range(n_actions)]
        self.model_file = file_name
    
    def build_policy_network(self):
        disable_eager_execution()
        input_layer = Input(shape=(self.input_dims))
        advantages = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input_layer)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        output_probs = Dense(self.n_actions, activation='softmax')(dense2)
        
        def custom_loss(y, y_hat):
            out = K.clip(y_hat, 1e-8, 1-1e-8)   # we will be using log function which is not defined around zero, so we shall restrict the value of the output with this "clip" function
            log_lik = y * K.log(out)  # log likelihood
            
            return K.sum(-log_lik * advantages)
        
        policy = Model(inputs=[input_layer, advantages], outputs=[output_probs])
        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss, experimental_run_tf_function=False)
        predict = Model(inputs=[input_layer], outputs=output_probs)
        
        return policy, predict
    
    def choose_action(self, obs):
        state = obs[np.newaxis, :]
        pr = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=pr)
        return action
    
    def store_transition(self, obs, action, reward):
        self.state_memory.append(obs)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
    
    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)
        
        actions = np.zeros([len(action_memory), self.n_actions])    # one-hot coding
        actions[np.arange(len(action_memory)), action_memory] = 1
        
        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            
            G[t] = G_sum
        
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean) / std
        
        cost = self.policy.train_on_batch([state_memory, self.G], actions)
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
        return cost
    
    def save_model(self):
        self.policy.save(self.model_file)
    
    def load_model(self):
        self.policy = load_model(self.model_file)
