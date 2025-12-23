import numpy as np
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done, log_prob, action_mask): 
        self.buffer.append((state, action, reward, next_state, done, log_prob, action_mask)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, log_prob, action_mask = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, log_prob, action_mask 

    def size(self): 
        return len(self.buffer)
    def clear(self):
        self.buffer.clear()
    
    def get(self):
        transitions = list(self.buffer)
        state, action, reward, next_state, done, log_prob, action_mask = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, log_prob, action_mask