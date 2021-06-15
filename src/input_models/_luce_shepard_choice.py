import numpy as np

class LuceShepardChoice():
    def __init__(self):
        self.query = None # [N choice, D dimensions]
        self.choice = None # integer

    def get_input(query):
        self.query = query
        print(query)
        self.choice = int(input('What will it be?'))

    def get_probability_of_input(omega):
        rewards = self.query * omega
        return np.exp(rewards[self.choice]) / np.sum(np.exp(rewards))