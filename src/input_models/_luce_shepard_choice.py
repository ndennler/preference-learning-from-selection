import numpy as np

class LuceShepardChoice():
    def __init__(self):
        self.query = None # [N choice, D dimensions]
        self.choice = None # integer
        pass

    def get_probability_of_input(self, omega):
        rewards = np.dot(self.query , omega.T)
        return np.exp(rewards[self.choice]) / np.sum(np.exp(rewards))

    def tell_input(self, choice, query):
        self.choice = choice
        self.query = query

    def get_choice_probabilities(self, options, omegas):
        '''
        options - N options by D dimensions
        omegas - S omegas by D dimensions

        returns N options by S omegas (where each [:, s] is the probability
            distribution over the choices for the sth omega)
        '''
        rewards = np.dot(options, omegas.T) # N options by S omegas of the rewards for each option
        return np.exp(rewards) / np.sum(np.exp(rewards), axis=0)