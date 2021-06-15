import numpy as np

class MonteCarloLinearReward:
    '''
    This class implements a Monte Carlo Simulation of a linear reward
    specification.


    '''

    def __init__(self, 
                 number_dimensions, 
                 number_samples=50_000):
        
        # [N_samples x D_dims]
        self.hypothesis_samples = np.random.uniform(low=-1, 
                                    high= 1,
                                    size=(number_samples, number_dimensions))
        # [N_samples]                                                 
        self.hypothesis_log_probabilities = np.zeros(number_samples)
    
    def update(self, prob_input_given_omega):
        for i, omega in enumerate(self.hypothesis_samples):
            p_input = prob_input_given_omega(omega)
            self.hypothesis_log_probabilities[i] += np.log(p_input)

    def get_expectation(self):
        unnormalize_probability = np.exp(self.hypothesis_log_probabilities)
        probability = unnormalize_probability/np.sum(unnormalize_probability)
        return np.average(self.hypothesis_samples,
                          axis=0,
                          weights=probability)
