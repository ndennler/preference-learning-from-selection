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
        self.N = number_samples
        self.D = number_dimensions
    
    def update(self, prob_input_given_omegas):
        p_input = prob_input_given_omegas(self.hypothesis_samples)
        self.hypothesis_log_probabilities += np.log(p_input)
        
    def reset(self):
        self.hypothesis_log_probabilities = np.zeros(self.N)

    def get_expectation(self):
        unnormalized_probability = np.exp(self.hypothesis_log_probabilities)
        probability = unnormalized_probability/np.sum(unnormalized_probability)
        return np.average(self.hypothesis_samples,
                          axis=0,
                          weights=probability)

    def get_human_entropy(self, query, choice_model):
        choice_probabilities = choice_model.get_choice_probabilities(query, self.hypothesis_samples)
        total_entropy = np.sum(-np.multiply(choice_probabilities, np.log2(choice_probabilities)))
        return total_entropy/len(self.hypothesis_samples)

    def get_best_entropy(self, query, choice_model):
        best_estimate = self.get_expectation()
        choice_probabilities = choice_model.get_choice_probabilities(query, best_estimate)
        entropy = np.sum(-np.multiply(choice_probabilities, np.log2(choice_probabilities)))
        return entropy