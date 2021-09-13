import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.input_models import LuceShepardChoice, WeakPreferenceChoice
from src.query_generation import InfoGainQueryGenerator, RandomQueryGenerator, VolumeRemovalQueryGenerator
from src.reward_parameterizations import MonteCarloLinearReward

def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))

#Experimental Constants
dim_embedding = 3
true_preference = np.random.uniform(low=-1, high=1, size=dim_embedding)
number_of_trials = 20
max_number_of_queries = 30

#User Input and Estimation of reward functions
user_choice_model = LuceShepardChoice()
user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=10_000)

#Generators
random_generator = RandomQueryGenerator( [(-1,1)] * dim_embedding)
vr_generator = VolumeRemovalQueryGenerator( [(-1,1)] * dim_embedding)
ig_generator = InfoGainQueryGenerator([(-1,1)] * dim_embedding)

generators = [random_generator, vr_generator, ig_generator]
names = ['Random', 'Volume Removal', 'Information Gain']

for generator, name in zip(generators, names):
    cumulative_values = []

    for _ in tqdm(range(number_of_trials)):

        user_estimate.reset()
        true_preference = np.random.uniform(low=-1, high=1, size=dim_embedding)
        alignment = [0]

        for _ in range(max_number_of_queries):
            query = generator.get_query(2, user_estimate, user_choice_model) #generates choice between two options
            choice = np.argmax(user_choice_model.get_choice_probabilities(query, np.array([true_preference]))) #selects choice from model
            user_choice_model.tell_input(choice, query)
            user_estimate.update(user_choice_model.get_probability_of_input)   
            alignment.append(alignment_metric(user_estimate.get_expectation(), true_preference))

        cumulative_values += [alignment]

    m = np.mean(np.array(cumulative_values), axis=0) 
    std = np.std(np.array(cumulative_values), axis=0) 
    plt.fill_between(range(max_number_of_queries+1), m-std, m+std, alpha=0.3)
    plt.plot(m, label=name)

plt.title('Alignment Scores by Methodology')
plt.xlabel('Number of Queries')
plt.ylabel('Alignment')
plt.legend()
plt.show()