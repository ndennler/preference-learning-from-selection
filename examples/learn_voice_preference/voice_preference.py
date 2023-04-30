from sample_voices import VoiceSampler
import wavio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from irlpreference.input_models import LuceShepardChoice, WeakPreferenceChoice
from irlpreference.reward_parameterizations import MonteCarloLinearReward


#initialize the voice sampler
voice_sampler = VoiceSampler()
print('voice sampler initialized!')

if __name__ == '__main__':
    #Experimental Constants
    dim_embedding = voice_sampler.speaker_vector_array.shape[1]
    true_preference = np.random.uniform(low=-1, high=1, size=dim_embedding)
    max_number_of_queries = 10
    number_of_options = 2

    #User Input and Estimation of reward functions
    user_choice_model = LuceShepardChoice()
    user_choice_model = WeakPreferenceChoice(.1)

    user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=20_000)

    for _ in range(max_number_of_queries):
        query = []

        wav, fs, spembs = voice_sampler.get_audio_sample_and_vector('this is voice one, what do you think?')
        query.append(spembs)
        wavio.write("data/sample1.wav", wav, fs, sampwidth=2)
        

        wav, fs, spembs = voice_sampler.get_audio_sample_and_vector('this is just a test. I am voice 2!')
        query.append(spembs)
        wavio.write("data/sample2.wav", wav, fs, sampwidth=2)

        choice = input('type 1 for voice one or 2 for voice two')
        choice = int(choice) - 1

        user_choice_model.tell_input(choice, query)
        user_estimate.update(user_choice_model.get_probability_of_input)

voice_sampler.generate_good_utterances(user_estimate.get_expectation(), 10)

