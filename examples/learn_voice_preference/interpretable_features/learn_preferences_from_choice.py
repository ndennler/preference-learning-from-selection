from generate_data import VoiceSampler
import wavio
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from src.input_models import LuceShepardChoice, WeakPreferenceChoice
from src.reward_parameterizations import MonteCarloLinearReward


#initialize the voice sampler
voice_sampler = VoiceSampler()
print('voice sampler initialized!')

if __name__ == '__main__':
    voice_data = pd.read_csv('voice_info.csv')
    #Experimental Constants
    print(voice_data.shape)
    dim_embedding = voice_data.shape[1] - 2 #subtract name and index columns
    max_number_of_queries = 10 #number of times we choose between voices
    number_of_options = 2 #number of voices to choose between
    fs = 24_000 #frames per second for audio samples

    #User Input and Estimation of reward functions
    user_choice_model = LuceShepardChoice()
    user_choice_model = WeakPreferenceChoice(.1)

    user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=20_000)

    for _ in range(max_number_of_queries):
        query = []

        random_row = voice_data.sample()
        wav = voice_sampler.gen_audio('this is voice one, what do you think?', random_row['name'].item())
        query.append(np.squeeze(random_row[['speed', 'avg_f0', 'std_f0']].values))
        wavio.write("data/sample1.wav", wav, fs, sampwidth=2)
        
        random_row = voice_data.sample()
        wav = voice_sampler.gen_audio('this is voice two. Do you like me better?', random_row['name'].item())
        query.append(np.squeeze(random_row[['speed', 'avg_f0', 'std_f0']].values))
        wavio.write("data/sample2.wav", wav, fs, sampwidth=2)

        print(query)
        choice = input('type 1 for voice one or 2 for voice two')
        choice = int(choice) - 1

        user_choice_model.tell_input(choice, query)
        user_estimate.update(user_choice_model.get_probability_of_input)

omega = user_estimate.get_expectation()
print(f'Best estimate of omega is {omega}')
