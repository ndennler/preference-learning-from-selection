import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from irlpreference.input_models import LuceShepardChoice, WeakPreferenceChoice
from irlpreference.query_generation import RandomQueryGenerator
from irlpreference.reward_parameterizations import MonteCarloLinearReward

def get_user_input(query_images, best_image, number_of_options, goal):
    valid = False
    while not valid:
        fig = plt.figure(figsize = (20, 50))   
        plt.title(f'select the one that looks most like {goal}')
        ax1 = plt.subplot(111)
        ax1.imshow(np.concatenate([np.concatenate(query_images, axis=1), np.concatenate([np.zeros((28,28)), best_image], axis=1)], axis=1), cmap='gray')
        
        pts = np.asarray(plt.ginput(1, timeout=-1))
        choice = int(pts[0][0]//28)

        if choice < number_of_options + 1:
            valid=True
    plt.close(fig)
    return choice

if __name__ == '__main__':
    data = np.load('./data/embeddings.npz')
    images = data['images'][:10000]
    embeddings = data['embeddings'][:10000]
    labels = data['labels'][:10000]

    #Experimental Constants
    dim_embedding = embeddings.shape[1]
    true_preference = np.random.uniform(low=-1, high=1, size=dim_embedding)
    max_number_of_queries = 15
    number_of_options = 2
    random_number_to_select = np.random.randint(0,10)

    #User Input and Estimation of reward functions
    user_choice_model = LuceShepardChoice()
    user_choice_model = WeakPreferenceChoice(.1)

    user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=20_000)

    #Generators
    random_generator = RandomQueryGenerator( [(-1,1)] * dim_embedding)

    for _ in range(max_number_of_queries):
        # query = random_generator.get_query(number_of_options, user_estimate, user_choice_model) #generates choice between two options
        # choice = np.argmax(user_choice_model.get_choice_probabilities(query, np.array([true_preference]))) #selects choice from model
        random_indices = np.random.choice(images.shape[0], size=number_of_options)
        query = embeddings[random_indices , :]

        best_so_far_index = np.argmax(np.matmul(embeddings, np.expand_dims(user_estimate.get_expectation(), 1)))


        print(random_indices, best_so_far_index)
        choice = get_user_input(images[random_indices], images[best_so_far_index], number_of_options, random_number_to_select)


        user_choice_model.tell_input(choice, query)
        user_estimate.update(user_choice_model.get_probability_of_input)   

n_row, n_col = 10,10
num_pictures = n_row*n_col

rewards = np.squeeze(np.matmul(embeddings, np.expand_dims(user_estimate.get_expectation(), 1)))
print(rewards.shape)
samples = np.argpartition(rewards, -num_pictures)[-num_pictures:]
imgs = images[samples]
lbls = labels[samples]

_, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
axs = axs.flatten()
for img, ax in zip(imgs, axs):
    ax.imshow(img, cmap='gray')

print(np.sum(lbls == random_number_to_select), embeddings.shape[1])


with open("./data/dimension_data.csv", "a") as a_file:
  a_file.write(f'{embeddings.shape[1]}, {np.sum(lbls == random_number_to_select)}, {random_number_to_select}\n')

plt.show()