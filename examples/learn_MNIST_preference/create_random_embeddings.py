import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from tqdm import tqdm
import pandas as pd


# this will automatically download the MNIST training set
mnist_train = torchvision.datasets.MNIST(root='./data', 
                                         train=True, 
                                         download=True, 
                                         transform=torchvision.transforms.ToTensor())
print("\n Download complete! Downloaded {} training examples!".format(len(mnist_train)))

# Let's display some of the training samples.
sample_images = []
mnist_it = iter(mnist_train)  # create simple iterator, later we will use proper DataLoader
for _ in range(5):
  sample = next(mnist_it)     # samples a tuple (image, label)
  sample_images.append(sample[0][0].data.cpu().numpy())

fig = plt.figure(figsize = (10, 50))   
ax1 = plt.subplot(111)
ax1.imshow(np.concatenate(sample_images, axis=1), cmap='gray')
# plt.show()



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # use GPU if available
nworkers = 4        # number of wrokers used for efficient data loading
batch_size = 128

mnist_data_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=False, num_workers=nworkers)

# now we can run a forward pass for encoder and decoder and check the produced shapes
from model_definitions.encoder import Encoder
nz = 256  # dimensionality of the learned embedding
encoder = Encoder(nz).to(device) #randomly initialized encoder


print(len(mnist_train[0]))
im_data = np.zeros([len(mnist_data_loader)*batch_size, 28, 28])
em_data = np.zeros([len(mnist_data_loader)*batch_size, nz])
label_data = np.zeros([len(mnist_data_loader)*batch_size])

index = 0

for sample_img, sample_label in tqdm(mnist_data_loader):
    enc = encoder(sample_img.to(device)) 
    
    for i, (image, embedding, label) in enumerate(zip(sample_img, enc, sample_label)):
        im_data[i + index, :,:] = torch.squeeze(image).numpy()
        em_data[i + index, :] = embedding.cpu().detach().numpy()
        label_data[i + index] = label.numpy()

    index += i

np.savez_compressed('./data/embeddings', images=im_data, embeddings=em_data, labels=label_data)
