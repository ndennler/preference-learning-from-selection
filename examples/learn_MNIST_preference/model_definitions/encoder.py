import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self, nz):
    super().__init__()
    ################################# TODO #########################################
    # Create the network architecture using a nn.Sequential module wrapper.        #
    # All convolutional layers should also learn a bias.                           #
    # HINT: use the given information to compute stride and padding                #
    #       for each convolutional layer. Verify the shapes of intermediate layers #
    #       by running partial networks (with the next cell) and visualizing the   #
    #       output shapes.                                                         #
    ################################################################################
    self.net = nn.Sequential(
        nn.Conv2d(1, 32, 4, stride=2),
        nn.LeakyReLU(),

        nn.Conv2d(32, 64, 4, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),

        nn.Conv2d(64,128,3, stride=2),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),

        nn.Conv2d(128,256,3),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),

        nn.Flatten(),
        nn.Linear(256, nz)
        # add your network layers here
        # ...
    )
    ################################ END TODO #######################################
  
  def forward(self, x):
    return self.net(x)
