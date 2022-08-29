from torch import nn

class LinearModel(nn.Module):

    def __init__(self, nclasses=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(34, 32), nn.LeakyReLU(),
            nn.Linear(32, 16), nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(16, nclasses)
        )


    def forward(self, x):
        return self.net(x)