from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(35, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 400),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(400, 200),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(200, 1),
        )

    def forward(self, x):
        return self.layers(x)