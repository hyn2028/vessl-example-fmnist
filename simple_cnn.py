import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, cls_hidden=128, dropout=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(7,7))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(7,7))

        self.layer_conv = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self.conv2,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(dropout),
        )

        self.layer_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, cls_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, 10)
        )

    def forward(self, x):
        x = self.layer_conv(x)
        logits = self.layer_classifier(x)
        return logits # return raw output of NN (logits, NOT PROBABILITY)