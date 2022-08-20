from PIL import Image
import torch
from torch import nn
from torch.nn import functional
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_size = 19200  # 120x160
num_classes = 3


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(120),
        transforms.Grayscale(1),
        transforms.Normalize((0.5), (0.5)),
    ]
)


class FeedForwardNeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FeedForwardNeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, num_classes)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.Relu(x)
        x = self.linear3(x)
        # no activation or softmax used
        return x


classes = ["paper", "rock", "scissors"]


model = FeedForwardNeuralNet(input_size=input_size, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("pre_trained_models_normalised\epoch_14_70.pt"))
model.eval()


image_paths = [
    "test_1660983815873138200.png",
    "test_1660983822457915900.png",
    "test_1660983834900045400.png",
    "test_1660983885367381600.png",
    "test_1660983891894835100.png",
    "test_1660983897608739900.png",
]

results = []

for path in image_paths:
    image = Image.open(path)
    tensor = transform(image)
    npimg = tensor.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    tensor = tensor.reshape(-1, input_size).to(device)

    output = model(tensor)
    _, predicted = torch.max(output.data, 1)

    # print(output)
    print(classes[predicted])
    results.append(classes[predicted])

print(results)
print(["rock", "paper", "scissors", "rock", "scissors", "paper"])
