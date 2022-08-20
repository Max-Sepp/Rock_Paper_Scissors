from PIL import Image
import cv2
import torch
from torch import nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model parameters
input_size = 19200  # 120x160
num_classes = 3

# transforms to perform on the image same as the ones used on train data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(120),
        transforms.Grayscale(1),
        transforms.Normalize((0.5), (0.5)),
    ]
)

# creating neural net model
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


# classes of output
classes = ["paper", "rock", "scissors"]

# loading the model from memory
model = FeedForwardNeuralNet(input_size=input_size, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("models\pre_trained_models_normalised\epoch_16_72.pt"))
model.eval()


camera_port = int(input("Enter the camera port of the webcam you wish to use"))
while True:
    # wait for user to run the game
    run = str(input("enter nothing to run game otherwise type anything else to escape"))
    if run == "":
        print("rock")
        time.sleep(1)
        print("paper")
        time.sleep(1)
        print("scissors")
        time.sleep(1)

        # take a image from webcam
        camera = cv2.VideoCapture(camera_port)
        result, image = camera.read()
        if result:
            cv2.imwrite("webcam_photo.png", image)
        else:
            print("No image detected")
        cv2.destroyAllWindows()

        # loading the image
        image = Image.open("webcam_photo.png")
        tensor = transform(image)
        img = tensor / 2 + 0.5

        # running the image through the model
        tensor = tensor.reshape(-1, input_size).to(device)
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
        print(classes[predicted])

        # show image to user
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
        plt.show()

        # deleting the taken image
        if os.path.exists("webcam_photo.png"):
            os.remove("webcam_photo.png")
        else:
            print("The file does not exist")
    else:
        break
