---
title: Building an AI Game Bot Using Imitation Learning and 3D Convolution ResNet
author: Akshay Ballal
stage: live
image: https://dev-to-uploads.s3.amazonaws.com/uploads/articles/8xnt8ktf7op24obtqnps.jpg
description: Unlocking the Secrets of AI Gaming with Imitation Learning, Motion Analysis, and Google Snake - Learn, Play, and Train Without the Hassle of Building Your Own Game
date: 01/02/2024
---
![Cover](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/29jy0lkg0lj3pi1q5bfv.png)

## Introduction

Hey guys, this article is going to be fun, and I am sure you will get to learn a lot about using AI in a fun way. Even though this article is about making an AI engine to play a game, the learnings can be used in more serious applications because the process is more or less the same. The good thing about learning to build AI with games, as seen in my previous [article](https://dev.to/akshayballal/how-to-build-an-ai-powered-game-bot-with-pytorch-and-efficientnet-18jm), is that you get to experiment a lot without consequences, and getting the data is cheap. Normally, for games, one would use reinforcement learning to train an agent. This can be done but requires a lot more effort to set up in regards to building your own simulator. But with this one, we are going to build an AI agent that plays the Google Snake game. You don’t have to build the game yourself. 

Broadly, this is what we are going to do. We are going to use the technique of imitation learning, where the agent learns from the human counterpart about how to make decisions. This means that we as humans are first going to play the game for a while and collect the data. And then, the AI uses this data to train itself on how to play the game and pick up patterns from the human counterpart. This is known as imitation learning (not to be confused with transfer learning). 

Getting into the technicalities, we are going to train a 3D convolution ResNet model. This is because we want to capture motion to know which direction the snake is heading. For this reason, we are going to feed our model 4 frames of the game at a time to infer motion. You could try just to use one frame and use a standard convolution model like EfficientNet, but it is not going to work that well without the motion information. 

Without further ado, let’s get started. First, we will see how to collect the data.

## Data Collection

There are two methods to gather data. The first involves using a Python script to capture the game screen and label the images with the keyboard command. The second method, which we will be using, relies on selenium, a Python tool that automates browser navigation and control. This is the code to use selenium and save the screen captures as images with the appropriate labels.

We can create a python script `capture.py` to collect the screenshots. 

```python
# Import the required modules
import base64
import io
import cv2
from PIL import Image
import numpy as np
import keyboard
import os
from datetime import datetime
from selenium import webdriver

from selenium.webdriver.common.by import By

# Check if the captures folder exists and delete any existing files in it
isExist = os.path.exists("captures")

if isExist:
    dir = "captures"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

else:
    os.mkdir("captures")

current_key = "1"
buffer = []

# Define a function to record the keyboard inputs
def keyboardCallBack(key: keyboard.KeyboardEvent):
    global current_key

    # If a key is pressed and not in the buffer, add it to the buffer
    if key.event_type == "down" and key.name not in buffer:
        buffer.append(key.name)

    # If a key is released, remove it from the buffer
    if key.event_type == "up":
        buffer.remove(key.name)

    # Sort the buffer and join the keys with spaces
    buffer.sort()
    current_key = " ".join(buffer)

# Hook the function to the keyboard events
keyboard.hook(callback=keyboardCallBack)

# Create a webdriver instance using Firefox
driver = webdriver.Firefox()
# Navigate to the Google Snake game website
driver.get("https://www.google.com/fbx?fbx=snake_arcade")

# Loop indefinitely
while True:
    # Find the canvas element on the webpage
    canvas = driver.find_element(By.CSS_SELECTOR, "canvas")

    # Get the base64 encoded image data from the canvas
    canvas_base64 = driver.execute_script(
        "return arguments[0].toDataURL('image/png').substring(21);", canvas
    )
    # Decode the base64 data to get the PNG image
    canvas_png = base64.b64decode(canvas_base64)

    # Convert the PNG image to a grayscale numpy array
    image = cv2.cvtColor(
        np.array(Image.open(io.BytesIO(canvas_png))), cv2.COLOR_BGR2GRAY
    )

    # Save the image to the captures folder with the current timestamp and keyboard inputs as the file name
    if len(buffer) != 0:
        cv2.imwrite(
            "captures/"
            + str(datetime.now()).replace("-", "_").replace(":", "_").replace(" ", "_")
            + " "
            + current_key
            + ".png",
            image,
        )
    else:
        cv2.imwrite(
            "captures/"
            + str(datetime.now()).replace("-", "_").replace(":", "_").replace(" ", "_")
            + " n"
            + ".png",
            image,
        )
```

Once you run this script, you can start playing the game. In the background, the script will keep saving screenshots of the game screen and name the images with a unique timestamp and the key being pressed right now. When no key is pressed, it is marked as `n`. 

This is how the directory will look after the images are saved.

![Image Directory](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/zvnjxtft8gdpfrxqp844.png)

## Convert Image Folder to CSV File

Now we can convert these images into a csv file with the file names and the corresponding actions. We do this in a new file `process.ipynb`.

```jsx
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv 
import os

labels = []
dir = 'captures'
file_path = "data/labels_snake.csv"

if not os.path.exists(file_path):
    os.mkdir('data')

for f in os.listdir(dir):
    key = f.rsplit('.',1)[0].rsplit(" ",1)[1]
    
    if key=="n":
        labels.append({'file_name': f, 'class': 0})
    elif key=="left":
        labels.append({'file_name': f, 'class': 1})
    elif key=="up":
        labels.append({'file_name': f, 'class': 2})
    elif key=="right":
        labels.append({'file_name': f, 'class': 3})
    elif key=="down":
        labels.append({'file_name': f, 'class': 4})
    

field_names= ['file_name', 'class']

with open('data/labels_snake.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(labels)
```

In this process, we are essentially creating a dataset of images with their corresponding labels, where each label represents the direction in which a key was pressed. Here are the steps involved:

1. **Importing dependencies:** We need certain libraries and modules to read and process the image files and write the labels to a CSV file. These are imported at the beginning of the process.
2. **Creating a directory**: We create a directory named "data" to save the CSV file that will contain the labels and image filenames.
3. **Reading the file nam**e: We extract the key pressed from the file name of each image. The key pressed can be either left, right, up, down, or no key at all.
4. **Classifying the images**: Based on the key pressed, we classify each image into one of four classes: 0 for no key being pressed, 1 for left, 2 for up, 3 for right, and 4 for down. This information is stored along with the filename of the image in a list called "labels".
5. **Writing the labels**: Finally, we write the labels to the CSV file. This dataset can now be used to train a machine learning model to recognize the direction in which a key is pressed, given an image.

Now, we create a new file `snake_resnet.ipynb`.

## Import Dependencies

```python
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision.transforms import transforms, Compose, ToTensor, Resize, Normalize, CenterCrop, Grayscale
from torch import nn
from tqdm import tqdm
from torchinfo import summary
import numpy as np
import math
from torchvision.models.video import r3d_18, R3D_18_Weights, mc3_18, MC3_18_Weights
```

## Create Dataset

To get started, we need to create a custom dataset object using PyTorch. Our dataset consists of a stack of four images that are arranged in chronological order. Each item drawn from the dataset represents a sequence of four frames, where the last frame is associated with a key press. Essentially, this dataset captures motion through the last four frames and associates it with a key press.

```python
class SnakeDataSet(Dataset):
    def __init__(self, dataframe, root_dir, stack_size, transform = None):
        self.stack_size = stack_size
        self.key_frame = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_frame) - self.stack_size *3

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        try:
            img_names = [os.path.join(self.root_dir, self.key_frame.iloc[idx + i, 0]) for i in range(self.stack_size)]
            images = [Image.open(img_name) for img_name in img_names]
            label = torch.tensor(self.key_frame.iloc[idx + self.stack_size, 1])
            if self.transform:
                images = [self.transform(image) for image in images]
        except:
            img_names = [os.path.join(self.root_dir, self.key_frame.iloc[0 + i, 0]) for i in range(self.stack_size)]
            images = [Image.open(img_name) for img_name in img_names]
            label = torch.tensor(self.key_frame.iloc[0 + self.stack_size, 1])
            if self.transform:
                images = [self.transform(image) for image in images]
        return torch.stack(images,dim = 1).squeeze(), label
```

Let's break down the code:

1. **Initialization (`__init__` method):**
    - Parameters:
        - **`dataframe`**: The dataset containing information about images and labels.
        - **`root_dir`**: The root directory where the image files are located.
        - **`stack_size`**: The number of images to be stacked together as a single data point.
        - **`transform`**: An optional parameter for image transformations (e.g., data augmentation).
2. **Length method (`__len__` method):**
    - Returns the length of the dataset, which is the total number of data points. The length is calculated as the length of **`key_frame`** minus three times the **`stack_size`**. This indicates that the dataset is expected to contain sequences of images, and each data point consists of a stack of images.
3. **Get item method (`__getitem__` method):**
    - Takes an index **`idx`** and returns the corresponding data point.
    - The code attempts to load a sequence of images starting from the index **`idx`**. It constructs a list of image file paths (**`img_names`**) based on the specified **`stack_size`** and **`root_dir`**.
    - It then opens each image using the Python Imaging Library (PIL) and stores the resulting image objects in a list (**`images`**).
    - The label for the data point is extracted from the **`key_frame`** dataframe.
    - If an image transformation function (**`transform`**) is provided, it applies the transformation to each image in the sequence.
    - The images are stacked along a new dimension (dimension 1) using **`torch.stack(images, dim=1)`**, and then the singleton dimension is removed using **`squeeze()`**. This results in a tensor representing the stacked images.
    - The stacked images tensor and the label tensor are returned as a tuple.
    - Note: There's a **`try-except`** block that catches potential errors, and if an error occurs, it falls back to using the first sequence in the dataset. This is a basic form of error handling, but it's often better to handle errors more explicitly depending on the use case.

## Balancing the Dataset and creating Dataloader

If you analyze the dataset closely, you will be able to see that there is a severe class imbalance in the dataset. This is because most of the time, while playing the game, the user is not pressing any key. Thus, most of the dataset items belong to class 0.

![Histogram](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/z4ptbb0yc6gt74ut3aun.png)

We fix this by using a PyTorch `RandomWeightedSampler` that takes as input the weights of each sample. We calculate these weights using the code below. 

```
STACK_SIZE = 4
BATCH_SIZE = 32

train, test = train_test_split(pd.read_csv("data/labels_snake.csv"), test_size=0.2, shuffle=False)
classes = ["n", "left", "up", "right", "down"]

labels_unique, counts = np.unique(train["class"], return_counts=True)
class_weights = [sum(counts)/c for c in counts]
example_weights = np.array([class_weights[l] for l in train['class']])
example_weights = np.roll(example_weights, -STACK_SIZE)
sampler = WeightedRandomSampler(example_weights, len(train))

labels_unique, counts = np.unique(test["class"], return_counts=True)
class_weights = [sum(counts)/c for c in counts]
test_example_weights = np.array([class_weights[l] for l in test['class']])
test_example_weights = np.roll(test_example_weights, -STACK_SIZE)
test_sampler = WeightedRandomSampler(test_example_weights, len(test))
```

The process of calculating the weight of each sample in a dataset is an important step in various machine-learning tasks such as image classification, text classification, and object detection, among others. The weight of each sample is used to balance the dataset so that each class contributes equally to the learning process. When a dataset is imbalanced, that is, when some classes have more samples than others, the learning algorithm may focus more on the majority class and ignore the minority class, leading to poor performance.

To calculate the weight of each sample, we first split the dataset csv into train and test using the train_test_split method from the scikit-learn (sklearn) library. After that, we get the unique labels and the count of each label. This helps us to understand the distribution of classes in the dataset. We then calculate the weightage of each class by dividing the sum of all counts (size of the dataset) by the number of times that class occurs. This gives us a measure of how important each class is in the dataset.

The next step is to get the weight of each example by assigning the class weight to the example. This is done by iterating through the dataset and assigning the weight to each sample based on its class label. We roll the example weights by the stack size because the label associated with a certain image is actually the label of the index of that image + STACK_SIZE. This ensures that each sample is given the correct weight based on its class label.

In summary, calculating the weight of each sample in a dataset is a crucial step in machine learning tasks. It helps to balance the dataset and ensures that each class contributes equally to the learning process. The process involves splitting the dataset into train and test sets, obtaining the unique labels and the count of each label, calculating the weightage of each class, and assigning the weight to each sample based on its class label.

Finally, we can create the dataloader for both the training and test datasets. 

```python
dataset = SnakeDataSet(root_dir="captures", dataframe = train, stack_size=STACK_SIZE, transform=transformer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last= True)
test_dataset = SnakeDataSet(root_dir="captures", dataframe = test, stack_size=STACK_SIZE,  transform=transformer)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler = test_sampler, drop_last=True)
```

## Create Transforms

For the model that we are going to use, we need to perform certain transforms on the data. First, we need to compute the mean and standard deviation of the dataset images. We can do this using this code. 

```python
def compute_mean_std(dataloader):
    '''
    We assume that the images of the dataloader have the same height and width
    source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py
    '''
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for batch_images, labels in tqdm(dataloader):  # (B,H,W,C)
        batch_images = batch_images.permute(0,3,4,2,1)
        channels_sum += torch.mean(batch_images, dim=[0, 1, 2, 3])
        channels_sqrd_sum += torch.mean(batch_images ** 2, dim=[0, 1, 2,3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

compute_mean_std(dataloader)
```

This will return the mean and standard deviation, which we can plug into the transformation below. 

```python
transformer = Compose([
    Resize((84,84), antialias=True),
    CenterCrop(84),
    ToTensor(),
    Normalize(mean =[ -0.7138, -2.9883,  1.5832], std =[0.2253, 0.2192, 0.2149]) 
])
```

This resizes the image to 84x84, crops it, converts it to tensor and normalizes it. 

## Creating the Model

As discussed in the introduction, we are going to use the PyTorch-provided `r3d` model. This is a 3D Convolution model that uses ResNet architecture.  

```python
model = r3d_18(weights = R3D_18_Weights.DEFAULT)
model.fc = nn.Linear(in_features=512, out_features=5, bias=True)
summary(model, (32,3,4,84,84))
```

We load the default weights and replace the last fully connected layer to predict 5 output classes as required in our application. This is what the network looks like. We have about 33 million trainable parameters.

![Network Architecture](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/u9f4owee1ajxg3bbioxp.png)

## Training the Model

Now, we can finally train the model. We first create the optimizer and the loss criterion. Here, we use a learning rate of 10e-5 and a weight decay of 0.1. 

```python
num_epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(model.parameters(), 10e-5, weight_decay=0.1)
model.to(device)
criterion = nn.CrossEntropyLoss()
```

Now, we create the training loop.

```python
for epoch in range(num_epochs):
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    val_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0

    # Set model to training mode
    model.train()

    # tqdm bar for progress visualization
    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss.item()
        _, predicted = torch.max(torch.softmax(outputs,1), 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Update tqdm bar with current loss and accuracy
        pbar.set_postfix({'Loss': total_loss / total_samples, 'Accuracy': correct_predictions / total_samples})
        steps = steps + 1

    model.eval()
    with torch.inference_mode():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels)

            # Update statistics
            val_loss += loss.item()
            _, predicted = torch.max(torch.softmax(outputs,1), 1)
            val_correct_predictions += (predicted == labels).sum().item()
            val_total_samples += labels.size(0)

    # Calculate and print epoch-level accuracy and loss for validation
    epoch_loss = val_loss / val_total_samples
    epoch_accuracy = val_correct_predictions / val_total_samples
    print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {epoch_loss:.4f}, Val Accuracy: {epoch_accuracy:.4f}')
    torch.save(model.state_dict(), "model_r3d.pth")
```

This will train the network. 

## Play the game

Create a new python script `play.py`. In this script, we will load the model and write code similar to `[capture.py](http://capture.py)` to load the game and collect screenshots of the game. We will stack 4 screenshots and pass it through our network. 

```python
import base64
import torch
import cv2
import keyboard
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,  Grayscale
from torch import nn
from collections import deque
from torchvision.models.video import r3d_18
from selenium import webdriver
from selenium.webdriver.common.by import By

label_keys= {
    0: "",
    1 :"left",
    2: "up",
    3: "right",
    4: "down"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = r3d_18(weights = None)
model.fc = nn.Linear(in_features=512, out_features=5, bias=True)

model.load_state_dict(torch.load("model_mc3.pth"))
model.to(device)
model.eval()

transformer = Compose([
    Resize((84,84), antialias=True),
    CenterCrop(84),
    ToTensor(),
    Normalize(mean =[ -0.7138, -2.9883,  1.5832], std =[0.2253, 0.2192, 0.2149] )
])

# Create a webdriver instance using Firefox
driver = webdriver.Firefox()
# Navigate to the Google Snake game website
driver.get("https://www.google.com/fbx?fbx=snake_arcade")

frame_stack = deque(maxlen=4)

while True:
     # Find the canvas element on the webpage
    canvas = driver.find_element(By.CSS_SELECTOR, "canvas")

    # Get the base64 encoded image data from the canvas
    canvas_base64 = driver.execute_script(
        "return arguments[0].toDataURL('image/png').substring(21);", canvas
    )
    # Decode the base64 data to get the PNG image
    canvas_png = base64.b64decode(canvas_base64)

    # Convert the PNG image to a grayscale numpy array
    image = cv2.cvtColor(
        np.array(Image.open(io.BytesIO(canvas_png))), cv2.COLOR_BGR2RGB
    ) 

    frame_stack.append(transformer(image))
    input = torch.stack([*frame_stack],dim = 1).to(device).squeeze().unsqueeze(0)

    if len(frame_stack) == 4:
        with torch.inference_mode():
            outputs = model(input).to(device)
            preds = torch.softmax(outputs, dim=1).argmax(dim = 1)

            if preds.item() != 0:
                keyboard.press_and_release(label_keys[preds.item()])
```

Once we run this script, you can see the snake playing.


![Demo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/kvt78c8gzk4ojrw7f3gh.gif)

## Conclusion

In wrapping up our journey into building an AI game bot for Google Snake, we've explored the exciting realm of imitation learning and harnessed the power of a 3D Convolution ResNet model. Beyond the fun of game development, the acquired skills have broader applications in serious AI scenarios.

We began by understanding the significance of imitation learning, focusing on training our AI agent through human gameplay. The use of a 3D Convolution ResNet model allowed us to accurately capture motion by stacking four frames of the game.

Practical details covered data collection using Selenium, creating labeled datasets, and transforming them into a PyTorch-friendly format. We highlighted the importance of addressing class imbalance through WeightedRandomSampler.

Implementation involved constructing a custom dataset class, selecting an appropriate model architecture, and training the model using essential transforms. The training loop and evaluation on a test set provided insights into the model's performance.

To complete the journey, we showcased how the trained model can be used to play the game in real-time. The provided Python script demonstrated capturing frames, making predictions, and controlling the snake's movements.

In essence, this concise guide empowers you to master AI in gaming through imitation learning and 3D Convolution ResNet models, with skills extending to broader AI applications.

Git Repo: https://github.com/akshayballal95/autodrive-snake/tree/blog

<div style="display: flex; gap:10px; align-items: center">

<div style = "display: flex; flex-direction:column; gap:10px; justify-content:space-between">
<p style="padding:0; margin:0">my website: <a href ="http://www.akshaymakes.com/">http://www.akshaymakes.com/</a></p>
<p  style="padding:0; margin:0">linkedin: <a href ="https://www.linkedin.com/in/akshay-ballal/">https://www.linkedin.com/in/akshay-ballal/</a></p>
<p  style="padding:0; margin:0">twitter: <a href ="https://twitter.com/akshayballal95">https://twitter.com/akshayballal95/</a></p>
</div>
</div>