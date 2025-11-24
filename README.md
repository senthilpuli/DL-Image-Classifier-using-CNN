# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.
## DESIGN STEPS
### STEP 1: 

Import necessary libraries for PyTorch, torchvision, visualization, and evaluation metrics.

### STEP 2: 

Define image transformations: convert tensors and normalize dataset between -1 and 1.


### STEP 3: 

Load MNIST training and testing datasets with transformations and enable downloading automatically.


### STEP 4: 

Create DataLoader objects for training and testing datasets with batch processing.


### STEP 5: 

Define CNNClassifier class with convolution, pooling, and fully connected layers.


### STEP 6: 

Implement forward function using ReLU activations, pooling, and flattening before classification.

### STEP 7: 

Initialize CNN model, move to GPU if available, and display summary.

### STEP 8:

Define cross-entropy loss function and Adam optimizer with learning rate 0.001.


### STEP 9:

Train model for epochs: forward pass, compute loss, backpropagate, and update parameters.


### STEP 10: 

Test model: predict outputs, calculate accuracy, store predictions, and generate confusion matrix.

### STEP 11: 

Visualize confusion matrix using heatmap and display classification report with precision metrics.


### STEP 12: 

Predict single image: load from dataset, infer, and display actual-predicted labels.



## PROGRAM

### Name: Senthil velan

### Register Number: 212222220041
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Get the shape of the first image in the training dataset
image, label = train_dataset[0]
print("Image shape:", image.shape)
print("Number of training samples:", len(train_dataset))

# Get the shape of the first image in the test dataset
image, label = test_dataset[0]
print("Image shape:", image.shape)
print("Number of testing samples:", len(test_dataset))


# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
      x=self.pool(torch.relu(self.conv1(x)))
      x=self.pool(torch.relu(self.conv2(x)))
      x=self.pool(torch.relu(self.conv3(x)))
      x=x.view(x.size(0),-1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x

from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name: senthil velan .s')
print('Register Number:  212222220041')
summary(model, input_size=(1, 28, 28))

# Initialize model, loss function, and optimizer
criterion = nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=10):
  for epoch in range(num_epochs):
     model.train()
     running_loss=0.0
     for images,labels in train_loader:
      if torch.cuda.is_available():
        images,labels=images.to(device),labels.to(device)
      optimizer.zero_grad()
      outputs=model(images)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
  print('Name: vignesh.v')
  print('Register Number: 212223230241 ')
# Train the model
train_model(model, train_loader, num_epochs=10)
## Step 4: Test the Model

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: vignesh.v ')
    print('Register Number: 212223230241 ')
    print(f'Test Accuracy: {accuracy:.4f}')
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))
# Evaluate the model
test_model(model, test_loader)


## Step 5: Predict on a Single Image
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    if torch.cuda.is_available():
        image = image.to(device)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)

    class_names = [str(i) for i in range(10)]

    print('Name: Senthil velan.s')
    print('Register Number: 212222220041')
    plt.imshow(image.cpu().squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')
# Example Prediction
predict_image(model, image_index=80, dataset=test_dataset)

```

### OUTPUT
<img width="615" height="539" alt="Screenshot 2025-10-07 224913" src="https://github.com/user-attachments/assets/ca6b0628-208d-4665-b54c-91ecb586f173" />

<img width="260" height="233" alt="Screenshot 2025-10-07 224921" src="https://github.com/user-attachments/assets/be5df355-95ef-439a-b29a-4a6fad146277" />

<img width="806" height="726" alt="Screenshot 2025-10-07 224935" src="https://github.com/user-attachments/assets/5eaa91d9-4293-4b7d-bbaa-c600ede29611" />

<img width="528" height="354" alt="Screenshot 2025-10-07 224943" src="https://github.com/user-attachments/assets/99ff9e3e-d9c7-4bb5-bcdc-0bab4d4c628b" />

<img width="499" height="553" alt="Screenshot 2025-10-07 225000" src="https://github.com/user-attachments/assets/a2b0c2d5-013e-4cee-a99e-91179ae52262" />

## RESULT
Developing a convolutional neural network (CNN) classification model for the given dataset was executed successfully.
