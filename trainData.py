import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
from torch import optim
from torchsummary import summary
import torch.nn.functional as F


#Define paths and filenames
datasetPath = './testData'
datasetOutputPath = './trainingData'
csvFilePath = 'output.csv'


# with open(f'./{datasetOutputPath}/groundTruth.json') as jsonData:
#     groundTruth = json.load(jsonData)
# total_images = len(groundTruth)
# print(total_images)
#for groundTruths in groundTruth:
    # print('\nTest Photo Name:', groundTruths['fileName'])  #Print name of the photo thats being examined
    # print('Number of faces found:',groundTruths['has_face'])  #Print number of faces of that photo pre-determined inside groundTruth.json
    # faceCount = groundTruths['has_face']
    # print(faceCount)
    

#Load groundTruth
def loadGroundTruth(jsonPath):
    with open(jsonPath) as jsonData:
        return json.load(jsonData)

#Dataset
torch.manual_seed(0)

class pytorchDataset(Dataset):
    
    def __init__(self, data_dir, transform=None, data_type="train"):
        self.full_filenames = loadGroundTruth(os.path.join(data_dir, 'groundTruth.json'))
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_dir, self.full_filenames[idx]['fileName'])
        image = Image.open(file_name).convert("RGB")
        label = torch.tensor(self.full_filenames[idx]['has_face'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

#Define the CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)
    
# #Binary Classifier Defining
# def findConv2dOutShape(hin,win,conv,pool=2):
#     # get conv arguments
#     kernel_size=conv.kernel_size
#     stride=conv.stride
#     padding=conv.padding
#     dilation=conv.dilation

#     hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
#     wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

#     if pool:
#         hout/=pool
#         wout/=pool
#     return int(hout),int(wout)


#Load and preprocess data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = pytorchDataset(data_dir=datasetOutputPath, transform=transform)

#Split dataset into train and validation sets
len_dataset = len(dataset)
len_train = int(0.8 * len_dataset)
train_dataset, val_dataset = random_split(dataset, [len_train, len_dataset - len_train])

#Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#Initialize the CNN model
model = SimpleCNN()

#Loss func.
loss_function = nn.NLLLoss(reduction="sum")

#Optimizer
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

#Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs.squeeze(), labels)
        loss.backward() 
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

#Save the trained model
torch.save(model.state_dict(), 'CNN_faceDetection.pth')
