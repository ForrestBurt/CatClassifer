import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import cv2

class CatClassifier(nn.Module):
    def __init__(self):
        super(CatClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc = nn.Linear(256 * 16 * 16, 512)

        # Output layer
        self.output = nn.Linear(512, 4)

    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the output of the max pooling layer
        x = x.view(-1, 256 * 16 * 16)

        # Pass the output through the fully connected layer
        x = F.relu(self.fc(x))

        # Pass the output through the output layer
        x = self.output(x)

        return x

cat_a_videos = []
cat_b_videos = []
no_cat_videos = []
both_cats_videos = []

for i in range(1, 50):
    video_a = cv2.VideoCapture(f"/data/apollo/apollo-{i}.mp4")
    video_b = cv2.VideoCapture(f"/data/remus/remus-{i}.mp4")
    video_nc = cv2.VideoCapture(f"/data/no/none-{i}.mp4")
    video_bc = cv2.VideoCapture(f"/data/both/both-{i}.mp4")
   
    print(f"Processing file: video-{i}.mp4")

    while True:
        ret_a, frame_a = video_a.read()
        ret_b, frame_b = video_b.read()
        ret_nc, frame_nc = video_nc.read()
        ret_bc, frame_bc = video_bc.read()

        
        if not ret_a or not ret_b or not ret_nc or not ret_bc:
            break

        frame_a = cv2.resize(frame_a, (256, 256))
        frame_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
        frame_a = frame_a.transpose((2, 0, 1))
        cat_a_videos.append(frame_a)
        
        frame_b = cv2.resize(frame_b, (256, 256))
        frame_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
        frame_b = frame_b.transpose((2, 0, 1))
        cat_b_videos.append(frame_b)
        
        frame_nc = cv2.resize(frame_nc, (256, 256))
        frame_nc = cv2.cvtColor(frame_nc, cv2.COLOR_BGR2RGB)
        frame_nc = frame_nc.transpose((2, 0, 1))
        no_cat_videos.append(frame_nc)
        
        frame_bc = cv2.resize(frame_bc, (256, 256))
        frame_bc = cv2.cvtColor(frame_bc, cv2.COLOR_BGR2RGB)
        frame_bc = frame_bc.transpose((2, 0, 1))
        both_cats_videos.append(frame_bc)
        
cat_a_videos = np.array(cat_a_videos)
cat_b_videos = np.array(cat_b_videos)
no_cat_videos = np.array(no_cat_videos)
both_cats_videos = np.array(both_cats_videos)

# Create the datasets
cat_a_dataset = torch.tensor(cat_a_videos, dtype=torch.float32)
cat_b_dataset = torch.tensor(cat_b_videos, dtype=torch.float32)
no_cat_dataset = torch.tensor(no_cat_videos, dtype=torch.float32)
both_cats_dataset = torch.tensor(both_cats_videos, dtype=torch.float32)

# Concatenate the datasets and create the targets
dataset = torch.cat([cat_a_dataset, cat_b_dataset, no_cat_dataset, both_cats_dataset], dim=0)
targets = torch.cat([torch.zeros(cat_a_dataset.shape[0]), torch.ones(cat_b_dataset.shape[0]),
                     2 * torch.ones(no_cat_dataset.shape[0]), 3 * torch.ones(both_cats_dataset.shape[0])], dim=0)

# Create a DataLoader for the dataset and targets
dataloader = torch.utils.data.DataLoader(list(zip(dataset, targets)), batch_size=32, shuffle=True)

print("ready for training")

# Train the model on two GPUs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(CatClassifier(), device_ids=[0, 1]).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.module.parameters(), lr=0.001, momentum=0.9)

num_epochs = 5

print("training initalizing")

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()


        outputs = model(inputs)
        labels = labels.long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()

        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

        print(i)
print('Finished Training')

torch.save(model.state_dict(), 'cat_classifier.pt')
