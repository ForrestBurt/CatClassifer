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

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(CatClassifier(), device_ids=[0, 1]).to(device)
model.load_state_dict(torch.load("/home/fburt/cat_classifier.pt"))
model.eval()

# Open the input .mp4 file using OpenCV
for i in range(49):
    print(f"processing video {i}")
    input_video = cv2.VideoCapture(f'/data/apollo/apollo-{i}.mp4')

# Get the video dimensions
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_video.get(cv2.CAP_PROP_FPS))

# Define the codec and create the output .mp4 file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(f'/data/test/output-newtest-apollo-{i}.mp4', fourcc, fps, (width, height))

# Loop over the frames in the video
    while True:
    # Read a frame from the video
        ret, frame = input_video.read()

    # Break the loop if the video has ended
        if not ret:
            break

        frame_orig = frame
    # Preprocess the frame
        frame = cv2.resize(frame, (256,256))
        frame = np.transpose(frame, (2,0,1))
        frame = np.expand_dims(frame, axis=0)
        frame = torch.tensor(frame, dtype=torch.float32)

    # Run the frame through the model
        output = model(frame)
        output = torch.softmax(output, dim=1)

    # Get the predicted class and the class probability
        _, predicted = torch.max(output, dim=1)
        predicted = int(predicted)
        probability = float(output[0, predicted])

    # Map the predicted class to the appropriate label
        if predicted == 0:
            label = 'Apollo (Cat A)'
        elif predicted == 1:
            label = 'Both Cats'
        elif predicted == 2:
            label = 'No Cats'
        else:
            label = 'Remus (Cat B)'

    #frame_orig = frame_orig.numpy().astype(np.uint8)
    #frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    # Display the frame, the label, and the probability
        cv2.putText(frame_orig, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame_orig, f'{probability:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        output_video.write(frame_orig)

# Release the input and output video files
    input_video.release()
    output_video.release()

