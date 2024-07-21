
# - **`train/images/`**: Contains training images.
# - **`train/masks/`**: Contains corresponding ground truth masks for training images.
# - **`test/`**: Contains test images for which predictions will be made.

# ### Preprocessing

# Images and masks are resized to 256x256 pixels and normalized. The following transformations are applied:

# - **Resize**: (256, 256)
# - **ToTensor**: Convert images and masks to PyTorch tensors
# - **Normalization**: Standardize the image pixel values

# ## Training the Model

# The model is trained using a binary cross-entropy loss with logits, suitable for binary segmentation tasks.

### Training Script

#```python
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from glob import glob

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define dataset class
class RoadDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# Prepare training data
train_image_dir = 'dataset/train/images'
train_mask_dir = 'dataset/train/masks'
train_image_paths = sorted(glob(os.path.join(train_image_dir, '*.jpg')))
train_mask_paths = sorted(glob(os.path.join(train_mask_dir, '*.png')))

train_dataset = RoadDataset(train_image_paths, train_mask_paths, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Load the model
model = deeplabv3_resnet50(pretrained=False)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images, masks = images.to('cuda'), masks.to('cuda')
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}')

# Save the trained model
torch.save(model.state_dict(), 'road_detection_model.pth')
