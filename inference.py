# Define test dataset class
class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_path = image_path.replace('\\', '/')  # Normalize the path
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

# Prepare test data
test_image_dir = 'dataset/test'
test_image_paths = sorted(glob(os.path.join(test_image_dir, '*.jpg')))

# Normalize paths
test_image_paths = [path.replace('\\', '/') for path in test_image_paths]

# Create test dataset and dataloader
test_dataset = TestDataset(test_image_paths, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the model and weights
model = deeplabv3_resnet50(pretrained=False)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
model.load_state_dict(torch.load('road_detection_model.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Run predictions
def predict(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, image_paths in dataloader:
            images = images.to(device)
            outputs = model(images)['out']
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = preds > 0.5  # Threshold the predictions
            predictions.extend(zip(image_paths, preds))
    return predictions

predictions = predict(model, test_loader, 'cuda' if torch.cuda.is_available() else 'cpu')

# Save the predictions
output_dir = 'dataset/predictions'
os.makedirs(output_dir, exist_ok=True)

for image_path, pred in predictions:
    pred_image = pred.squeeze(0)  # Remove the batch dimension
    pred_image = (pred_image * 255).astype(np.uint8)  # Convert to uint8
    pred_image = Image.fromarray(pred_image)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    pred_image.save(output_path)

    # Optional: Display the prediction
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(Image.open(image_path))
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_image, cmap='gray')
    plt.show()
