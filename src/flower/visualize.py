import torch
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
from datasets import load_dataset
from flower.task import Net

# Fashion-MNIST class names
FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# Load the saved model 
model.load_state_dict(torch.load("final_model.pt"))

pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

def apply_transforms(batch):
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

# Load test data
test_dataset = load_dataset("zalando-datasets/fashion_mnist", split="test")
test_dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
test_loader = DataLoader(test_dataset, batch_size=16)

# Get predictions
model.eval()
with torch.no_grad():
    batch = next(iter(test_loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

# Visualize
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.ravel()

for i in range(16):
    # Denormalize image for display
    image = images[i].cpu().squeeze()
    image = image * 0.5 + 0.5  # Reverse normalization
    
    axes[i].imshow(image, cmap='gray')
    pred_label = FASHION_MNIST_CLASSES[predictions[i]]
    true_label = FASHION_MNIST_CLASSES[labels[i]]
    color = 'green' if predictions[i] == labels[i] else 'red'
    axes[i].set_title(f'Pred: {pred_label}\nTrue: {true_label}', color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('predictions.png')
plt.show()
print("Visualization saved as predictions.png")