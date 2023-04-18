import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import time
# Define transforms for the training data
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transforms for the validation data
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create train and validation datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

# Define data loader for train and validation datasets
train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# Define the ResNet50 model
model = torchvision.models.resnet50(pretrained=False, num_classes=10)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
start_time = time.time()
# Train the model
for epoch in range(1):
    # Train the model for one epoch
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validate the model
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    # Print epoch statistics
    print(f"Epoch {epoch+1}, Validation Accuracy: {100 * total_correct / total_samples:.2f}%")
end_time = time.time()
print(f"training time {end_time - start_time}")