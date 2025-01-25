import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

def train_model(data_dir, save_dir, arch='vgg16', hidden_units=512, learning_rate=0.001, epochs=5, gpu=False, batch_size=32):
    print("Starting training process...")  # Debug information
    
    # Define transforms for the training and validation sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    valid_data = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=valid_transforms)
    print("Datasets loaded.")  # Debug information

    # Check dataset sizes
    print(f"Training dataset size: {len(train_data)}")
    print(f"Validation dataset size: {len(valid_data)}")

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    print("Data loaders initialized.")  # Debug information

    # Load pre-trained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("Architecture not recognized. Defaulting to VGG16.")
        model = models.vgg16(pretrained=True)
    print("Model loaded.")  # Debug information

    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    print("Classifier defined.")  # Debug information

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print("Loss and optimizer defined.")  # Debug information

    # Move to GPU if available and requested
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")  # Debug information

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        print(f"Starting epoch {epoch+1}/{epochs}...")  # Debug information
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1} completed. Train loss: {running_loss/len(train_loader):.3f}")  # Debug information

        # Validation loop
        model.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                valid_loss += batch_loss.item()
                
                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Validation loss: {valid_loss/len(valid_loader):.3f}, Validation accuracy: {accuracy/len(valid_loader):.3f}")  # Debug information

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Save directory created: {save_dir}")

    # Save checkpoint
    print("Saving checkpoint...")  # Debug information
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
    print(f"Checkpoint saved to {os.path.join(save_dir, 'checkpoint.pth')}")

# Direct function call for testing in the notebook environment
data_dir = '/workspace/cd0673/43ef1733-cf74-40a9-8c22-b6e440128200/image-classifier-part-1-workspace/home/aipnd-project/flowers'  # Add the path to your flowers directory here
save_dir = '/workspace/cd0673/43ef1733-cf74-40a9-8c22-b6e440128200/image-classifier-part-1-workspace/home/aipnd-project/checkpoints'  # Add the path to your save directory here
arch = 'vgg16'
hidden_units = 512
learning_rate = 0.001
epochs = 5
gpu = True  # Use GPU if available

train_model(data_dir, save_dir, arch, hidden_units, learning_rate, epochs, gpu, batch_size=32)

# Verify the contents of the checkpoints directory
print("Contents of the checkpoints directory:", os.listdir(save_dir))
