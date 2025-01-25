import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import argparse
import os

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('image_name', type=str, help='File name of the image (required)')
parser.add_argument('checkpoint_name', type=str, help='File name of the trained model (required)')
parser.add_argument('--category_names', type=str, help='File name of the JSON mapping categories to names')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes. Default is 5')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
args_in = parser.parse_args()

# Check if GPU is available and requested
if args_in.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=device)
    
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 102),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
       returns a Numpy array
    '''
    img = Image.open(image_path) 
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    img = np.array(transform(img))

    return img

def predict(image_path, model, topk=5):
    '''Predict the class (or classes) of an image using a trained deep learning model.'''
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.to(device)
    
    model.eval()
    
    with torch.no_grad():
        output = model.forward(image)
        
    output_prob = torch.exp(output)
    
    probs, indices = output_prob.topk(topk)
    probs   =   probs.to('cpu').numpy().tolist()[0]
    indices = indices.to('cpu').numpy().tolist()[0]
    
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indices]
    
    return probs, classes

# File paths
current_dir = os.getcwd()
image_path = os.path.join(current_dir, args_in.image_name)
checkpoint_path = os.path.join("/workspace", "checkpoints", args_in.checkpoint_name)
category_names_path = os.path.join(current_dir, args_in.category_names) if args_in.category_names else None

# Check if the file exists
if not os.path.isfile(checkpoint_path):
    print(f"Checkpoint file not found: {checkpoint_path}")
else:
    # Load the checkpoint
    model = load_checkpoint(checkpoint_path)
    model.to(device)

    # Predict the class
    probs, classes = predict(image_path, model, topk=args_in.top_k)

    # Map category names if provided
    if category_names_path and os.path.isfile(category_names_path):
        with open(category_names_path, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(cls)] for cls in classes]
        print("Class names:", class_names)

    # Print the results
    print("Class numbers:", classes)
    print("Probabilities (%):", [round(prob * 100, 2) for prob in probs])
