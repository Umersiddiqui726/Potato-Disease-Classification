import torch
import torch.nn as nn
from PIL import Image
import argparse
from torchvision import transforms
import os

from models.plant_disease_model import PlantDiseaseModel
from models.cnn_model import PlantDiseaseCNN
from config import *

def load_model(device):
    if MODEL_TYPE == "cnn":
        model = PlantDiseaseCNN().to(device)
        model_path = os.path.join(CHECKPOINT_DIR, 'best_cnn_model.pth')
    else:
        model = PlantDiseaseModel().to(device)
        model_path = os.path.join(CHECKPOINT_DIR, 'best_vit_model.pth')
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(image_path, model, device):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

def main():
    parser = argparse.ArgumentParser(description='Predict plant disease from image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'vit'], default=MODEL_TYPE,
                      help='Model type to use for prediction (cnn or vit)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(device)
    
    # Get class names
    class_names = sorted(os.listdir(DATASET_PATH))
    
    # Make prediction
    predicted_class, confidence = predict_image(args.image_path, model, device)
    
    print(f'\nPrediction Results:')
    print(f'Model Type: {MODEL_TYPE.upper()}')
    print(f'Predicted Disease: {class_names[predicted_class]}')
    print(f'Confidence: {confidence:.2%}')

if __name__ == '__main__':
    main() 