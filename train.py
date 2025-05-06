import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime

from models.plant_disease_model import PlantDiseaseModel
from models.cnn_model import PlantDiseaseCNN
from utils.data_utils import get_dataloaders
from config import *

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss/len(val_loader), 100.*correct/total

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss/len(test_loader), 100.*correct/total

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model based on configuration
    if MODEL_TYPE == "cnn":
        model = PlantDiseaseCNN().to(device)
        print("Using CNN model")
    else:
        model = PlantDiseaseModel().to(device)
        print("Using Vision Transformer model")
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Number of classes: {len(train_loader.dataset.classes)}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = model.get_optimizer(LEARNING_RATE, WEIGHT_DECAY)
    scheduler = model.get_scheduler(optimizer, NUM_EPOCHS, WARMUP_EPOCHS)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(LOG_DIR, f"{MODEL_TYPE}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'best_{MODEL_TYPE}_model.pth'))
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Final test evaluation
    print("\nRunning final test evaluation...")
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    writer.close()
    print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    main() 