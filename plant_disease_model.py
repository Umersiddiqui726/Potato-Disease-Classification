import torch
import torch.nn as nn
import timm
from config import MODEL_NAME, NUM_CLASSES

class PlantDiseaseModel(nn.Module):
    def __init__(self):
        super(PlantDiseaseModel, self).__init__()
        
        # Load pre-trained Vision Transformer
        self.vit = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=NUM_CLASSES
        )
        
    def forward(self, x):
        return self.vit(x)
    
    def get_optimizer(self, learning_rate, weight_decay):
        return torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def get_scheduler(self, optimizer, num_epochs, warmup_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 0.5 * (1 + torch.cos(torch.tensor(epoch - warmup_epochs) / (num_epochs - warmup_epochs) * torch.pi))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) 