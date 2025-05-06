import os

# Dataset Configuration
DATASET_PATH = "data/plant_disease"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Model Configuration
MODEL_TYPE = "cnn"  # Using CNN model
MODEL_NAME = "vit_base_patch16_224"  # Only used for ViT
NUM_CLASSES = 15  # Updated to match the number of classes found
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training Configuration
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
WARMUP_EPOCHS = 5

# Paths
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True) 