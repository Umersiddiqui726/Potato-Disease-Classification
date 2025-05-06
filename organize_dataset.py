import os
import sys
import shutil
from pathlib import Path
import random

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import DATASET_PATH, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

def organize_dataset(source_dir):
    """
    Organize the dataset into train, validation, and test sets.
    Assumes the source directory contains subdirectories for each class.
    """
    # Create train, val, and test directories
    train_dir = os.path.join(DATASET_PATH, 'train')
    val_dir = os.path.join(DATASET_PATH, 'val')
    test_dir = os.path.join(DATASET_PATH, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(source_dir) 
                 if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"Found {len(class_dirs)} classes")
    
    total_processed = 0
    for class_name in class_dirs:
        print(f"\nProcessing class: {class_name}")
        
        # Create class directories in train, val, and test
        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # Get all images in the class directory
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split sizes
        total_images = len(images)
        train_size = int(total_images * TRAIN_RATIO)
        val_size = int(total_images * VAL_RATIO)
        
        # Split images
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]
        
        # Move images to respective directories
        for img, dst_dir in [
            (img, os.path.join(train_dir, class_name)) for img in train_images
        ] + [
            (img, os.path.join(val_dir, class_name)) for img in val_images
        ] + [
            (img, os.path.join(test_dir, class_name)) for img in test_images
        ]:
            try:
                src = os.path.join(class_path, img)
                dst = os.path.join(dst_dir, img)
                shutil.move(src, dst)
                total_processed += 1
                if total_processed % 100 == 0:
                    print(f"Processed {total_processed} images...")
            except Exception as e:
                print(f"Error processing {img}: {str(e)}")
        
        print(f"Class {class_name}:")
        print(f"  Total images: {total_images}")
        print(f"  Train: {len(train_images)}")
        print(f"  Validation: {len(val_images)}")
        print(f"  Test: {len(test_images)}")
    
    print(f"\nTotal images processed: {total_processed}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize plant disease dataset')
    parser.add_argument('--source_dir', type=str, required=True,
                      help='Path to the directory containing the downloaded dataset')
    args = parser.parse_args()
    
    organize_dataset(args.source_dir) 