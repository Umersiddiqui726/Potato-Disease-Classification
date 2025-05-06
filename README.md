# Plant Disease Classification using CNN and Vision Transformer

![Plant Disease Classification](https://img.shields.io/badge/DeepLearning-PyTorch-blue) ![Vision Transformer](https://img.shields.io/badge/Model-ViT%20%7C%20CNN-green)

## 🌱 Overview

This project implements a deep learning pipeline for classifying plant diseases from leaf images using both Convolutional Neural Networks (CNN) and Vision Transformers (ViT). The goal is to provide an automated, accurate, and scalable solution for early plant disease detection, which can help farmers and agricultural experts improve crop yield and reduce losses.

---

## 🚀 Features

- **Supports both CNN and ViT architectures**
- **Handles large datasets (>1GB)**
- **Data augmentation and normalization**
- **Train/validation/test split with reproducibility**
- **Easy model selection via configuration**
- **Prediction script for new images**
- **Well-documented and modular code**

---

## 📂 Dataset

- **Source:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Structure:** Images organized by disease class, split into `train`, `val`, and `test` folders.
- **Classes:** 15 (for the color subset)

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/plant-disease-classification.git
   cd plant-disease-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and organize the dataset:**
   - Download the PlantVillage dataset from Kaggle.
   - Use the provided script to organize the dataset:
     ```bash
     python utils/organize_dataset.py --source_dir path_to_downloaded_dataset
     ```

---

## ⚙️ Configuration

Edit `config.py` to set:
- `MODEL_TYPE = "cnn"` or `"vit"`
- `NUM_CLASSES` (set to 15 for PlantVillage color subset)
- Other hyperparameters as needed

---

## 🏋️‍♂️ Training

Train the model (CNN or ViT) by running:
```bash
python train.py
```
- The best model will be saved in the `checkpoints/` directory.

---

## 🔍 Prediction

Predict the disease class for a new image:
```bash
python predict.py --image_path path_to_image.jpg --model_type cnn
# or
python predict.py --image_path path_to_image.jpg --model_type vit
```

---

## 📊 Results

- **CNN Test Accuracy:** 98%
- **ViT Test Accuracy:** _(Fill in your result)_

Training and validation curves are saved and can be visualized using TensorBoard or matplotlib.

---

## 📑 Project Structure

```
.
├── data/                   # Dataset directory
├── models/                 # Model definitions (CNN, ViT)
├── utils/                  # Data utilities and organization script
├── checkpoints/            # Saved model weights
├── train.py                # Training script
├── predict.py              # Prediction script
├── config.py               # Configuration file
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📚 References

- [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [timm Library](https://github.com/huggingface/pytorch-image-models)
- [Plant Disease Detection using Deep Learning](https://arxiv.org/abs/1604.03169)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open an issue or submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

**Let's help farmers and researchers fight plant diseases with AI!** 
