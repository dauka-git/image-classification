

# CIFAR-10 Image Classification

A deep learning project that classifies images from the CIFAR-10 dataset using a convolutional neural network (CNN), with a Streamlit web interface for real-time predictions.

## Project Structure

- **`cifar10_training.ipynb`** - Jupyter notebook for model training and evaluation
- **`streamlit_app.py`** - Web application for image classification
- **`cifar_net.pth`** - Pre-trained model weights (generated after training)

## Model Architecture

The project uses `SimpleCIFAR10Net`, a CNN with:
- 2 convolutional layers (32 and 64 filters)
- Max pooling layers
- 2 fully connected layers (256 and 10 units)
- Output: 10 classes (plane, car, bird, cat, deer, dog, frog, horse, ship, truck)

## Training Details

- **Dataset**: CIFAR-10 (50k training, 10k test images)
- **Augmentation**: Random horizontal flips and rotations
- **Optimizer**: Adam with learning rate 0.001
- **Training**: 10 epochs with step learning rate scheduling
- **Accuracy**: ~70% on test set

## Usage

### Training
```bash
jupyter notebook cifar10_training.ipynb
```

### Web Application
```bash
streamlit run streamlit_app.py
```

Upload any image through the web interface to get real-time classification predictions with confidence scores across all 10 CIFAR-10 classes.

## Requirements

- torch
- torchvision  
- streamlit
- matplotlib
- scikit-learn
- seaborn
- PIL

The model automatically uses GPU if available, otherwise falls back to CPU.
