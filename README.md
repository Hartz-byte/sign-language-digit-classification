# Sign Language Digit Classification using CNN
This project involves building a Convolutional Neural Network (CNN) to classify hand gestures representing digits (0–9) in American Sign Language. The model is trained on a pre-processed image dataset and evaluated for accuracy, precision, recall, and overall performance using standard metrics and visualizations.

---

## Overview
This project aims to recognize sign language digits using grayscale images of hand gestures. A custom CNN model is trained from scratch to classify the images into one of ten classes (0–9). The project demonstrates core deep learning skills, including:
- Image preprocessing
- CNN design and tuning
- Regularization strategies
- Model evaluation with detailed metrics
- Visualization of results and errors

---

## Dataset
The dataset consists of grayscale images of hand gestures with dimensions 64x64. Each image corresponds to a digit from 0 to 9.
- Input shape: (64, 64, 1)
- Number of classes: 10
- Total samples: ~2000
- Labels: One-hot encoded, later converted to class integers for training

---

## Preprocessing
- Normalization: Pixel values are scaled to the range [0, 1]
- Reshaping: Channel dimension added for CNN compatibility
- Train-Test Split: 80% training, 20% testing (stratified)
- One-hot Encoding: Applied to training and testing labels

---

## Training Strategy
- Loss Function: Categorical Crossentropy
- Optimizer: Adam (learning rate = 0.0001)
- Batch Size: 32
- Epochs: Up to 50 (early stopping used)
- Validation Split: 10% of training data
- Callbacks:
    - ModelCheckpoint: Save the best model based on validation accuracy
    - EarlyStopping: Halt training if no improvement in validation loss for 5 epochs
    - ReduceLROnPlateau: Reduce LR by factor of 0.5 if validation accuracy plateaus

---

## Results
Final Model Performance
- Train Accuracy: ~99.2%
- Validation Accuracy: ~88.4% (best)
- Test Accuracy: 85.47%
- Test Loss: 0.4309

---

## Sample Confusion Matrix
The model performs very well overall, but confusion is notable in:
- Class 6 vs Class 8
- Class 5 vs Class 6
- Class 3 vs Class 2

## Accuracy and Loss Curves
- Training accuracy increases steadily.
- Validation accuracy plateaus after a point, suggesting limitations in generalization.

---

## Challenges Faced
### 1. Validation Accuracy Plateau
Despite using callbacks and a low learning rate, the validation accuracy plateaued around ~88% and did not show significant improvement in later epochs. This was addressed with ReduceLROnPlateau and EarlyStopping, which helped stabilize training.

### 2. Model Confusion in Similar Classes
Some classes (especially 6, 8, 5, and 2) visually resemble each other in grayscale. This caused frequent misclassifications in those categories, as seen in the confusion matrix and classification report.

### 3. Risk of Overfitting
With a relatively small dataset and a powerful model, overfitting was a potential issue. This was mitigated by:
- Adding Dropout (0.4)
- Batch Normalization layers
- Using Early Stopping
- Still, training accuracy reached ~99%, whereas test accuracy stayed at ~85%, showing a gap that could be narrowed with data augmentation or more varied data.

---

## Future Work
- Implement data augmentation to improve generalization
- Experiment with pretrained models using transfer learning
- Extend to multi-hand or color gesture datasets
- Deploy the model using TensorFlow Lite or Flask API

---

## ⭐️ Give it a Star

If you found this repo helpful or interesting, please consider giving it a ⭐️. It motivates me to keep learning and sharing!

---
