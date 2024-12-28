# Sea Creatures Classification Using Pre-trained MobileNetV2 Model

## 1. Project Overview
This project focuses on developing a deep learning model to classify images from the **"SeaDataset."** The dataset comprises labeled images of various sea creature categories. The model employs **MobileNetV2**, a pre-trained model on ImageNet, to enhance classification performance and leverage transfer learning techniques.

---

## 2. Introduction
The goal of this project is to build a robust image classification model that can accurately differentiate between various categories of sea creatures. Key features of the project include:

- Utilizing **MobileNetV2** for feature extraction.
- Addressing challenges such as **class imbalance** and **overfitting**.
- Employing data preprocessing, augmentation, and class weighting to ensure effective training.

This approach demonstrates the power of transfer learning, providing better alignment with the dataset characteristics and yielding insights into model optimization.

---

## 3. Objectives
1. Develop an image detection model capable of accurately categorizing images into predefined classes.
2. Handle data imbalance using augmentation techniques and class weighting.
3. Visualize and evaluate the model's performance on training, validation, and test datasets.
4. Demonstrate predictions on unseen test data to validate model robustness.

---

## 4. Dataset
- **Name:** SeaDataset  
- **Path:** `SeaDataset/SeaDataset/`  
- **Structure:** Organized into subdirectories, where each directory corresponds to a class.  

**Preprocessing Steps:**
- Images resized to **224x224 pixels** to match MobileNetV2 input requirements.
- Labels extracted as integers from folder names.

---

## 5. Methodology

### 5.1 Data Preparation
- **Loading:** The dataset is loaded using TensorFlow’s `image_dataset_from_directory` function.  
- **Shuffling and Splitting:**
  - Training Set: **70%** of the dataset.
  - Validation Set: **20%** of the dataset.
  - Test Set: **10%** of the dataset.
- **Data Augmentation:** Applied augmentations include random flipping, rotation, zooming, and translation to improve generalization.

### 5.2 Class Balancing
- Class weights computed and applied during training.
- Class distributions visualized for better understanding of imbalance.

### 5.3 Transfer Learning
- **Base Model:** MobileNetV2 pre-trained on ImageNet.
- **Fine-tuning:**
  - First 110 layers frozen to retain pre-trained features.
  - Remaining layers fine-tuned for the custom dataset.

### 5.4 Model Architecture
- **Input Layer:** Accepts `224x224x3` images.
- **Feature Extractor:** MobileNetV2 (excluding the top layer).
- **Pooling Layer:** Global average pooling.
- **Output Layer:** Fully connected dense layer with softmax activation for multi-class classification.

### 5.5 Training Configuration
- **Optimizer:** Adam with a learning rate of `1e-5`.
- **Loss Function:** Sparse Categorical Cross-Entropy.
- **Metrics:** Accuracy.
- **Callbacks:**
  - Early Stopping: Stops training if validation loss stagnates.
  - TensorBoard: Logs metrics for visualization.

---

## 6. Evaluation
Model performance is evaluated through:
1. **Training and Validation Metrics:** Accuracy and loss monitored during training.
2. **Test Accuracy:** Assessed on the test dataset after training.

### Visualizations:
- Accuracy and loss curves for training and validation.
- Class distribution in the dataset.
- Sample predictions with true and predicted labels displayed alongside images.

---

## 7. Deliverables
1. A trained MobileNetV2-based image classification model.
2. A detailed report with performance metrics and model insights.
3. Plots and visualizations illustrating training progress and test predictions.

---

## 8. Tools and Libraries
### Python Libraries:
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn

### Hardware Requirements:
- **GPU Acceleration** is recommended for efficient model training.

---

## 9. Expected Results
1. Accurate classification of images into their respective categories with high test accuracy.
2. Effective handling of data imbalance using augmentation and class weighting.
3. Detailed visualizations to understand model behavior.

---

## 10. How to Run the Code
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/sea-creatures-classification.git
   cd sea-creatures-classification

### 2. Required Libraries
To run the code, make sure to install the necessary libraries. These libraries are required for the project:

- **NumPy**: For numerical operations and array manipulations.
- **TensorFlow**: For building, training, and deploying deep learning models.
- **Matplotlib**: For creating visualizations and plots.
- **Scikit-learn**: For machine learning utilities like computing class weights.

To install the required libraries, use the following command:

```bash
pip install numpy tensorflow matplotlib scikit-learn

