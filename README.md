# Myocardial-Infraction-using-PPG-signal-analysis
This project aims to predict the likelihood of Myocardial Infarction (MI) by analyzing Photoplethysmography (PPG) signals. The model uses a hybrid CNN-LSTM architecture to classify PPG signals into two categories: MI (Myocardial Infarction) and Normal.

Table of Contents
Project Overview

Dataset

Requirements

Installation

Usage

Model Architecture

Results

Contributing

License

Project Overview
The project involves preprocessing PPG signals, augmenting the dataset, and training a deep learning model to classify signals as either indicative of Myocardial Infarction (MI) or normal. The model combines Convolutional Neural Networks (CNN) for feature extraction and Long Short-Term Memory (LSTM) networks for capturing temporal dependencies in the PPG signals.

Dataset
The dataset used in this project is sourced from Kaggle. It contains PPG signals with corresponding labels indicating whether the signal corresponds to a patient with MI or a normal individual.

Dataset Link: PPG Dataset on Kaggle

Features: 2000 time steps of PPG signal values.

Labels: Binary classification (MI or Normal).

Requirements
To run this project, you need the following dependencies:

Python 3.7–3.10

TensorFlow 2.x

NumPy

Pandas

Scikit-learn

Matplotlib

Installation
Clone the repository:

bash
Copy
git clone https://github.com/your-username/mi-prediction-ppg.git
cd mi-prediction-ppg
Install the required dependencies:

bash
Copy
pip install -r requirements.txt
Download the dataset from Kaggle and place it in the project directory.

Usage
Preprocess the Data:

The PPG signals are normalized and reshaped into a 3D format (samples, time_steps, features).

Labels are converted to binary values (1 for MI, 0 for Normal).

Data Augmentation:

The training data is augmented by adding noise, applying time shifts, and scaling to improve model generalization.

Train the Model:

The model is trained using the augmented dataset.

The training process includes early stopping and learning rate reduction on plateau to prevent overfitting.

Evaluate the Model:

The model is evaluated on a test set, and performance metrics (accuracy, loss) are displayed.

Visualize Results:

Training and validation accuracy/loss curves are plotted to analyze model performance.

To run the project, execute the following command:

bash
Copy
python mi_prediction_ppg.py
Model Architecture
The model consists of the following layers:

Convolutional Layers:

Two 1D convolutional layers with 64 and 128 filters, respectively.

Dropout and L2 regularization are applied to prevent overfitting.

LSTM Layer:

An LSTM layer with 128 units to capture temporal dependencies.

Dense Layers:

A fully connected dense layer with 128 units and ReLU activation.

Dropout and L2 regularization are applied.

Output Layer:

A single unit with a sigmoid activation function for binary classification.

Results
Test Accuracy: The model achieves an accuracy of 89% on the test set.


