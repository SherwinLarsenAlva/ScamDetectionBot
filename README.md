# Scam Detection Bot

Welcome to the Scam Detection Bot repository! This project focuses on detecting whether a text message is ham, spam, or phishing using deep learning models. The project includes code for training, fine-tuning, and evaluating the models, as well as scripts for data processing and deployment.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Dataset](#dataset)

## Introduction
This project aims to provide a robust solution for detecting scam messages. The models are trained on a diverse dataset and fine-tuned for optimal performance. The code is designed to be easy to use and extend.

## Installation
To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/SherwinLarsenAlva/ScamDetectionBot.git
cd ScamDetectionBot
pip install -r requirements.txt 
```

## Usage
Training the Model
To train the model, run the following script:
```bash
python train_model.py
```

Fine-Tuning the Model
For fine-tuning the model on your specific dataset, use:
```bash
python fine_tune_model.py
```

## Models
The pre-trained models are available on Hugging Face:
ScamDetector (https://huggingface.co/SparkyPilot/ScamDetector/tree/main)

## Dataset
The dataset used for training and evaluation is available on Hugging Face:
Scam Detection Data (https://huggingface.co/datasets/SparkyPilot/scam-detection-data/tree/main)
