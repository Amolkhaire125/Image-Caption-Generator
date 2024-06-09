# Image Caption Generator

## Overview
This project aims to generate captions for images using deep learning techniques. The model takes an image as input and generates a descriptive caption that describes the contents of the image.

## Dataset
The dataset used for this project consists of images along with their corresponding captions. The images are located in the directory specified by `images_path`, and the captions are stored in the file specified by `captions_path`.

## Preprocessing
1. **Data Loading**: Load the images and their corresponding captions from the dataset.
2. **Text Preprocessing**: Tokenize the captions, remove punctuation, and convert text to lowercase.
3. **Feature Extraction**: Use a pre-trained InceptionV3 model to extract features from the images.

## Model Architecture
- **Encoder**: Utilize InceptionV3 as the image encoder to extract features from the input images.
- **Decoder**: Implement a decoder using LSTM layers to generate captions based on the extracted features.

## Training
- **Data Splitting**: Split the dataset into training and testing sets.
- **Word Embeddings**: Use pre-trained GloVe word embeddings to initialize the embedding layer.
- **Training Process**: Train the model using the training data and optimize it using the Adam optimizer.

## Evaluation
- Evaluate the performance of the model using metrics such as loss during training.
- Generate captions for test images and visually inspect the results for correctness.

## Dependencies
- Python 3.x
- NumPy
- pandas
- scikit-learn
- Keras
- TensorFlow
- OpenCV
- Matplotlib
