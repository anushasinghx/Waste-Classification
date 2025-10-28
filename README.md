# Waste Classification Using Deep Learning
This project is a deep learning model developed to classify waste images into two categories: Organic and Recyclable. The model is trained using a dataset of images, and the classification task helps in segregating waste efficiently. The project is built using TensorFlow and Keras, with a ResNet50 base model for image feature extraction.

## Key Features
- Image Data Augmentation: Augments the training images using rotation, zoom, horizontal/vertical flipping, and rescaling.
- Transfer Learning: Utilizes a pre-trained ResNet50 model with frozen layers for feature extraction.
- Binary Classification: The model predicts whether an image belongs to the Organic or Recyclable waste category.
- Callbacks: Early stopping and model checkpointing are used to prevent overfitting and save the best model during training.
- Evaluation: The model is evaluated on a test dataset to check its accuracy in classifying waste images.

## Results
The model achieves accurate classification of waste into Organic or Recyclable categories based on the provided dataset. Visualizations of the training process, including loss and validation metrics, are also included.
This project demonstrates the power of deep learning in solving real-world problems like waste classification.
