# Skin-Cancer-Dermatoscopic-Image-Classification

Skin cancer , the most common human malignancy, is primarily diagnosed visually. The HAM 10000 dataset, which contains the dermatoscopic images from different
populations, acquired and stored by different modalities. This project chooses this as the topic to classify the images into diverse types of skin cancer. The data for
each type of skin cell is unbalanced, this project used various image transformation methods to apply augmentation, in order to increase the diversity of each type of image
without losing features, and applied transfer learning with the base model of vgg19 and resnet50 to develop a convolutional neural network model for skin cancer image classification, using Amazon EC2 P3 instances and AWS Deep Learning AMIs. 
The best result is of 82% accuracy.

### Data Source:
The HAM10000 Dataset: 
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Skin Cancer MNIST: HAM10000: 
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/home

### Algorithms:

1.	CNN (Convolutional neural network)
In neural networks, Convolutional neural network (ConvNets or CNNs) is one of the main categories to do images recognition, images classifications. Objects detections, recognition faces etc., are some of the areas where CNNs are widely used. Technically, deep learning CNN models to train and test, each input image will pass it through a series of convolution layers with filters (Kernals), Pooling, fully connected layers (FC) and apply Softmax function to classify an object with probabilistic values between 0 and 1. The below figure is a complete flow of CNN to process an input image and classifies the objects based on values.

2.	Transfer Learning
In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest. The three major Transfer Learning scenarios look as follows:

    <1> ConvNet as fixed feature extractor
    <2> Fine-tuning the ConvNet
    <3> Pretrained models

3.	Two popular Models
    
    <1> VGGNet Model
    <2> ResNet Model
