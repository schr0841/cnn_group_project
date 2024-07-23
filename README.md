# cnn_group_project
Medical Image Classification with Convolutional Neural Networks

Project outline: https://docs.google.com/document/d/1iy3c7ik2YjbI88vtGTbysj7AKmAjGtXfu4V3f6WxbHg/edit?usp=sharing


Explore pre-trained models (Xception, VGG19, ResNet50, MobileNet) - not sure useful for this case

Model (sort of) from scratch in tensorflow, get baseline accuracy

Optimization methods: image augmentation, dropout, early stopping



# Overview and Purpose

In this document, we train a convolutional neural network from scratch, and then investigate the added benefit of using pre-trained cnn models on the same dataset. Furthermore, we investigate model ensembling in general and its potential benefits and drawbacks.





From the Kaggle notebook (2) we have some code to work with for pre-trained models. 



## Table of results using CategoricalCrossEntropy loss function and class_mode='categorical' in data generator functions

| model | loss | accuracy | val_loss | val_accuracy |
|-------|------|----------|----------|--------------|
| Base cnn  | 6.9153  | 0.2414  | 7.6113 | 0.1806 |
| EfficientNetB3  | 0.0324 | 0.9837 | 0.8413 | 0.8063 |
| ResNet50 | 0.0145 | 0.9967 | 1.7889 | 0.7111 |
| InceptionV3 | 0.0157 | 0.9935 | 2.4461 | 0.5143 |
| Ensemble | 0.3838 | 0.9028 |   |   |

loss: 6.9153 - accuracy: 0.2414 - val_loss: 7.6113 - val_accuracy: 0.1806

loss: 0.0324 - accuracy: 0.9837 - val_loss: 0.8413 - val_accuracy: 0.8063

loss: 0.0145 - accuracy: 0.9967 - val_loss: 1.7889 - val_accuracy: 0.7111

loss: 0.0157 - accuracy: 0.9935 - val_loss: 2.4461 - val_accuracy: 0.5143

loss: 0.3838 - accuracy: 0.9028

Project topics:

* Investigate what pre-training means
* Sparse categorical vs categorical loss functions / sparse vs non-sparse class mode generators
* Investigate ensemble models
* Ensemble validation set - figure out why only 1 epoch / why validation accuracy so low
* Figure out what .h5 files are doing



## References

"Chest CT-Scan images Dataset" (2020) Retrieved from https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

"Ensemble model CT scan" (2024) Retrieved from https://www.kaggle.com/code/prthmgoyl/ensemblemodel-ctscan

"How to build CNN in TensorFlow: examples, code and notebooks" (2024) Retrieved from https://cnvrg.io/cnn-tensorflow/

"Convolutional Neural Network (CNN)" (2024) Retrieved from https://www.tensorflow.org/tutorials/images/cnn

"TensorFlow documentation: tf.keras.losses.SparseCategoricalCrossentropy" (2024) Retrieved from https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy

"What does class_mode parameter in Keras image_gen.flow_from_directory() signify?" (Jan. 2020) Retrieved from https://stackoverflow.com/questions/59439128/what-does-class-mode-parameter-in-keras-image-gen-flow-from-directory-signify