# Pneumonia_xray
This project contains an implementation of a convolutional neural network to detect the presence of pneumonia in chest x-ray scans. The model classifies chest x-ray images as either "normal" or "pneumonia". The dataset is inherently unbalanced as 25.7% of the training dataset belongs to the "normal" class, and 37.5% of the test dataset belongs to the "normal" class. Due to the unbalanced nature of this dataset and the relative difficulty in differentiation between "normal" and "pneumonia" chest x-rays, this dataset presents a challenging binary classification problem.

# Dependencies
This project uses the following modules and versions:
* Python 3.7.6
* PyTorch 1.5.0
* NVIDIA CUDA Toolkit 10.2.89
* NumPy 1.18.1
* Matplotlib 3.2.1
* Seaborn 0.10.1
* Pandas 1.0.3

# Setup
1. Download the file 'ChestXRay2017.zip' from [here](https://data.mendeley.com/datasets/rscbjbr9sj/2).
2. Extract the folder 'chest_xray' to the relative directory of the files 'LoadTrainTest.py', 'NeuralNet.py', 'CreateValidationSet.py', and 'Functions.py'
3. Run the file 'CreateValidationSet.py' to create and populate the validation dataset folders.
4. Run the file 'LoadTrainTest.py' to load the data, train the neural network model, and test the model against the test dataset.
5. Optional hyperparameters can be changed to save the model, change batch size, or change training time.

# Results
The current implementation yields a test accuracy of 85.90%. The sensitivity of the model is 93.590% and the specificity of the model is 73.077%.

The following graph shows the count of correct and incorrect predictions for each class as a proportion of the the total images of each class in the test dataset.

![Graph of Per Class Accuracy](https://github.com/matthew-brinard/Pneumonia_xray/blob/master/PerClassAccuracy.png)

This graph depicts the change in accuracy on the training and validation datasets as training time increases.
![Graph of training and validation accuracy versus Epoch](https://github.com/matthew-brinard/Pneumonia_xray/blob/master/ModelAccuracy.png)

# References
* Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, v2
http://dx.doi.org/10.17632/rscbjbr9sj.2
