# Pneumonia_xray
This project contains an implementation of a convolutional neural network to detect the presence of pneumonia in chest x-ray scans. The model classifies chest x-ray images as either "normal" or "pneumonia".

# Dependencies
This project uses the following modules and versions:
* Python 3.7.6
* PyTorch 1.5.0
* NVIDIA CUDA Toolkit 10.2.89
* NumPy 1.18.1
* Matplotlib 3.2.1
* Pandas 1.0.3

# Setup
1. Download the file 'ChestXRay2017.zip' from [here](https://data.mendeley.com/datasets/rscbjbr9sj/2).
2. Extract the folder 'chest_xray' to the relative directory of the files 'LoadTrainTest.py', 'NeuralNet.py', 'CreateValidationSet.py', and 'Functions.py'
3. Run the file 'CreateValidationSet.py' to create and populate the validation dataset folders.
4. Run the file 'LoadTrainTest.py' to load the data, train the neural network model, and test the model against the test dataset.
5. Optional hyperparameters can be changed to save the model, or change training time.

# Results

# References
* Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, v2
http://dx.doi.org/10.17632/rscbjbr9sj.2
