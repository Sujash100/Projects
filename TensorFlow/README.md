# ANN-in-Tensorflow
A feed forward ANN in Tensorflow to predict the handwritten digits in 28x28 pixel image

# Additional Packages
  You will need Pillow to input the test image, and also the numpy.
  
          pip3 install Pillow
          pip3 install numpy
            
          

# About the data set
  Data set has been imported from tensorflow's official website which contains total of 55000 handwritten digits images.
  
# ANN
  ANN has been trained upto 1000 iterations and after each 100 iterations, a batch of 128 images has been used to calculate the accuracy and that accuracy has been plotted in the Accuracy Graph as can be seen from tensorboard.
  This ANN contains 3 hidden layers, but we can remove/add one or more layer.
  After training 3 images has been given to the ANN to get the output, and those outputs are correctly predicted.
  
# Output of Console after training and predicting the 3 images

              =================Starting Traing============
              Iteration 0  Completed
              Iteration 100  Completed
              Iteration 200  Completed
              Iteration 300  Completed
              Iteration 400  Completed
              Iteration 500  Completed
              Iteration 600  Completed
              Iteration 700  Completed
              Iteration 800  Completed
              Iteration 900  Completed
              ===================Traing Completed===========
              Accuracy on test set: 0.9158
              Prediction for test image: 2
              Prediction for test image: 3
              Prediction for test image: 6
              
# Tensorboard Output of Accuracy and DataFlow Graph

<img src = "https://github.com/manugond/ANN-in-TensorFlow/blob/master/AccuracyOutput.JPG" width="800">

<img src = "https://github.com/manugond/ANN-in-TensorFlow/blob/master/DataFlowGraphOutput.JPG" width="800">
