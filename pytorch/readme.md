# pytorch utils

* based on the code give during techinon course by RoyiFixelAlgorithms from it public repo at FixelAlgorithmsTeam/FixelCourses (MPL2);


## DataManipulation.py

1. **DownloadGDriveZip**
   - **Description**: Downloads a zip file from Google Drive using its file ID, deletes existing files in the list, unpacks the archive if it's in a supported format, and removes the archive file.

2. **DownloadDecompressGzip**
   - **Description**: Downloads a gzipped file from a given URL, decompresses it, and writes the decompressed content to a specified file.

3. **DownloadUrl**
   - **Description**: Downloads a file from a specified URL if the file does not already exist in the local filesystem.

4. **ConvertMnistDataDf**
   - **Description**: Reads and converts MNIST image and label files into numpy arrays representing the image data and corresponding labels.

5. **ConvertBBoxFormat**
   - **Description**: Converts bounding box coordinates from one format to another, supporting COCO, PASCAL VOC, and YOLO formats, given the image size.

6. **GenLabeldEllipseImg**
   - **Description**: Generates an image of specified size with a specified number of random ellipses, returns the image, labels (color channels), and bounding boxes in the specified format.


## DataVisualization.py


1. **PlotBinaryClassData**
   - **Description**: Plots binary 2D data as a scatter plot, with different colors for each class and optional axis title.

2. **Plot2DLinearClassifier**
   - **Description**: Plots a binary 2D classifier's decision boundary on a scatter plot of the data points, including accuracy.

3. **PlotMnistImages**
   - **Description**: Plots a grid of MNIST images from the provided data, with optional class labels and random or sequential selection.

4. **PlotLabelsHistogram**
   - **Description**: Plots a histogram of class labels, with optional class names and label rotation.

5. **PlotConfusionMatrix**
   - **Description**: Plots a confusion matrix from the true and predicted labels, with optional normalization, class labels, and additional scores in the title.

6. **PlotDecisionBoundaryClosure**
   - **Description**: Creates a closure function to plot decision boundaries of a classifier over a 2D grid.

7. **PlotRegressionData**
   - **Description**: Plots 1D regression data points on a scatter plot, with optional axis title.

8. **PlotRegressionResults**
   - **Description**: Plots ground truth vs. predicted values for regression results, with optional axis title.

9. **PlotScatterData**
   - **Description**: Plots 2D scatter data with optional labels and customizable marker size and edge color.

10. **PlotScatterData3D**
    - **Description**: Plots 3D scatter data with optional labels and colors, with support for 2D and 3D projections.

11. **PlotDendrogram**
    - **Description**: Plots a dendrogram for hierarchical clustering, with customizable linkage method and threshold level.

12. **PlotBox**
    - **Description**: Plots a bounding box on an image with an optional label and score, assuming YOLO format.

13. **PlotBBox**
    - **Description**: Plots a bounding box on an existing axis with an optional label and score, assuming YOLO format.



## DeepLearningBlocks.py

1. **LinearLayer**
   - **Description**: A linear (fully connected) layer for neural networks, supporting initialization, forward pass, and backward pass. Uses different weight initialization methods (CONST, KAIMING, XAVIER).

2. **DropoutLayer**
   - **Description**: Implements dropout regularization for neural networks, supporting training (forward pass with dropout mask) and inference (forward pass without dropout).

3. **ReLULayer**
   - **Description**: Implements the ReLU activation function for neural networks, with support for forward and backward passes.

4. **LeakyReLULayer**
   - **Description**: Implements the Leaky ReLU activation function for neural networks, with a customizable alpha value, supporting forward and backward passes.

5. **CrossEntropyLoss**
   - **Description**: Calculates the cross-entropy loss and its gradient for classification tasks, assuming input logits and one-hot encoded labels.

6. **MseLoss**
   - **Description**: Calculates the mean squared error (MSE) loss and its gradient for regression tasks.

7. **ModelNN**
   - **Description**: Represents a sequential neural network model, supporting initialization, forward pass, and backward pass. Handles different operation modes (TRAIN, INFERENCE).

8. **SGD**
   - **Description**: Implements the Stochastic Gradient Descent (SGD) optimization algorithm, with optional momentum and weight decay.

9. **Adam**
   - **Description**: Implements the Adam optimization algorithm, with support for momentum, adaptive learning rates, and weight decay.

10. **Optimizer**
    - **Description**: Manages optimization of neural network parameters using a specified optimization algorithm (SGD, Adam).

11. **DataSet**
    - **Description**: Represents a dataset for training/testing, supporting batching, shuffling, and iteration.

12. **TrainEpoch**
    - **Description**: Trains a neural network model for one epoch, updating model parameters using gradient descent. Calculates and returns the average loss and score over the epoch.

13. **ScoreEpoch**
    - **Description**: Evaluates a neural network model for one epoch without updating parameters, calculating and returning the average loss and score.

14. **RunEpoch**
    - **Description**: Runs one epoch of training or evaluation for a neural network model, using the specified optimizer and calculating average loss and score.

15. **ScoreAccLogits**
    - **Description**: Calculates classification accuracy from logits, assuming logits are monotonic with class probabilities.

16. **CountModelParams**
    - **Description**: Calculates the total number of parameters in a neural network model.


## DeepLearningPyTorch.py


1. **TBLogger**
   - **Description**: Manages TensorBoard logging, with initialization and closure functions.

2. **TestDataSet**
   - **Description**: Custom dataset class for loading image files from a directory, with optional transformations.

3. **ObjectLocalizationDataset**
   - **Description**: Custom dataset class for object localization tasks, returning image data, labels, and bounding boxes.

4. **ResetModelWeights**
   - **Description**: Resets the weights of a given PyTorch model, applying recursively to all children layers if necessary.

5. **InitWeights**
   - **Description**: Initializes weights of a linear layer using Kaiming normal initialization.

6. **InitWeightsKaiNorm**
   - **Description**: Initializes weights of specified layer types using Kaiming normal initialization.

7. **GenDataLoaders**
   - **Description**: Generates training and validation data loaders with specified batch size, number of workers, and other options.

8. **RunEpoch**
   - **Description**: Runs a single epoch of training or inference for a model, calculating and returning the average loss and score. Supports training with an optimizer.

9. **RunEpochSch**
   - **Description**: Runs a single epoch of training or inference with a learning rate scheduler, calculating and returning the average loss, score, and learning rate. Supports logging with TensorBoard.

10. **TrainModel**
    - **Description**: Trains a model over a specified number of epochs, using training and validation data loaders, an optimizer, and optionally a scheduler and TensorBoard writer. Returns the model and training/validation loss and score histories.

11. **TrainModelSch**
    - **Description**: Trains a model over a specified number of epochs with a learning rate scheduler, using training and validation data loaders, an optimizer, and optionally a TensorBoard logger. Returns the model and training/validation loss and score histories.

12. **ResidualBlock**
    - **Description**: Implements a simple residual block with two convolutional layers, batch normalization, and ReLU activation, adding the input tensor to the output (skip connection).