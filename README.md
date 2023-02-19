# Movie Rating Sentiment Analysis

## The Neural Network
This neural network predicts whether or not a movie review has a positive or negative tone. The model will predict a value close to 0 if the review is predicted to be negative value close to 1 if the review is predicted to be of a positive sentiment. Since the model only predicts binary categorical values, the model uses a binary crossentropy loss function and has 1 output neuron. The model uses a default RMSprop optimizer and uses early stopping to prevent overfitting. The model has an architecture consisting of:
- 1 Embedding layer (with an input size of 88584 and an output size of 32) 
- 1 LSTM layer (with 32 input neurons)
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset is built into Tensorflow and includes movie reviews (as inputs) and their corresponding tone (negative or positive, encoded as 0 and 1 respectively). As part of data preprocessing, each input string is standardized to a length of 250 words (padded with zeros) and comes pre-encoded.

## Libraries
This neural network was created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
