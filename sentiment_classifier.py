import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# Initialize yperparameters
vocab_size = 88584
max_len = 250
batch_size = 64
dimensions = 32

# Load dataset (provided for by Keras)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = vocab_size) 

# Pad sequences to max_len with 0's to standardize input shape for the neural network
train_data = sequence.pad_sequences(train_data, max_len)
test_data = sequence.pad_sequences(test_data, max_len)

# Create model
model = Sequential()

# Input layer
model.add(Embedding(vocab_size, dimensions))

# Hidden layer
model.add(LSTM(dimensions))

# Output layer
model.add(Dense(1, activation = 'sigmoid')) # Sigmoid activation function so that values output a 1 or a 0 for positive or negative

# Compile model
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Train model and store training history
epochs = 5
history = model.fit(train_data, train_labels, epochs = epochs, validation_data = (test_data, test_labels), callbacks = [early_stopping])

# Visualize loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plt.plot(loss, label = 'Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['binary_accuracy']
val_accuracy = history_dict['val_binary_accuracy']

plt.plot(accuracy, label = 'Training Accuracy')
plt.plot(val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate model
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose = 0) # Change verbose to 1 or 2 for more information
print(f'Test accuracy: {test_acc * 100}%')


# Make predictions based on inputs
classes = ['negative', 'positive'] # The index of the class corresponds to its categorization
word_index = imdb.get_word_index()
input_index = {v : k for (k, v) in word_index.items()}

# Function to encode text inputs
def encode_text(text):
  token = tf.keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in token]
  return sequence.pad_sequences([tokens], max_len)[0]

# Function to decode inputs into text
def decode_text(input):
  pad = 0
  text = ""
  for val in input:
    if val != pad:
      text += input_index[val] + " "
  return text[:-1]

# Function to determine the model's certainty in predictions
def find_certainty(prediction):
  pred = 1 if prediction > 0.5 else 0
  if pred == 0:
    certainty = (1 - prediction) * 100
  else:
    certainty = prediction * 100
  return certainty, pred

# Function to make predictions
def predict(text):
  encoded = encode_text(text)
  prediction_input = np.zeros((1, 250)) # (1, 250) template because that is the length of inputs 
  prediction_input[0] = encoded # Insert input into template

  # Get prediction
  prediction = model.predict(prediction_input, verbose = 0)[0]
  certainty, pred_class = find_certainty(prediction)

  output_text = f"Model's Prediction ({certainty[0]}% certainty): {pred_class} ({classes[pred_class]})"
  return output_text

# Prediction vs. actual value (change the index to view a different input and output set in the test data)
index = 0
pred_prob = model.predict(test_data, verbose = 0)[index][0]
cert, predicted_class = find_certainty(pred_prob) # Get certainty and class predictions from the model
actual_class = int(test_labels[index]) # Get the actual class
text_input = decode_text(test_data[index])

print("Test Data Sample:")
print("   - Input: " + text_input)
print(f"   - Model's Prediction ({cert}% certainty): {predicted_class} ({classes[predicted_class]}) | Actual Class: {actual_class} ({classes[actual_class]})")

# Sample input data predictions
sample_text_positive = "That movie was good! I would definitely watch it again!"
print("\nPositive Data Sample (From User Input):")
print("   - Input: " + sample_text_positive)
print("   - " + predict(sample_text_positive))

sample_text_negative = "That movie sucked. I hated it. Wouldn't watch it again."
print("\nNegative Data Sample:")
print("   - Input: " + sample_text_negative)
print("   - " + predict(sample_text_negative))
