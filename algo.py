import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import lstIn
import lstOut

# Define constants for the range
MIN_VALUE = 0
MAX_VALUE = 4294967296

# Normalize function
def normalize(data, min_val=MIN_VALUE, max_val=MAX_VALUE):
    return (data - min_val) / (max_val - min_val)

# Denormalize function
def denormalize(data, min_val=MIN_VALUE, max_val=MAX_VALUE):
    return data * (max_val - min_val) + min_val

# Normalize the input and output data
input_numbers = normalize(lstIn.input_numbers)
output_numbers = normalize(lstOut.output_numbers)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_numbers, output_numbers, test_size=0.25, random_state=42)

# Reshape inputs to add time steps dimension (1 time step)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
y_test = y_test.reshape((y_test.shape[0], 1, y_test.shape[1]))

# Define the Seq2Seq model with attention
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Reshape
from tensorflow.keras.models import Model

# Encoder
encoder_inputs = Input(shape=(1, 8))
encoder = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(1, 8))
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
encoder_outputs = Reshape((1, 256))(encoder_outputs)
attention = Attention()
attention_outputs = attention([decoder_outputs, encoder_outputs])
decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention_outputs])
decoder_dense = Dense(8, activation='sigmoid')  # Use sigmoid to ensure output is between 0 and 1
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
decoder_input_data = np.zeros_like(X_train)  # Use zeros as the initial decoder inputs
model.fit([X_train, decoder_input_data], y_train, epochs=300, batch_size=4, validation_split=0.2)

# Generate predictions for the test set
decoder_input_data_test = np.zeros_like(X_test)
predictions = model.predict([X_test, decoder_input_data_test])

# Denormalize predictions
predictions = denormalize(predictions)

# Print predictions and real outputs as real numbers
np.set_printoptions(suppress=True, formatter={'float_kind': '{:f}'.format})

print("Predicted output:")
print(predictions)

print("Real output:")
print(denormalize(y_test))
