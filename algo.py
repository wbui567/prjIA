import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import lstIn
import lstOut
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model


# Normalize data (optional, helps with training)
input_data = lstIn.input_numbers / 4294967296
output_data = lstOut.output_numbers / 4294967296

# Define model parameters
input_dim = input_data.shape[1]
output_dim = output_data.shape[1]
latent_dim = 128  # Increased dimension of the hidden state

# Encoder
encoder_inputs = Input(shape=(input_dim, 1))
encoder = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# Add more layers to the encoder
encoder = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder(encoder_outputs)

encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(output_dim, 1))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Add more layers to the decoder
decoder_lstm = LSTM(latent_dim, return_sequences=True)
decoder_outputs = decoder_lstm(decoder_outputs)

decoder_dense = TimeDistributed(Dense(1))
decoder_outputs = decoder_dense(decoder_outputs)

# Define model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape input data to match RNN requirements (samples, timesteps, features)
input_data = np.expand_dims(input_data, axis=-1)
output_data = np.expand_dims(output_data, axis=-1)

# Train model with more epochs
model.fit([input_data, input_data], output_data, epochs=500, batch_size=4)

# Prediction
predictions = model.predict([input_data, input_data])

# Denormalize predictions
predictions = predictions * 4294967296

# Apply modulus operation to ensure predictions are in the correct range
predictions = np.mod(predictions, 4294967296)

# Print predictions
print(predictions)
