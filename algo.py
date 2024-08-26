import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import lstIn
import lstOut

# Prepare the data
X = lstIn.input_numbers
y = lstOut.output_numbers

# Normalize the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Reshape input to be 3D [samples, time steps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
y_reshaped = y_scaled.reshape((y_scaled.shape[0], y_scaled.shape[1], 1))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Define the model
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(X_reshaped.shape[1], 1))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(y_reshaped.shape[1], 1))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(1)
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit([X_train, y_train], y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.2,
                    verbose=1)

# Make predictions
predictions_scaled = model.predict([X_test, y_test])

# Inverse transform the predictions
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, y.shape[1]))
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, y.shape[1]))

# Evaluate the model
mse = np.mean((predictions - y_test_original) ** 2)
rmse = np.sqrt(mse)
print(f"\nRoot Mean Squared Error: {rmse:.4f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.scatter(y_test_original.flatten(), predictions.flatten())
plt.plot([0, 100], [0, 100], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Print sample of predictions and real outputs
np.set_printoptions(suppress=True, formatter={'float_kind': '{:f}'.format})

print("\nSample of Predicted Outputs:")
print(predictions[:5])

print("\nSample of Real Outputs:")
print(y_test_original[:5])

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
