import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pyod.models.auto_encoder import AutoEncoder
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
encoder = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(y_reshaped.shape[1], 1))
decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(1, activation='relu')  # Added ReLU activation
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Custom loss function
def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    scale_factor = tf.reduce_mean(tf.math.log(tf.math.abs(y_true) + 1))
    return mse * scale_factor

# Compile the model
model.compile(optimizer='adam', loss=custom_loss)

# Train the model
history = model.fit([X_train, y_train], y_train, 
                    epochs=150,  # Increased epochs
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
plt.plot([0, np.max(y_test_original)], [0, np.max(y_test_original)], 'r--', lw=2)
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
