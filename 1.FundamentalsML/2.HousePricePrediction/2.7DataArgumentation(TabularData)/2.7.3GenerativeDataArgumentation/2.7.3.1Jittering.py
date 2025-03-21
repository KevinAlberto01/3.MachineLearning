import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler

# --- Load data ---
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

#3.INITIAL DATA SUMMARY
print("Summary of first dates:")
print(df.describe())

# --- Preprocessing: Only numerical variables (the autoencoder works better with numbers)  --- Preprocessing: Only numerical variables (the autoencoder works better with numbers) ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = df[numeric_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# --- Define Autoencoder (Generative Model) ---
encoding_dim = 16  # You can adjust according to the number of columns
input_dim = X_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# --- Train Autoencoder ---
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)

# --- Generate new synthetic points ---
encoded_points = encoder.predict(X_scaled)

# Small random perturbation in the latent space
noise = np.random.normal(0, 0.05, encoded_points.shape)  # You can adjust the magnitude
synthetic_points_encoded = encoded_points + noise

# Decode back to the original space
decoder_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(decoder_input, decoder_layer(decoder_input))

synthetic_points = decoder.predict(synthetic_points_encoded)

# --- Invert the scaling ---
synthetic_points = scaler.inverse_transform(synthetic_points)

# --- Create DataFrame ---
df_synthetic = pd.DataFrame(synthetic_points, columns=numeric_cols)

# --- Handling categorical variables ---
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
df_categorical = df[categorical_cols]

# For the synthetic rows, we take random samples from the existing categories
df_categorical_synthetic = df_categorical.sample(n=df_synthetic.shape[0], replace=True, random_state=42).reset_index(drop=True)

# Join the numerical and categorical parts
df_final_augmented = pd.concat([df_synthetic.reset_index(drop=True), df_categorical_synthetic], axis=1)

# --- Save augmented dataset ---

output_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Jittering.csv'
df_final_augmented.to_csv(output_path, index=False)

# --- Final message ---
print(f"Generative Data Augmentation (Autoencoder) completed. File saved at: {output_path}")
print(f"Original size: {df.shape[0]} rows")
print(f"Augmented size: {df_final_augmented.shape[0]} rows")
