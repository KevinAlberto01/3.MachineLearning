#1.IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Ignore feature names warnings
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load data
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

# Separate features and target
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Codificación one-hot (columnas categóricas)
X = pd.get_dummies(X, drop_first=True)

# Scale the data (only features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the original dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Augmentation methods
def augment_original(X, y):
    return X, y


def augment_jittering(X, y, sigma=0.01):
    X_jittered = X.copy()
    num_cols = X_jittered.select_dtypes(include=[np.number]).columns.tolist()
    X_jittered[num_cols] += np.random.normal(0, sigma, X_jittered[num_cols].shape)
    return pd.concat([X, X_jittered]), pd.concat([y, y])


def augment_knn(X, y, n_neighbors=5, samples_per_instance=1):
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X)

    synthetic_samples = []
    synthetic_targets = []

    for i, x in enumerate(X.values):
        _, indices = knn.kneighbors([x])

        for _ in range(samples_per_instance):
            neighbor_idx = np.random.choice(indices[0][1:])
            neighbor = X.iloc[neighbor_idx].values
            synthetic_sample = x + np.random.rand() * (neighbor - x)
            synthetic_samples.append(synthetic_sample)
            synthetic_targets.append(y.iloc[i])

    X_synthetic = pd.DataFrame(synthetic_samples, columns=X.columns)
    y_synthetic = pd.Series(synthetic_targets)

    return pd.concat([X, X_synthetic]), pd.concat([y, y_synthetic])


def augment_generative(X, y, sigma=0.05):
    y_noisy = y + np.random.normal(0, sigma * y.std(), y.shape)
    return X, y_noisy


# Training and evaluation function
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


# Dictionary of Augmentation Methods
augmentation_methods = {
    "original": augment_original,
    "jittering": augment_jittering,
    "knn": augment_knn,
    "generative": augment_generative
}

# Store results
results = {}

# Evaluate each method
for method_name, augment_func in augmentation_methods.items():
    print(f"Evaluating: {method_name}")

    try:
        X_aug, y_aug = augment_func(X_train, y_train)
        mse = train_and_evaluate(X_aug, X_test, y_aug, y_test)
        results[method_name] = mse
        print(f"{method_name} - MSE: {mse:.2f}\n")
    except Exception as e:
        print(f"Error in {method_name}: {e}")
        results[method_name] = np.nan  # En caso de fallo, guardamos np.nan


# Show final summary
print("\nFinal MSE Comparison:")
for method, mse in results.items():
    print(f"{method}: {mse:.2f}")


# --- Real Bar Chart ---
# Ensure that all methods are present (if any failed, np.nan or 9999)
all_methods = ['original', 'jittering', 'knn', 'generative']
for method in all_methods:
    if method not in results:
        results[method] = np.nan  # You can change it to 9999 if you prefer

# Create graph
plt.figure(figsize=(10, 6))
bars = plt.bar(results.keys(), results.values(), color='skyblue', width=0.6)  # Ajuste de ancho de las barras

# Show the MSE value above each bar
for bar in bars:
    yval = bar.get_height()
    if not np.isnan(yval):
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 100, f"{yval:.0f}", ha='center', va='bottom')
    else:
        plt.text(bar.get_x() + bar.get_width() / 2, 5000, "Error", ha='center', va='bottom', color='red')

# Set up graph
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of Data Augmentation - MSE')
plt.xticks(rotation=45)

# Adjust the Y-axis range so everything fits without crowding
max_mse = max(filter(lambda x: not np.isnan(x), results.values()), default=5000)  # Obtain the maximum valid MSE
plt.ylim(0, max_mse + 500)  # Expand the Y-axis

# Add space between the bars and the MSE values
plt.tight_layout()
plt.show()
