import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from time import time
import os
import kagglehub

# Download dataset
path = kagglehub.dataset_download("muhammadshahidazeem/customer-churn-dataset")
print("Path to dataset files:", path)

# Load dataset
dataset_path = os.path.join(path)
train_file = os.path.join(dataset_path, "customer_churn_dataset-training-master.csv")
df = pd.read_csv(train_file)

# Clean and prepare data
df.drop(['CustomerID'], axis=1, inplace=True)
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Subscription Type', 'Contract Length'], drop_first=False)

# Split data
x = df.drop('Churn', axis=1)
y = df['Churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model parameters
input_size = x_train.shape[1]
batch_size = 200
hidden1, hidden2, hidden3 = 400, 20, 10
epochs = 10

# Build model
model = Sequential([
    Dense(hidden1, input_dim=input_size, activation='relu'),
    Dense(hidden2, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

# Train model
tic = time()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
toc = time()
print(f"Model training time: {toc - tic:.2f} sec")

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

# Predict and evaluate
pred = (model.predict(x_test) > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred))
# Save model
model.save("churn_model.keras")