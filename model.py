import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import LinearRegression
from sklearn.metrics import accuracy_score

# Suppress the warning
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# Load the dataset
df = pd.read_csv('diabetes_Full_clean_data.csv')

# Separate features (X) and target (y)
X = df.drop(columns='Outcome', axis=1)
y = df['Outcome']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the training data without feature names
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler.fit_transform(X_train)

# Initialize and train the SVM classifier (Support vector machine)
clf = SVC(kernel='linear')
clf.fit(X_train_scaled, y_train)

linear = LinearRegression()
linear.fit(X_train_scaled,y_train)

# Evaluate the model on the training set
y_train_pred = linear.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train_pred, y_train)
print("Training Accuracy:", train_accuracy)

# Prepare new input data for prediction

input_sample = [6,148,72,35.0,0.0,33.6,0.627,29.0]
input_np_array = np.asarray(input_sample).reshape(1, -1)

# Standardize the new input data without feature names
input_sample_scaled = scaler.transform(input_np_array)

# Make prediction using the trained model
prediction = clf.predict(input_sample_scaled)

# Display the prediction result
if prediction[0] == 0:
    print("Person is not diabetic")
else:
    print("Person is diabetic")
