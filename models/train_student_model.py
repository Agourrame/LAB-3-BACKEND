import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load the dataset
student_data = pd.read_csv('educational_system/data/student-por.csv', sep=';')

# Encode categorical variables
student_data = pd.get_dummies(student_data, drop_first=True)

# Split the data
X = student_data.drop('G3', axis=1)
y = student_data['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Save the model and scaler
with open('educational_system/models/student_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('educational_system/models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
