from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

from preprocess import preprocess

# Load and preprocess the dataset
X, y = preprocess("data/supply_chain_data.csv")

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
score = f1_score(y_test, predictions)

print("F1 score:", score)
print("\nClassification Report:")
print(classification_report(y_test, predictions))