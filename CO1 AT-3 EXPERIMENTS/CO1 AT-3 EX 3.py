# Building a Machine Learning Classification Model

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 1: Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Data Preprocessing (Feature Scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train the Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Test the Model
y_pred = model.predict(X_test)

# Step 6: Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Display Confusion Matrix
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0,1,2], iris.target_names)
plt.yticks([0,1,2], iris.target_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Display values inside the matrix
for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j],
                 ha='center',
                 va='center',
                 color='red')

plt.show()
