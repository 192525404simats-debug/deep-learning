from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Actual class labels
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 0, 2]

# Predicted class labels
y_pred = [0, 2, 2, 0, 1, 2, 1, 1, 2, 1, 0, 0]

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0, 1, 2])
disp.plot(cmap="Blues")
plt.title("Multi-Class Confusion Matrix")
plt.show()

# Print accuracy
print("\nAccuracy:", accuracy_score(y_true, y_pred))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
