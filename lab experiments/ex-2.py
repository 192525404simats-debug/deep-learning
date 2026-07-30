from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Actual values
y_true = [1, 1, 1, 1, 0, 0, 1, 0, 1, 0]

# Predicted values
y_pred = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[0, 1])

disp.plot(cmap="Blues")
plt.title("2-Class Confusion Matrix")
plt.show()
