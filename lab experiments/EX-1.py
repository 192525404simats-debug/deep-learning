from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Actual labels
y_true = ["Dog", "Dog", "Not Dog", "Dog", "Not Dog",
          "Dog", "Not Dog", "Dog", "Not Dog", "Dog"]

# Predicted labels
y_pred = ["Dog", "Not Dog", "Not Dog", "Dog", "Not Dog",
          "Dog", "Dog", "Dog", "Not Dog", "Dog"]

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=["Dog", "Not Dog"])

print("Confusion Matrix:")
print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Dog", "Not Dog"])
disp.plot(cmap="Blues")
plt.title("Dog vs Not Dog Confusion Matrix")
plt.show()
