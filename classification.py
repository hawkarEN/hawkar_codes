
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
import sklearn.svm as svm
from sklearn.naive_bayes import GaussianNB
import sklearn.neighbors as neighbors

# Load data
data = pd.read_csv('D:/.........csv')
y = data['class_label']
X = data.drop('class_label', axis=1)

# Feature selection using Chi-Squared test
chi2_selector = SelectKBest(chi2, k=800)
X_selected = chi2_selector.fit_transform(X, y)

# Split data into training + validation (80%) and test (20%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_selected, y, train_size=0.8, random_state=42)

# Further split training data into actual training (87.5% of training data) and validation (12.5% of training data)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.875, random_state=42)

# Define classifiers
classifiers = {
    'SVM-rbf': svm.SVC(probability=True, C=1, gamma=0.01),
    'SVM-poly': svm.SVC(kernel='poly', degree=5, C=1, probability=True),
     'KNN': neighbors.KNeighborsClassifier(n_neighbors=5),
     'Decision Tree': DecisionTreeClassifier(max_depth=10),
     'Naive Bayes': GaussianNB(),
}

# Initialize dictionaries to store evaluation metrics
results = {
    'Classifier': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': [],
    'AUC': []
}

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, name, normalize=False, title='Confusion Matrix', cmap=plt.cm.Dark2_r):
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'{title} - {name}', fontsize=14)  # Increase title font size
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)  # Increase x-axis tick font size
    plt.yticks(tick_marks, classes, fontsize=12)  # Increase y-axis tick font size

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=18)  # Increase text font size within the matrix

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)  # Increase y-axis label font size
    plt.xlabel('Predicted label', fontsize=14)  # Increase x-axis label font size
    plt.show()

# Train and evaluate classifiers
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    results['Classifier'].append(name)
    results['Accuracy'].append(accuracy_score(y_test, y_pred))
    results['Precision'].append(precision_score(y_test, y_pred, average='weighted'))
    results['Recall'].append(recall_score(y_test, y_pred, average='weighted'))
    results['F1-score'].append(f1_score(y_test, y_pred, average='weighted'))
    results['AUC'].append(roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A')

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=np.unique(y), name=name, normalize=True)

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)

# Plot ROC curves for each classifier
plt.figure(figsize=(10, 8))

# Define a dictionary for the line styles for each classifier
line_styles = {
    'SVM-rbf': 'solid',
    'SVM-poly': 'solid',
    'KNN': 'solid',
    'Decision Tree': 'dotted',
    'Random Forest': 'solid',
    'Naive Bayes': 'dashdot',
}

# Define a dictionary for colors to represent each classifier
color_dict = {
    'SVM-rbf': 'blue',
    'SVM-poly': 'orange',
    'KNN': 'green',
    'Decision Tree': 'black',
    'Random Forest': 'purple',
    'Naive Bayes': 'brown',
}

# Plot the ROC curves
for name, model in classifiers.items():
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        # Specify the positive class label explicitly if your y_test contains string labels
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label='Malignant')  # Adjust 'Malignant' as needed
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color_dict[name], linestyle=line_styles[name],
                 label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'r--', label='Chance (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

