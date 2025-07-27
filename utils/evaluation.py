import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def evaluate_model(model, X_test, y_test, save_path):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()
    print(classification_report(y_test, y_pred))

def visualize_importance(model, features, save_path):
    importances = model.feature_importances_
    sns.barplot(x=importances, y=features)
    plt.title("Feature Importance")
    plt.savefig(save_path)
    plt.close()

def visualize_distribution(df, target_col, save_path):
    sns.countplot(x=df[target_col])
    plt.title("Class Distribution")
    plt.savefig(save_path)
    plt.close()
