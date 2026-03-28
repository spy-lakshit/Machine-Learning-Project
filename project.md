Title: Intelligent Cardiovascular Disease Prediction and Diagnostic System
Problem Statement: Cardiovascular diseases are a leading cause of mortality globally. Analyzing complex, multi-dimensional medical records manually is time-consuming and prone to human error. Hospitals need a reliable, automated second-opinion system to identify high-risk patients early.
Objective: To build and evaluate a multi-model classification engine that predicts the likelihood of heart disease in a patient. By comparing different algorithms, the system identifies the most reliable model for medical diagnosis.
Algorithms Used:
1. Artificial Neural Network (ANN): For complex, non-linear pattern recognition in patient vitals.
2. Naïve Bayes Classifier: For rapid probabilistic baseline predictions based on conditional independence.
3. K-Nearest Neighbors (KNN): To classify new patients based on historical cases with similar physiological profiles.
Dataset Suggestion: The standard UCI heart.csv dataset  (contains features like age, cholesterol, blood pressure, and a binary output / target variable indicating disease presence).

Complete Python Code:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Load Dataset
# Ensure you have a 'heart.csv' file in your directory
url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv" # Sample real-world link
data = pd.read_csv(url)

# 2. Preprocessing
# Split features and target
X = data.drop('target', axis=1) # Target column is often named 'target' or 'output'
y = data['target']

# Split dataset into training and testing (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (crucial for ANN and KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Initialize Models
models = {
    "Naïve Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Artificial Neural Network": MLPClassifier(hidden_layer_sizes=(30, 20), max_iter=2000, random_state=42)
}

# 4. Train, Predict, and Evaluate
results = {}

print("--- Diagnostic System Evaluation ---\n")
for name, model in models.items():
    # Use scaled data for ANN and KNN, regular for Naive Bayes (though scaled is fine too)
    train_data = X_train_scaled if name != "Naïve Bayes" else X_train
    test_data = X_test_scaled if name != "Naïve Bayes" else X_test
    
    # Train
    model.fit(train_data, y_train)
    
    # Predict
    y_pred = model.predict(test_data)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    results[name] = acc
    print(f"[{name}]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

# 5. Conclusion
best_model = max(results, key=results.get)
print(f"🏆 Best performing model for this medical dataset: {best_model} with {(results[best_model]*100):.2f}% accuracy.")

Step-by-Step Explanation:
1. Data Loading & Splitting: We load the heart dataset and separate the input features (X) from the output label (y) . We reserve 20% of the data strictly for testing.
2. Standardization: KNN relies on distance metrics, and ANNs rely on gradient descent. Scaling ensures features with large numbers (like cholesterol) don't overpower smaller ones.
3. Model Training: We iterate through the three algorithms, fitting them to the training data.
4. Evaluation: We generate predictions and calculate Accuracy, Precision, and Recall.

Output Explanation & Metrics:
1. Accuracy: Overall correctness.
2. Precision: Out of all patients predicted to have heart disease, how many actually do? (Crucial to avoid false alarms).
3. Recall: Out of all actual heart disease cases, how many did the model find? (In healthcare, high recall is vital so you don't miss sick patients).
4. Confusion Matrix: Shows True Positives, True Negatives, False Positives, and False Negatives.

Future Improvements:
1. Implement a Voting Classifier (Ensemble) that takes the majority vote of all three models.
2. Deploy as a simple web app using Streamlit or Flask where a doctor can input patient vitals.

Viva Questions & Answers:
Q: Difference between ANN and traditional algorithms? 
A: Traditional algorithms (like Naïve Bayes) rely on statistical probability or distance rules. ANNs learn complex, non-linear representations through hidden layers and backpropagation, simulating human brain neurons .
Q: What is the role of K in KNN?
A: 'K' represents the number of nearest neighbors checked to determine the class of a new data point. A small K is sensitive to noise, while a large K smooths out predictions.
Q: Why is Naïve Bayes considered "Naïve"? 
A: It assumes that all input features are independent of one another, which is rarely perfectly true in real life (e.g., age and blood pressure are usually correlated).
