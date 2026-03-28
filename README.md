# 🫀 Intelligent Cardiovascular Disease Prediction System

## 📌 Project Overview
This repository contains a multi-model Machine Learning pipeline designed to predict the likelihood of cardiovascular disease in patients based on their physiological data. It serves as a comparative study between distance-based, probabilistic, and neural network algorithms.

## 🚀 Features
- **Data Preprocessing:** Standardized pipeline for medical tabular data.
- **Multi-Model Training:** Implements K-Nearest Neighbors (KNN), Naïve Bayes, and Artificial Neural Networks (ANN).
- **Performance Evaluation:** Compares models using Accuracy, Precision, Recall, and Confusion Matrices.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib
- **Dataset:** UCI Heart Disease Dataset

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/spy-lakshit/Machine-Learning-Project
   ```
2. Navigate to the directory and install dependencies:
   cd Cardio-ML-Predictor
pip install pandas numpy scikit-learn matplotlib

3. Run the Jupyter Notebook or Python script:
   python model_training.py

   📊 Results Summary
The system evaluates all three algorithms against a 20% holdout test set. (Update with your actual results)

ANN Accuracy: ~88%

KNN Accuracy: ~85%

Naïve Bayes Accuracy: ~82%
Due to the non-linear nature of medical vitals, the Artificial Neural Network provided the most robust predictive performance.

4. UI Idea: Simple HTML Frontend
If you want to impress your examiner, show them how a doctor would actually use this. Here is a clean, simple Bootstrap HTML frontend. 

*(To make this fully functional, you would use a lightweight Python framework like **Flask** or **FastAPI** to connect this HTML form to your trained ML model, but even showing the frontend concept gets you extra points).*

**`index.html`**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cardio Diagnostic AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f4f7f6; padding-top: 50px; }
        .card { border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .btn-predict { background-color: #d9534f; color: white; width: 100%; }
        .btn-predict:hover { background-color: #c9302c; color: white; }
    </style>
</head>
<body>

<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-6">
            <div class="card p-4">
                <h3 class="text-center mb-4">🫀 Patient Cardiac Assessment</h3>
                <form action="/predict" method="POST">
                    
                    <div class="mb-3">
                        <label class="form-label">Patient Age</label>
                        <input type="number" class="form-control" name="age" required>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">Resting Blood Pressure (mm Hg)</label>
                        <input type="number" class="form-control" name="trestbps" required>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">Serum Cholestoral (mg/dl)</label>
                        <input type="number" class="form-control" name="chol" required>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">Maximum Heart Rate Achieved</label>
                        <input type="number" class="form-control" name="thalach" required>
                    </div>

                    <button type="submit" class="btn btn-predict mt-3">Run AI Diagnosis</button>
                </form>
            </div>
        </div>
    </div>
</div>

</body>
</html>
