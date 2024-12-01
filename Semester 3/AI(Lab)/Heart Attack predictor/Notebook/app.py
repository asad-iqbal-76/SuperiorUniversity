import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

model = joblib.load("heart_model_rf.pkl")
scaler = joblib.load("scaler_rf.pkl")

def predict_risk():
    try:
        age = float(entries["entry_age"].get())
        sex = int(entries["entry_sex"].get())
        cp = int(entries["entry_cp"].get())
        trestbps = float(entries["entry_trestbps"].get())
        chol = float(entries["entry_chol"].get())
        fbs = int(entries["entry_fbs"].get())
        restecg = int(entries["entry_restecg"].get())
        thalach = float(entries["entry_thalach"].get())
        exang = int(entries["entry_exang"].get())
        oldpeak = float(entries["entry_oldpeak"].get())
        slope = int(entries["entry_slope"].get())
        ca = int(entries["entry_ca"].get())
        thal = int(entries["entry_thal"].get())
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_data)
        if hasattr(model, "predict_proba"):
            prediction_prob = model.predict_proba(input_scaled)[:, 1]  
        else:
            prediction_prob = [0.5]  
        prediction = model.predict(input_scaled) 
        result = "Heart Attack Risk" if prediction[0] == 1 else "No Heart Attack Risk"
        messagebox.showinfo("Prediction Result", f"Prediction: {result}\nProbability: {prediction_prob[0]:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("Heart Attack Prediction with Random Forest")
root.geometry("400x550")

fields = [
    ("Age:", "entry_age"),
    ("Sex (1: Male, 0: Female):", "entry_sex"),
    ("Chest Pain Type (0-3):", "entry_cp"),
    ("Resting Blood Pressure (trestbps):", "entry_trestbps"),
    ("Cholesterol (chol):", "entry_chol"),
    ("Fasting Blood Sugar (fbs, 1: >120mg/dl, 0: <=120mg/dl):", "entry_fbs"),
    ("Resting Electrocardiographic Results (restecg, 0-2):", "entry_restecg"),
    ("Maximum Heart Rate Achieved (thalach):", "entry_thalach"),
    ("Exercise Induced Angina (exang, 1: Yes, 0: No):", "entry_exang"),
    ("Oldpeak (Depression induced by exercise):", "entry_oldpeak"),
    ("Slope of Peak Exercise ST Segment (0-2):", "entry_slope"),
    ("Number of Major Vessels Colored by Fluoroscopy (ca, 0-3):", "entry_ca"),
    ("Thalassemia (thal, 3: Normal, 6: Fixed defect, 7: Reversible defect):", "entry_thal"),
]

entries = {}
for idx, (label_text, entry_name) in enumerate(fields):
    label = tk.Label(root, text=label_text)
    label.pack()
    entry = tk.Entry(root)
    entry.pack()
    entries[entry_name] = entry

predict_button = tk.Button(root, text="Predict Risk", command=predict_risk)
predict_button.pack()

root.mainloop()
