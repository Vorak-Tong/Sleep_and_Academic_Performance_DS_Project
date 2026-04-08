#To run it:
#pip install gradio
#cd Sleep_and_Academic_Performance
#py app.py 

import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Train model on startup
df = pd.read_csv("data/processed/student_lifestyle_cleaned.csv")

#Encode stress level
stress_map = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}
df["Stress_Level"] = df["Stress_Level"].map(stress_map)

feature_cols = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Stress_Level"
]

X = df[feature_cols]
y = df["GPA"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


def predict_gpa(study, sleep, social, physical, extra, stress):
    stress_value = stress_map[stress]
    features = np.array([[study, sleep, social, physical, extra, stress_value]])
    gpa = model.predict(features)[0]
    gpa = float(np.clip(gpa, 0.0, 4.0))
    return f"{gpa:.2f} / 4.00"


with gr.Blocks(title="GPA Predictor") as demo:
    gr.Markdown("# Student GPA Predictor")
    gr.Markdown(
        "Adjust your daily lifestyle hours and see the predicted GPA. "
        "Model: **Linear Regression** (Test R² = 0.55)"
    )

    with gr.Row():
        with gr.Column():
            study   = gr.Slider(5.0, 10.0, value=7.5, step=0.1, label="Study Hours per Day")
            sleep   = gr.Slider(5.0, 10.0, value=7.5, step=0.1, label="Sleep Hours per Day")
            social  = gr.Slider(0.0,  6.0, value=2.7, step=0.1, label="Social Hours per Day")
            physical = gr.Slider(0.0, 13.0, value=4.3, step=0.1, label="Physical Activity Hours per Day")
            extra   = gr.Slider(0.0,  4.0, value=2.0, step=0.1, label="Extracurricular Hours per Day")
            stress = gr.Dropdown(choices=["Low", "Moderate", "High"], value="Moderate", label="Stress Level")
            btn     = gr.Button("Predict GPA", variant="primary")

        with gr.Column():
            output = gr.Text(label="Predicted GPA", interactive=False)

    btn.click(
        fn=predict_gpa,
        inputs=[study, sleep, social, physical, extra, stress],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
