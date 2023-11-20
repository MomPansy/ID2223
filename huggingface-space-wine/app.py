import gradio as gr
from PIL import Image
import requests 
import hopsworks 
import joblib 
import pandas as pd 

project = hopsworks.login() 
fs = project.get_feature_store() 

mr = project.get_model_registry()
lr_model = joblib.load(project.get_model_registry().get_model("linear_regression_model", version=3).download() + "/linear_regression_model.pkl")
dt_model = joblib.load(project.get_model_registry().get_model("decision_tree_model", version=2).download() + "/decision_tree_model.pkl")

print("Model downloaded")

def wine(type, 
         fixed_acidity, 
         volatile_acidity, 
         citric_acid, 
         residual_sugar, 
         chlorides,
         free_sulfur_dioxide, 
         total_sulfur_dioxide, 
         density, 
         ph,
         sulphates,
         alcohol):
    print("Calling function") 
    columns = ['type', 
               'fixed_acidity', 
               'volatile_acidity', 
               'citric_acid', 
               'residual_sugar',
               'chlorides', 
               'free_sulfur_dioxide', 
               'total_sulfur_dioxide', 
               'density', 
               'ph', 
               'sulphates',
               'alcohol']
    dt_df = pd.DataFrame([[type, 
                           fixed_acidity, 
                           volatile_acidity, 
                           citric_acid, 
                           residual_sugar, 
                           chlorides, 
                           free_sulfur_dioxide, 
                           total_sulfur_dioxide, 
                           density, 
                           ph,
                           sulphates, 
                           alcohol]], 
                           columns = columns)
    lr_df = dt_df.drop(['residual_sugar', 'ph', 'sulphates'], axis = 1)
    print("Predicting") 

    lr_res = lr_model.predict(lr_df)
    dt_res = dt_model.predict(dt_df)

    print(lr_res)
    print(dt_res) 

    return lr_res[0], dt_res[0]
        

demo = gr.Interface( 
    fn = wine, 
    title = "wine predictive analytics", 
    description = 'Experiment with features to predict the wine quality', 
    allow_flagging = 'never', 
    inputs = [
        gr.inputs.Number(default = 0, label = 'type (0 or 1)'), 
        gr.Slider(minimum=3.80000, maximum=15.90000 , value=7.216579, label='fixed_acidity'),
        gr.Slider(minimum=0.08000, maximum=1.58000, value=0.339691, label='volatile_acidity'),
        gr.Slider(minimum=0.00000, maximum= 1.66000, value=0.318722, label='citric_acid'),
        gr.Slider(minimum=0.60000, maximum= 65.80000, value=5.444326, label='residual_sugar'),
        gr.Slider(minimum=0.00900, maximum= 0.61100, value=0.056042, label="chlorides"),
        gr.Slider(minimum=1.00000, maximum= 289.00000, value=30.525319, label="free_sulfur_dioxide"),
        gr.Slider(minimum=6.00000, maximum= 440.00000, value=115.744574, label="total_sulfur_dioxide"),
        gr.Slider(minimum=0.98711, maximum= 1.03898, value=0.994697, label="density"),
        gr.Slider(minimum=2.72000, maximum= 4.01000, value=3.218395, label="ph"),
        gr.Slider(minimum=0.22000, maximum= 2.00000, value=0.531215, label="sulphates"),
        gr.Slider(minimum=8.00000, maximum= 14.90000, value=10.491801, label="alcohol")
    ], 
    outputs = [
         gr.Number(label='Linear Regression Prediction'),
        gr.Number(label='Decision Tree Prediction')
    ]
)

demo.launch(debug = True)