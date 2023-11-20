import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/classification_accuracy.png", overwrite = True)
dataset_api.download("Resources/images/confusion_matrix.png", overwrite = True)
dataset_api.download("Resources/images/test_mse.png", overwrite = True)
dataset_api.download("Resources/images/predictions_comparison.png", overwrite = True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Label("Today's Predictions Comparison")
            input_img = gr.Image("predictions_comparison.png", elem_id="predictions-comparison")      
    with gr.Column():
            gr.Label("Today's LR Test MSE")
            input_img = gr.Image("test_mse.png", elem_id="test-mse")  
    with gr.Row():
        with gr.Column():
            gr.Label("Today's DT Classification Accuracy")
            input_img = gr.Image("classification_accuracy.png", elem_id="classification-accuracy")
        with gr.Column():          
            gr.Label("Confusion Maxtrix with Historical Prediction Performance")
            input_img = gr.Image("confusion_matrix.png", elem_id="confusion-matrix")        

demo.launch()
