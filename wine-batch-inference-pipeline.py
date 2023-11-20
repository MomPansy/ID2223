import os
import modal
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np


LOCAL = False  # Set to False to deploy on the cloud using Modal

if LOCAL == False:
    stub = modal.Stub("batch_predict")
    wine_image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "pandas", "numpy", "scikit-learn == 1.1.1", "matplotlib", "dataframe_image", "datetime", "seaborn"])

    @stub.function(image=wine_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks_api_key"))
    def g():
        batch_inference()

def batch_inference():
    import hopsworks
    from sklearn.metrics import mean_squared_error
    from datetime import datetime
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    import seaborn as sb
    import matplotlib.pyplot as plt
    import dataframe_image as dfi
    from PIL import Image, ImageDraw

    # Log in to Hopsworks and get the feature store
    project = hopsworks.login()
    fs = project.get_feature_store() 

    # Fetch the models from the model registry
    wine_fg = fs.get_feature_group(name = 'wine_v2', version = 2)
    today = datetime.today().date()

    lr_model = joblib.load(project.get_model_registry().get_model("linear_regression_model", version = 3).download() + "/linear_regression_model.pkl")
    dt_model = joblib.load(project.get_model_registry().get_model("decision_tree_model", version = 2).download() + "/decision_tree_model.pkl")

    # Retrieve batch data from feature view
    query = wine_fg.filter(wine_fg.date_added == today).read(read_options={"use_hive": True})
    labels = query['quality']
    lr_batch_data = query.drop(['residual_sugar', 'ph', 'sulphates', 'date_added', 'id', 'quality'],axis = 1)
    dt_batch_data = query.drop(['date_added', 'id', 'quality'], axis = 1)

    # Make predictions using both models
    lr_predictions = lr_model.predict(lr_batch_data)
    dt_predictions = dt_model.predict(dt_batch_data)

    dataset_api = project.get_dataset_api()

    compare_predictions(labels, lr_predictions, dt_predictions, dataset_api)

    save_mse(labels, lr_predictions, dataset_api)

    save_model_performance(labels, dt_predictions, dataset_api)

def save_model_performance(labels, predictions, dataset_api, accuracy_path='classification_accuracy.png', matrix_path='confusion_matrix.png'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, confusion_matrix
    # Calculate and save classification accuracy
    accuracy = accuracy_score(labels, predictions)
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f'Classification Accuracy: {accuracy:.2f}', fontsize=15, ha='center')
    ax.axis('off')
    plt.savefig(accuracy_path, bbox_inches='tight')
    dataset_api.upload(accuracy_path, "Resources/images", overwrite = True)
    plt.close(fig)  # Close the figure to free memory

    # Compute and save confusion matrix heatmap

    cm = confusion_matrix(labels, predictions, labels=[3, 4, 5, 6, 7, 8, 9])

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=[3, 4, 5, 6, 7, 8, 9], yticklabels=[3, 4, 5, 6, 7, 8, 9])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plt.savefig(matrix_path, bbox_inches='tight')
    dataset_api.upload(matrix_path, "Resources/images", overwrite = True)
    plt.close()  # Close the figure to free memory


def save_mse(labels, lr_predictions, dataset_api):
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    test_mse = mean_squared_error(labels, lr_predictions)

    # Create a figure and an axis
    fig, ax = plt.subplots()

    # Hide the axes
    ax.axis('off')

    # Add text to the figure
    ax.text(0.5, 0.5, f'Test MSE: {test_mse}', fontsize=15, ha='center')

    # Save the figure
    plt.savefig('test_mse.png', bbox_inches='tight')

    dataset_api.upload("test_mse.png", "Resources/images", overwrite = True)


def compare_predictions(labels, lr_predictions, dt_predictions, dataset_api, file_path='predictions_comparison.png'):
    indices = np.arange(len(labels))  # Indices for x-axis

    plt.figure(figsize=(12, 10))

    # Plot for LR Predictions
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    plt.plot(indices, labels, label='Actual Labels', marker='o')
    plt.plot(indices, lr_predictions, label='LR Predictions', marker='x')
    plt.title('Actual Labels vs LR Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Labels / Predictions')
    plt.legend()

    # Plot for DT Predictions
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    plt.plot(indices, labels, label='Actual Labels', marker='o')
    plt.plot(indices, dt_predictions, label='DT Predictions', marker='x')
    plt.title('Actual Labels vs DT Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Labels / Predictions')
    plt.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')

    dataset_api.upload(file_path, "Resources/images", overwrite = True)
    plt.close()

if __name__ == "__main__":
    if LOCAL:
        batch_inference()
    else:
        with stub.run():
            g.remote()
