from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import joblib
import pandas as pd

def train_model():
    # Load the data
    features_df = pd.read_csv('process_data/processed_data/tennis_data_features.csv')
    labels = np.loadtxt('process_data/processed_data/tennis_data_labels.csv', delimiter=',', skiprows=1)
    
    # Separate image info from features
    image_info = features_df[['filename', 'full_path']]
    X = features_df.iloc[:, 2:].values  # Skip the first two columns (filename and full_path)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
        X, labels, image_info, test_size=0.1, random_state=42
    )
    
    # Train the model
    print("Training model...")
    model = LinearRegression()
    with tqdm(total=100, desc="Training Progress") as pbar:
        model.fit(X_train, y_train)
        pbar.update(100)
    
    # Save the model and test data
    joblib.dump(model, 'linear_regression/tennis_model.joblib')
    joblib.dump({
        'X_test': X_test,
        'y_test': y_test,
        'image_info': info_test.reset_index(drop=True)
    }, 'test_data.joblib')
    
    return model, X_test, y_test, info_test

def load_model_and_data():
    # Load the saved model and test data
    model = joblib.load('linear_regression/tennis_model.joblib')
    test_data = joblib.load('linear_regression/test_data.joblib')
    return model, test_data['X_test'], test_data['y_test'], test_data['image_info']

def predict(model, input_features):
    # Make predictions using the model
    return model.predict(input_features)

def evaluate_model(model, X_test=None, y_test=None, image_info=None):
    """
    Evaluate the model's performance using MSE and custom metrics for each coordinate.
    
    Args:
        model: Trained LinearRegression model
        X_test: Test features
        y_test: Test labels
        image_info: DataFrame containing filename and full_path for test images
    
    Returns:
        dict: Dictionary containing various error metrics and image information
    """
    if X_test is None or y_test is None or image_info is None:
        # Load all data if not provided
        features_df = pd.read_csv('tennis_data_features.csv')
        X_test = features_df.iloc[:, 2:].values
        y_test = np.loadtxt('tennis_data_labels.csv', delimiter=',', skiprows=1)
        image_info = features_df[['filename', 'full_path']]
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate MSE for each coordinate
    mse_x1 = np.mean((predictions[:, 0] - y_test[:, 0]) ** 2)
    mse_y1 = np.mean((predictions[:, 1] - y_test[:, 1]) ** 2)
    mse_x2 = np.mean((predictions[:, 2] - y_test[:, 2]) ** 2)
    mse_y2 = np.mean((predictions[:, 3] - y_test[:, 3]) ** 2)
    
    # Calculate average absolute error for each coordinate
    mae_x1 = np.mean(np.abs(predictions[:, 0] - y_test[:, 0]))
    mae_y1 = np.mean(np.abs(predictions[:, 1] - y_test[:, 1]))
    mae_x2 = np.mean(np.abs(predictions[:, 2] - y_test[:, 2]))
    mae_y2 = np.mean(np.abs(predictions[:, 3] - y_test[:, 3]))
    
    # Calculate overall MSE and MAE
    overall_mse = np.mean((predictions - y_test) ** 2)
    overall_mae = np.mean(np.abs(predictions - y_test))
    
    # Create results dictionary with predictions and image info
    results = {
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'coordinate_mse': {
            'x1': mse_x1,
            'y1': mse_y1,
            'x2': mse_x2,
            'y2': mse_y2
        },
        'coordinate_mae': {
            'x1': mae_x1,
            'y1': mae_y1,
            'x2': mae_x2,
            'y2': mae_y2
        },
        'predictions': predictions,
        'true_values': y_test,
        'image_info': image_info
    }
    
    return results

def analyze_bias_variance(model, X_test, y_test, image_info):
    # Calculate bias and variance for each coordinate
    """
    Analyze bias and variance of the linear regression model predictions.
    
    Returns:
        dict: Contains bias and variance metrics and analysis
    """
    predictions = model.predict(X_test)
    
    # Calculate errors for each point
    errors = predictions - y_test
    
    # Calculate bias (average error) for each coordinate
    bias = np.mean(errors, axis=0)
    
    # Calculate variance (spread of predictions) for each coordinate
    variance = np.var(errors, axis=0)
    
    # Calculate average distance between predicted and true points
    distances = np.sqrt(
        (predictions[:, 0] - y_test[:, 0])**2 + 
        (predictions[:, 1] - y_test[:, 1])**2
    )
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    results = {
        'bias': {
            'x1': bias[0],
            'y1': bias[1], 
            'x2': bias[2],
            'y2': bias[3]
        },
        'variance': {
            'x1': variance[0],
            'y1': variance[1],
            'x2': variance[2], 
            'y2': variance[3]
        },
        'avg_distance': avg_distance,
        'std_distance': std_distance
    }
    
    return results

def visualize_predictions(evaluation_results, images_dir, output_dir='linear_regression/visualize_predictions', num_samples=9):
    """
    Visualize model predictions by drawing points on the original images and saving to output directory
    
    Args:
        evaluation_results: Dictionary containing model evaluation results
        images_dir: Directory containing the original images
        output_dir: Directory to save visualization results
        num_samples: Number of random samples to visualize
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import random
    from pathlib import Path
    import os
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get random sample indices
    n_samples = len(evaluation_results['predictions'])
    print(f"Number of samples: {n_samples}")
    sample_indices = random.sample(range(n_samples), min(num_samples, n_samples))
    
    for idx in sample_indices:
        # Get image path and load image
        image_path = Path(images_dir) / evaluation_results['image_info'].iloc[idx]['filename']
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue
            
        # Load and convert image
        img = Image.open(image_path).convert('RGB')
        img_width, img_height = img.size
        
        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
        # Get predictions and true values
        pred = evaluation_results['predictions'][idx]
        true = evaluation_results['true_values'][idx]
        
        # Scale coordinates back to image dimensions
        pred_coords = [
            (pred[0] * img_width, pred[1] * img_height),
            (pred[2] * img_width, pred[3] * img_height)
        ]
        true_coords = [
            (true[0] * img_width, true[1] * img_height),
            (true[2] * img_width, true[3] * img_height)
        ]
        
        # Plot predictions in red
        plt.scatter([pred_coords[0][0], pred_coords[1][0]], 
                   [pred_coords[0][1], pred_coords[1][1]], 
                   c='red', s=100, label='Predicted')
        
        # Plot true values in green
        plt.scatter([true_coords[0][0], true_coords[1][0]], 
                   [true_coords[0][1], true_coords[1][1]], 
                   c='green', s=100, label='True')
        
        plt.title(f"Image: {image_path.name}")
        plt.legend()
        
        # Save figure to output directory
        output_file = output_path / f"pred_{image_path.name}"
        plt.savefig(output_file)
        plt.close()
        
        print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    # Either train a new model
    # model, X_test, y_test, image_info = train_model()
    if not Path('linear_regression/tennis_model.joblib').exists():
        model, X_test, y_test, image_info = train_model()
    # Or load an existing model
    else:
        model, X_test, y_test, image_info = load_model_and_data()
    
    # Evaluate the model
    evaluation = evaluate_model(model, X_test, y_test, image_info)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Overall MSE: {evaluation['overall_mse']:.4f}")
    print(f"Overall MAE: {evaluation['overall_mae']:.4f}")
    
    # Print some example predictions with their corresponding image info
    print("\nSample Predictions:")
    for i in range(min(5, len(evaluation['predictions']))):
        print(f"\nImage: {evaluation['image_info'].iloc[i]['filename']}")
        print(f"Predicted coordinates: {evaluation['predictions'][i]}")
        print(f"True coordinates: {evaluation['true_values'][i]}")

    images_dir = '/Users/oscaralberigo/Desktop/CDING/TennisAI/coco-annotator/datasets/tennis'

    # Visualize predictions
    visualize_predictions(evaluation, images_dir, output_dir='linear_regression/visualize_predictions', num_samples=5)

    # Analyze bias and variance
    bias_variance = analyze_bias_variance(model, X_test, y_test, image_info)
    
    print("\nBias-Variance Analysis:")
    print(f"Average distance from true points: {bias_variance['avg_distance']:.4f}")
    print(f"Standard deviation of distances: {bias_variance['std_distance']:.4f}")
    print("\nBias for each coordinate:")
    for coord, value in bias_variance['bias'].items():
        print(f"{coord}: {value:.4f}")
    print("\nVariance for each coordinate:")
    for coord, value in bias_variance['variance'].items():
        print(f"{coord}: {value:.4f}")