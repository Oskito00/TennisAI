from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import joblib
import pandas as pd
from pathlib import Path

def train_model():
    # Load the data
    features_df = pd.read_csv('process_data/processed_data/tennis_data_features.csv')
    labels = np.loadtxt('process_data/processed_data/tennis_data_labels.csv', delimiter=',', skiprows=1)
    
    # Separate image info from features
    image_info = features_df[['filename', 'full_path']]
    X = features_df.iloc[:, 2:].values  # Skip filename and full_path
    
    # Split the data
    X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
        X, labels, image_info, test_size=0.1, random_state=42
    )
    
    # Train separate SVR models for each coordinate
    models = []
    print("Training SVM models...")
    for i in range(4):  # 4 coordinates (x1, y1, x2, y2)
        print(f"\nTraining model for coordinate {i+1}/4")
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        with tqdm(total=100, desc=f"Coordinate {i+1} Progress") as pbar:
            model.fit(X_train, y_train[:, i])
            pbar.update(100)
        models.append(model)
    
    # Save models and test data
    joblib.dump(models, 'SVM/tennis_svm_model.joblib')
    joblib.dump({
        'X_test': X_test,
        'y_test': y_test,
        'image_info': info_test.reset_index(drop=True)
    }, 'SVM/test_data.joblib')
    
    return models, X_test, y_test, info_test

def load_model_and_data():
    models = joblib.load('SVM/tennis_svm_model.joblib')
    test_data = joblib.load('SVM/test_data.joblib')
    return models, test_data['X_test'], test_data['y_test'], test_data['image_info']

def predict(models, input_features):
    predictions = np.zeros((input_features.shape[0], 4))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(input_features)
    return predictions

def evaluate_model(models, X_test=None, y_test=None, image_info=None):
    if X_test is None or y_test is None or image_info is None:
        features_df = pd.read_csv('tennis_data_features.csv')
        X_test = features_df.iloc[:, 2:].values
        y_test = np.loadtxt('tennis_data_labels.csv', delimiter=',', skiprows=1)
        image_info = features_df[['filename', 'full_path']]
    
    predictions = predict(models, X_test)
    
    # Calculate errors
    mse_coords = np.mean((predictions - y_test) ** 2, axis=0)
    mae_coords = np.mean(np.abs(predictions - y_test), axis=0)
    
    results = {
        'overall_mse': np.mean(mse_coords),
        'overall_mae': np.mean(mae_coords),
        'coordinate_mse': {
            'x1': mse_coords[0],
            'y1': mse_coords[1],
            'x2': mse_coords[2],
            'y2': mse_coords[3]
        },
        'coordinate_mae': {
            'x1': mae_coords[0],
            'y1': mae_coords[1],
            'x2': mae_coords[2],
            'y2': mae_coords[3]
        },
        'predictions': predictions,
        'true_values': y_test,
        'image_info': image_info
    }
    
    return results

def analyze_bias_variance(models, X_test, y_test, image_info):
    """
    Analyze bias and variance of the SVM model predictions.
    
    Returns:
        dict: Contains bias and variance metrics and analysis
    """
    predictions = predict(models, X_test)
    
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

def visualize_predictions(evaluation_results, images_dir, output_dir='SVM/visualize_predictions', num_samples=9):
    """
    Visualize model predictions by drawing points on the original images
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import random
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get random sample indices
    n_samples = len(evaluation_results['predictions'])
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
        plt.scatter(true_coords[0][0], true_coords[0][1], 
                   c='lime', s=100, label='Top True')
        plt.scatter(pred_coords[0][0], pred_coords[0][1], 
                   c='orange', s=100, label='Top Predicted')
        
        # Plot bottom point (point 2)
        plt.scatter(true_coords[1][0], true_coords[1][1], 
                   c='darkgreen', s=100, label='Bottom True')
        plt.scatter(pred_coords[1][0], pred_coords[1][1], 
                   c='red', s=100, label='Bottom Predicted')
        
        plt.title(f"Image: {image_path.name}")
        plt.legend()
        
        # Save figure
        output_file = output_path / f"pred_{image_path.name}"
        plt.savefig(output_file)
        plt.close()
        
        print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    # Either train new models
    if not Path('SVM/tennis_svm_model.joblib').exists():
        models, X_test, y_test, image_info = train_model()
    else:
        models, X_test, y_test, image_info = load_model_and_data()
    
    # Evaluate the models
    evaluation = evaluate_model(models, X_test, y_test, image_info)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Overall MSE: {evaluation['overall_mse']:.4f}")
    print(f"Overall MAE: {evaluation['overall_mae']:.4f}")
    
    # Print some example predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(evaluation['predictions']))):
        print(f"\nImage: {evaluation['image_info'].iloc[i]['filename']}")
        print(f"Predicted coordinates: {evaluation['predictions'][i]}")
        print(f"True coordinates: {evaluation['true_values'][i]}")

    # Set your images directory
    images_dir = '/Users/oscaralberigo/Desktop/CDING/TennisAI/coco-annotator/datasets/tennis'

    # Visualize predictions
    visualize_predictions(evaluation, images_dir, output_dir='svm/visualize_predictions', num_samples=5)

    # Analyze bias and variance
    bias_variance = analyze_bias_variance(models, X_test, y_test, image_info)
    
    print("\nBias-Variance Analysis:")
    print(f"Average distance from true points: {bias_variance['avg_distance']:.4f}")
    print(f"Standard deviation of distances: {bias_variance['std_distance']:.4f}")
    print("\nBias for each coordinate:")
    for coord, value in bias_variance['bias'].items():
        print(f"{coord}: {value:.4f}")
    print("\nVariance for each coordinate:")
    for coord, value in bias_variance['variance'].items():
        print(f"{coord}: {value:.4f}")