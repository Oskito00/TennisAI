from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from PIL import Image
import joblib
import pandas as pd


def train_model(alpha=1.0, use_wavelets=True):
    # Load the data
    if use_wavelets:
        features_df = pd.read_csv('process_data/cropped_data/wavelet_features.csv')
    else:
        features_df = pd.read_csv('process_data/cropped_data/cropped_features.csv')
    
    labels = np.loadtxt('process_data/cropped_data/cropped_labels.csv', delimiter=',', skiprows=1)
    
    # Separate image info from features
    image_info = features_df.iloc[:, :2]  # Keep first two columns for image info
    X = features_df.iloc[:, 2:].values    # Use all columns after the first two for features
    
    # Split the data
    X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
        X, labels, image_info, test_size=0.1, random_state=42
    )
    
    # Train the model with Ridge regression (alpha=10000)
    print("Training model...")
    model = Ridge(alpha=alpha)
    with tqdm(total=100, desc="Training Progress") as pbar:
        model.fit(X_train, y_train)
        pbar.update(100)
    
    # Save the model and test data
    joblib.dump(model, 'linear_regression/tennis_model.joblib')
    joblib.dump({
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train,
        'image_info': info_test.reset_index(drop=True)
    }, 'linear_regression/test_data.joblib')
    
    return model, X_test, y_test, X_train, y_train, info_test

def load_model_and_data():
    # Load the saved model and test data
    model = joblib.load('linear_regression/tennis_model.joblib')
    test_data = joblib.load('linear_regression/test_data.joblib')
    return model, test_data['X_test'], test_data['y_test'], test_data['X_train'], test_data['y_train'], test_data['image_info']

def predict(model, input_features):
    # Make predictions using the model
    return model.predict(input_features)

def evaluate_model(model, X_test=None, y_test=None, image_info=None, use_wavelets=True):
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
        if use_wavelets:
            features_df = pd.read_csv('process_data/cropped_data/wavelet_features.csv')
        else:
            features_df = pd.read_csv('process_data/cropped_data/cropped_features.csv')
        
        X_test = features_df.iloc[:, 1:].values
        y_test = np.loadtxt('process_data/cropped_data/cropped_labels.csv', delimiter=',', skiprows=1)
        image_info = features_df[['filename']]
    
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

def visualize_predictions(evaluation_results, images_dir, output_dir='predictions'):
    """
    Visualize predictions vs ground truth for all images in the evaluation results.
    
    Args:
        evaluation_results (dict): Dictionary containing evaluation data
        images_dir (str): Directory containing the original images
        output_dir (str): Directory to save visualization results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    total_images = len(evaluation_results['image_info'])
    print(f'Visualizing predictions for {total_images} images...')
    
    for idx in range(total_images):
        try:
            # Load image
            image_path = Path(images_dir) / evaluation_results['image_info'].iloc[idx]['filename']
            image = Image.open(image_path)
            width, height = image.size
            
            # Create figure and axis
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            
            # Get coordinates and scale them to image dimensions
            true_coords = evaluation_results['true_values'][idx]
            pred_coords = evaluation_results['predictions'][idx]
            
            # Scale coordinates to image dimensions
            true_coords_scaled = [
                true_coords[0] * width,   # x1
                true_coords[1] * height,  # y1
                true_coords[2] * width,   # x2
                true_coords[3] * height   # y2
            ]
            
            pred_coords_scaled = [
                pred_coords[0] * width,   # x1
                pred_coords[1] * height,  # y1
                pred_coords[2] * width,   # x2
                pred_coords[3] * height   # y2
            ]
            
            # Plot points
            plt.scatter(true_coords_scaled[0], true_coords_scaled[1], c='g', marker='o', s=100, label='True Top')
            plt.scatter(true_coords_scaled[2], true_coords_scaled[3], c='g', marker='s', s=100, label='True Bottom')
            plt.scatter(pred_coords_scaled[0], pred_coords_scaled[1], c='r', marker='o', s=100, label='Predicted Top')
            plt.scatter(pred_coords_scaled[2], pred_coords_scaled[3], c='r', marker='s', s=100, label='Predicted Bottom')
            
            # Draw lines
            plt.plot([true_coords_scaled[0], true_coords_scaled[2]], 
                    [true_coords_scaled[1], true_coords_scaled[3]], 'g-', label='True Racket')
            plt.plot([pred_coords_scaled[0], pred_coords_scaled[2]], 
                    [pred_coords_scaled[1], pred_coords_scaled[3]], 'r-', label='Predicted Racket')
            
            plt.legend()
            plt.title(f'Image {idx}: {image_path.name}')
            
            # Calculate and display error
            mse = np.mean((np.array(true_coords) - np.array(pred_coords)) ** 2)
            plt.xlabel(f'MSE: {mse:.2f}')
            
            # Save and close
            plt.savefig(output_dir / f'prediction_{idx}.png')
            plt.close()
            
            if idx % 10 == 0:
                print(f'Processed {idx}/{total_images} images')
                
        except Exception as e:
            print(f'Error processing image {idx}: {str(e)}')
            continue
    
    print('Visualization complete!')

if __name__ == "__main__":
    # Train and evaluate model with wavelet features
    
    use_wavelets = False  # Define the variable first
    model, X_test, y_test, X_train, y_train, info_test = train_model(use_wavelets=use_wavelets)
    if use_wavelets:
        print("Training model with wavelet features...")
    else:
        print("Training model with cropped features...")
    
    # Evaluate on training set
    train_evaluation = evaluate_model(model, X_train, y_train, info_test, use_wavelets=use_wavelets)
    print(f"\nTraining Metrics (Wavelet Features):")
    print(f"MSE: {train_evaluation['overall_mse']:.4f}")
    print(f"MAE: {train_evaluation['overall_mae']:.4f}")
    
    # Evaluate on test set
    test_evaluation = evaluate_model(model, X_test, y_test, info_test, use_wavelets=use_wavelets)
    print(f"\nTest Metrics (Wavelet Features):")
    print(f"MSE: {test_evaluation['overall_mse']:.4f}")
    print(f"MAE: {test_evaluation['overall_mae']:.4f}")

    images_dir = '/Users/oscaralberigo/Desktop/CDING/TennisAI/coco-annotator/datasets/tennis_cropped'

    # Visualize predictions
    output_dir = f'linear_regression/visualize_predictions_test'
    visualize_predictions(test_evaluation, images_dir, output_dir=output_dir)

    # Analyze bias and variance
    bias_variance = analyze_bias_variance(model, X_test, y_test, info_test)
    
    print(f"\nBias-Variance Analysis (test set):")
    print(f"Average distance from true points: {bias_variance['avg_distance']:.4f}")
    print(f"Standard deviation of distances: {bias_variance['std_distance']:.4f}")
    print("\nBias for each coordinate:")
    for coord, value in bias_variance['bias'].items():
        print(f"{coord}: {value:.4f}")
    print("\nVariance for each coordinate:")
    for coord, value in bias_variance['variance'].items():
        print(f"{coord}: {value:.4f}")