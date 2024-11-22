# TennisAI
This project aims to predict the top and bottom positions of a tennis racket in images.

## Structure
- `coco-annotator/`: COCO annotation tool.
- `ml-model/`: Machine learning scripts for training and prediction.
- `datasets/`: Annotated images and labels.
- `notebooks/`: Experimentation notebooks.

## Usage
Steps to reproduce the work:
1. Annotate images using `coco-annotator`.
2. Process annotations and save the data to a csv file like `tennis_data_features.csv` and `tennis_data_labels.csv`.
3. Train the model with a chosen model like in `/linear_regression`.
4. The predictions can be visualised in `/visualize_predictions`.
5. Iterate and improve choices of model and parameters.

## Work So Far
Initial experiments have focused on using a simple linear regression model to predict racket positions. The model was trained on annotated images with normalized coordinates for the top and bottom of each tennis racket.

The results from this basic approach were inaccurate, showing significant deviation between predicted and actual racket positions. The visualizations in `/linear_regression/visualize_predictions` demonstrate this gap, with:
- Red dots showing predicted racket endpoints
- Green dots showing actual racket endpoints

This suggests that a linear regression model may be too simplistic for this computer vision task, as it:
1. Cannot capture the complex visual patterns needed to identify racket positions
2. Treats the relationship between pixel values and coordinates as purely linear
3. Does not leverage spatial relationships in the image data

## Support Vector Machine (SVM) Approach
Following the limitations of linear regression, a Support Vector Machine (SVM) model was implemented using Support Vector Regression (SVR). The SVM approach offers several advantages:

1. Non-linear modeling through kernel functions (RBF kernel used)
2. Better handling of high-dimensional feature spaces
3. Improved generalization through margin optimization

The SVM implementation includes:
- Separate SVR models for each coordinate (x1, y1, x2, y2)
- Model evaluation with MSE and MAE metrics
- Bias-variance analysis to understand model behavior
- Visualization of predictions vs ground truth

Results show a slight decrease in MSE but a slightly higher MAE
Overall this model is still finding it hard to predict the postion of the tennis racket across images.
This could be due to the model not being able to capture the complex differences between images (camera angle, lighting, etc.)

## Contributors
Contributed by Oscar Alberigo

## License
MIT License