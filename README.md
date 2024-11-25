# TennisAI
This project aims to predict the top and bottom positions of a tennis racket in images.

## Structure
- `coco-annotator/`: COCO annotation tool.
- `ml-model/`: Machine learning scripts for training and prediction.
- `datasets/`: Annotated images and labels.
- `notebooks/`: Experimentation notebooks.

## Usage
Steps to run the project:
1. Run the functions in `process_data/` to preprocess the data in the format you want.
2. This will create a `features.csv` and `labels.csv` file.
3. Choose a model (linear regression or SVM) and run the corresponding script in `linear_regression/` or `svm/`.
4. This will train your model and print the results.
5. Visualise the results in `visualize_predictions/`.

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

## Cropping Images

I decided to crop the images to the size of the racket to reduce the complexity of the problem.
This is also a form of data augmentation.

The linear regression model trained on the cropped images performed better than the model trained on the full images. But still not very good.

I evaluated the model on the training set and got a MSE of 0.0000 and MAE of 0.0000. This basically means the model predictions were exactly the same as the ground truth. So the model was overfitting.

Lets add some regularization to the model to see if that helps.

## Limitations of simpler models on image data 

## Checking if I need more data

To check if I need more data, I can train the model on subsets of the data 25%, 50%, 75% and 100% of the data and compare the performance. If performance is better with more data, then I know I need more data.

![MSE vs Training Size](create_visualisation_data_MSE.png)

Key observations:
- Test MSE improves with more training data, but improvements diminish
- Very low training MSE suggests overfitting
- Gap between training and test MSE indicates high model variance

## Contributors
Contributed by Oscar Alberigo

## License
MIT License