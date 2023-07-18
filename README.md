# U-Net Semantic Segmentation

This project implements the U-Net model for semantic segmentation on the Cityscapes dataset using PyTorch. Semantic segmentation is the task of assigning a class label to each pixel in an image, enabling pixel-level understanding of the scene.

The U-Net architecture is widely used for semantic segmentation tasks due to its ability to capture both global context and local details. It consists of a contracting path (downsampling) and an expanding path (upsampling) that enables the model to learn hierarchical representations.

## Dataset

The Cityscapes dataset is used for training and evaluation. It provides high-quality pixel-level annotations for urban street scenes. The dataset contains images captured in 50 cities, with diverse environmental and weather conditions.

To use the Cityscapes dataset, you need to download it from the official website (https://www.cityscapes-dataset.com/) and set the appropriate paths in the code.

## Requirements

- Python (>=3.6)
- PyTorch (>=1.9)
- NumPy
- scikit-image
- scikit-learn
- tqdm
- matplotlib


## Usage

1. Clone the repository

2. Set up the dataset:
   - Download the Cityscapes dataset from the official website.
   - Extract the dataset and specify the path in the code (`path_data` variable).

3. Train the model:
   - Adjust the training parameters in the code (e.g., learning rate, number of epochs).
   - Run the training script

4. Evaluate the model:
- After training, the model will be saved.
- Run the evaluation script to calculate the metrics on the validation set


5. Customize the model:
- You can modify the U-Net architecture by adjusting the number of input and output channels, as well as the number of features in each block.
- Experiment with different hyperparameters to improve the performance.

## Results

The model's performance is evaluated using various metrics, including accuracy, Jaccard score (IoU), specificity, sensitivity, and F1 score. The results are logged and displayed during training and evaluation.







