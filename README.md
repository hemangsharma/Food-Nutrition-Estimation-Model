# Food-Nutrition-Estimation-Model

## Overview
This project trains a machine learning model to estimate calories and macronutrients (protein, fat, carbohydrates) from food images. The model is trained using a combination of the **Nutrition5k**, **Food Nutrients Dataset**, and **Meal Calorie Estimator** datasets to improve accuracy and generalization.

## Dataset

1. <a href="https://github.com/google-research-datasets/Nutrition5k?utm_source=chatgpt.com">Github</a><br>
2. <a href="https://huggingface.co/datasets/mmathys/food-nutrients?utm_source=chatgpt.com">Huggingface</a><br>
3. <a href="https://universe.roboflow.com/food-nuzjo/meal-calorie-estimator?utm_source=chatgpt.com">universe.roboflow.com</a><br>

## Project Phases
1. **Data Acquisition & Preprocessing**  
   - Load datasets.
   - Preprocess images (resize, normalize).
   - Align nutritional labels.
2. **Model Development**  
   - Train a CNN-based model to extract features from food images.
   - Predict calorie and macronutrient content.
3. **Training & Evaluation**  
   - Train using a GPU-enabled setup.
   - Evaluate performance using MAE (Mean Absolute Error) and fine-tune with transfer learning.
4. **Deployment**  
   - Implement a Jupyter Notebook interface for user input.
   - (Optional) Deploy as an API for further scalability.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow/Keras
- OpenCV
- Pandas, NumPy, SciKit-Learn

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/hemangsharma/Food-Nutrition-Estimation-Model.git
   cd Food-Nutrition-Estimation-Model
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download datasets and place them in the `data/` directory.

## Usage
Run the Jupyter Notebook:
```sh
jupyter notebook food_nutrition_ml.ipynb
```
Upload a food image, and the model will predict its calorie and macronutrient content.

## Next Steps
- Improve accuracy with more diverse datasets.
- Optimize model performance.
- Deploy as a web application or mobile app.

