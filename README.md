# Heart Disease Prediction System

## Overview
The Heart Disease Prediction System is a machine learning-based web application designed to predict the likelihood of heart disease in individuals based on various health indicators.

## Features
- **Machine Learning Models**: Implements multiple machine learning algorithms including Logistic Regression, Decision Trees, Random Forests, Support Vector Machines, and K-Nearest Neighbors for prediction.
- **Data Visualization**: Provides insightful visualizations such as correlation heatmaps, bar graphs, and box plots to understand the relationship between various factors and heart disease.
- **User Authentication**: Includes a user-friendly interface with signup, login, and logout functionality to ensure secure access.
- **Backend Integration**: Utilizes Flask for backend operations, managing model predictions and user interactions efficiently.

## Project Structure
- **Frontend**: Contains the HTML, CSS, and JavaScript files for the web pages including the signup, login, home, FAQ, and prediction pages.
- **Backend**: Built with Flask, the backend handles user data, manages sessions, and serves the machine learning model to generate predictions.
- **Model**: The machine learning model is trained on the Framingham Heart Study dataset, which includes various health indicators such as age, cholesterol levels, blood pressure, smoking habits, etc.

## Installation
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/heart-disease-prediction-system.git
    cd heart-disease-prediction-system
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```bash
    python app.py
    ```
   Navigate to `http://127.0.0.1:5000/` in your web browser.

## Usage
1. **Signup/Login**: Users must sign up or log in to access the heart disease prediction features.
2. **Predict Now**: Navigate to the "Predict Now" page, enter your health details, and click "Predict" to see the result.
3. **View Results**: Based on the input data, the system will predict whether the user is at risk of heart disease.

## Data
- **Framingham Heart Study Dataset**: This dataset includes over 4,000 observations with 16 variables related to heart health. It is used to train the machine learning models implemented in this project.

## Machine Learning Models
- **Random Forest Classifier**
- **Support Vector Machine**
- **Decision Tree Classifier**
- **Logistic Regression**
- **K-Nearest Neighbors**

## Results
- The models were evaluated based on accuracy, precision, recall, and F1 score. The Random Forest and K-Nearest Neighbors classifiers provided the best performance with accuracy rates above 83%.

## Future Scope
- **Expand Dataset**: Incorporate additional datasets for more comprehensive predictions.
- **Improve Accuracy**: Explore advanced machine learning techniques and hyperparameter tuning to improve model accuracy.
- **Real-time Data**: Integrate the system with real-time health monitoring devices for continuous risk assessment.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
