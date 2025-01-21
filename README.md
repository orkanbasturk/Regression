# Regression
# Data Analysis and Prediction Application

This application is a **Machine Learning and Data Analytics Tool** designed to process user-selected datasets, perform analysis, build a linear regression model, and generate predictions. It supports various datasets and allows users to dynamically select the target variable for regression analysis.

---

## 🚀 **Overview**

This application provides the following functionalities:
1. **Data Processing and Analysis:**
   - Cleans, scales, and prepares the input dataset.
   - Dynamically handles missing or inconsistent data.
   - Automatically detects the file delimiter (`,` or `;`) for flexibility in handling various CSV formats.

2. **Regression Modeling:**
   - Builds a linear regression model using PyTorch.
   - Trains the model on the selected dataset and visualizes the learning process.

3. **Explanatory Analytics:**
   - Generates a **correlation matrix** with numerical values for easy interpretation.
   - Provides **SHAP analysis** to explain how each feature impacts predictions.

4. **Prediction and Interactivity:**
   - Allows users to input custom feature values to generate predictions.
   - Outputs visualizations such as learning loss, correlation matrix, and SHAP summary.

---

## 🛠️ **Technologies Used**

This project leverages the following tools and technologies:
- **Python**: Primary language for data manipulation, analysis, and machine learning.
- **PyTorch**: Used to build and train the linear regression model.
- **Pandas**: Handles dataset loading, cleaning, and manipulation.
- **NumPy**: Performs efficient numerical computations.
- **Matplotlib and Seaborn**: Generates visualizations for learning loss and correlation matrix.
- **SHAP (SHapley Additive ExPlanations)**: Explains the contributions of features to the model’s predictions.
- **Tkinter**: Provides a graphical file picker for user-friendly dataset selection.
- **Scikit-Learn**: Standardizes the data and evaluates model performance (MSE, R²).

---


---

## 📥 **Installation**

To set up the application locally, follow these steps:

### Prerequisites:
- Python 3.8 or later installed on your system.
- A CSV file with numerical data (ensure the file has a target variable for regression).

### 1. Clone the Repository

git clone https://github.com/your-username/your-repository.git
cd your-repository

### 2. Set Up a Virtual Environment
It’s recommended to use a virtual environment to manage dependencies.

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the Application
python main.py

📝 How to Use

Step 1: Select a Dataset
When prompted, use the graphical file picker to select a CSV file.

Step 2: Choose a Target Variable
Enter the name of the column you want the model to predict (e.g., Revenue).

Step 3: Training and Visualization
The application will clean the data, train the model, and generate the following outputs in the output/ directory:
loss_graph.jpg: Shows the model's learning progress over epochs.
correlation_matrix.jpg: Displays relationships between features and the target variable.
shap_summary.jpg: Visualizes the impact of each feature on the model's predictions.

Step 4: Make Predictions
Enter custom values for the features when prompted to get predictions for the target variable.

📊 Outputs and Visualizations
Correlation Matrix:

Provides a heatmap showing the relationships between features and the target variable.
Learning Loss Graph:

Displays the loss over epochs during model training.
SHAP Analysis:

Explains how each feature contributes to the model's predictions.

⚙️ Customization
You can customize the application for specific use cases:

Modify the learning rate, batch size, or number of epochs in the main.py script.
Extend the application to support other machine learning models (e.g., neural networks, decision trees).

🛡️ Limitations
The application assumes numerical data; categorical data needs to be encoded beforehand.
Works best with datasets where the target variable has a linear relationship with the features.

🤝 Contributing
Contributions are welcome! If you have suggestions for improving the application, feel free to fork the repository, make changes, and submit a pull request.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

👨‍💻 Author
orkanbasturk – https://github.com/orkanbasturk





