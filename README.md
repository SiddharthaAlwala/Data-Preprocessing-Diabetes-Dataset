# Data Preprocessing: Diabetes Dataset

This project involves the preprocessing of a diabetes dataset to prepare it for machine learning modeling. The dataset contains features such as `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `Age`, and the target variable `Outcome`. The preprocessing steps include handling outliers, feature scaling (normalization and standardization), and understanding the relationships between features using correlation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Steps in Preprocessing](#steps-in-preprocessing)
- [Key Learnings](#key-learnings)
- [Code Examples](#code-examples)
- [Usage](#usage)

---

## Project Overview
This project aims to:
1. Identify and handle outliers in the dataset using boxplots and the IQR method.
2. Explore and visualize feature correlations.
3. Normalize and standardize features to prepare for machine learning algorithms.
4. Understand the impact of outliers on scaling techniques and select appropriate preprocessing methods.

---

## Dataset Description
The dataset includes the following features:
- **Pregnancies**: Number of pregnancies.
- **Glucose**: Plasma glucose concentration.
- **BloodPressure**: Diastolic blood pressure.
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-hour serum insulin (mu U/ml).
- **BMI**: Body mass index.
- **Age**: Age in years.
- **Outcome**: Target variable (1 = Diabetes, 0 = Not Diabetes).

---

## Steps in Preprocessing
### 1. **Outlier Detection and Handling**
- **Boxplots**: Visualized using Matplotlib and Seaborn to identify potential outliers.
- **IQR Method**: Applied to detect outliers beyond 1.5 times the interquartile range (IQR). Rows with outliers were either removed or capped as necessary.

### 2. **Feature Correlation**
- A correlation matrix was computed to identify relationships between features.
- Visualized using a Seaborn heatmap to interpret positive and negative correlations.
- Key Insights:
  - `Glucose` has the highest correlation with `Outcome` (0.47).
  - Moderate correlations exist between features like `Insulin` and `SkinThickness` (0.44).

### 3. **Normalization vs. Standardization**
- **Normalization**: Scaled features to a range [0, 1] using Min-Max scaling. Suitable for models like neural networks or k-NN.
  
- **Standardization**: Scaled features to have a mean of 0 and standard deviation of 1, making it robust to outliers.
  

### 4. **Proportionality Check**
- A pie chart visualized the distribution of the target variable (`Outcome`) to identify class imbalance.
  - Example: `65% Not Diabetes` vs. `35% Diabetes`.

---

## Key Learnings
- **Outlier Handling**: Identified outliers in features like `Glucose`, `BMI`, and `Insulin` using the IQR method.
- **Scaling Techniques**:
  - Normalization is sensitive to outliers as it depends on the range of the data.
  - Standardization is more robust to outliers as it uses the standard deviation.
- **Correlation Analysis**: Helped select features for modeling by understanding their relationships with the target variable (`Outcome`).
- **Visualization**: Boxplots and heatmaps provided quick insights into the data distribution and relationships.

---

## Code Examples
### Detect and Remove Outliers
```python
# Function to remove outliers using IQR
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # First Quartile
    Q3 = df[column].quantile(0.75)  # Third Quartile
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply to dataset
for column in df.columns[:-1]:
    df = remove_outliers_iqr(df, column)
```

### Min-Max Normalization
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)
```

### Standardization
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df)
```

### Correlation Heatmap
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix
correlation_matrix = df.corr()

# Heatmap visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/preprocessing-diabetes-dataset.git
   ```
2. Navigate to the project directory:
   ```bash
   cd preprocessing-diabetes-dataset
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the preprocessing script to clean and scale the data:
   ```bash
   python preprocess.py
   ```

---

## License
This project is licensed under the MIT License. Feel free to use and modify the code.

