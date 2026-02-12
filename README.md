# California House Price Prediction with XGBoost

This project builds a machine learning model to predict house prices using the **California Housing Dataset** from `scikit-learn`. The goal is to understand feature relationships, evaluate model performance, and visualize predictions using a tree-based model (XGBoost).

---

## Dataset

The dataset contains housing information for districts in California, including:

- `MedInc` – Median income
- `HouseAge` – Median house age
- `AveRooms` – Average number of rooms
- `AveBedrms` – Average number of bedrooms
- `Population` – District population
- `AveOccup` – Average household occupancy
- `Latitude` – Geographic latitude
- `Longitude` – Geographic longitude
- `price` – Median house value (target variable, in hundreds of thousands of dollars)

---

## Project Workflow

### Data Preparation
- Loaded dataset using `sklearn.datasets.fetch_california_housing()`
- Converted features into a Pandas DataFrame
- Added the target variable (`price`)
- Performed train-test split

### Exploratory Data Analysis (EDA)
- Correlation matrix heatmap
- Geographic price visualization
- Feature-to-price scatter plots
- Distribution analysis of housing prices

### Model Training
- Used **XGBRegressor** (XGBoost)
- Trained model on training data
- Generated predictions for both training and test sets

### Model Evaluation
Metrics used:
- **R² (Coefficient of Determination)**
- **Mean Absolute Error (MAE)**

Results:

| Dataset | R² | MAE |
|----------|------|------|
| Training | ~0.94 | ~$19k |
| Test | ~0.83 | ~$31k |

The model explains approximately **83% of the variation in unseen housing prices**, demonstrating strong predictive performance with mild overfitting.

---

## Visualization

- Actual vs Predicted scatter plot
- Geographic price distribution (Longitude vs Latitude)
- Feature importance (optional extension)
- Price distribution histogram

The model performs well overall but shows slightly higher prediction error for very expensive homes, partly due to the dataset’s price cap.

---

## Key Insights

- Median income is the strongest linear predictor of house price.
- Geographic location effects are highly non-linear.
- Tree-based models outperform linear regression due to interaction effects.
- High-price properties exhibit larger prediction variance.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Log transformation of target variable
- Cross-validation
- Comparison with Linear Regression
- Residual analysis
- Feature importance visualization

---

## Conclusion

This project demonstrates how tree-based ensemble methods like **XGBoost** effectively capture non-linear relationships in structured data. The model generalizes well to unseen data and provides interpretable insights into the key drivers of housing prices.


