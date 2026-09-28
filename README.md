# SIT720 — Sydney Housing Price Prediction and Decision Support System
##1. Project Overview
This project develops a machine learning-based regression system for predicting residential house sale prices across three Sydney suburbs: Chatswood, Parramatta, and Blacktown.

The project was completed as part of SIT720 and covers data collection, exploratory data analysis, feature engineering, regression modelling, model evaluation, prediction-error investigation, comparison with human and LLM estimates, and deployment of the selected model as a Streamlit web application.

## 2. Project Resources
**GitHub Repository:**  
https://github.com/YoggJoshi7/SIT720-Sydney-Housing-Price-Prediction

**Deployed Streamlit Application:**  
https://sit720-sydney-housing-price-prediction.streamlit.app/

## 3. Dataset

The dataset contains 100 manually collected sold-property records from three Sydney suburbs:

- Chatswood — 31 properties
- Parramatta — 34 properties
- Blacktown — 35 properties

The target variable is **Sale Price**.

The dataset includes property characteristics such as:

- Suburb
- Sale Date
- Sale Price
- Property Type
- Bedrooms
- Bathrooms
- Parking
- Land Size
- Sale Method
- Sale Year

The data was collected from publicly available property listing sources. Missing information was retained where it could not be reliably obtained rather than being artificially populated.

## 4. Machine Learning Models

Three regression approaches were evaluated:

1. Linear Regression
2. Random Forest Regression
3. Ridge Regression

The models were evaluated using k-fold cross-validation and regression metrics including RMSE, MAE, and R².

Linear Regression was selected as the final model based on its cross-validation performance.

## 5. Repository Structure

| File | Description |
|---|---|
| `SIT720_8.1D_COMPLETE_Sydney_Housing_Project.ipynb` | Complete and corrected project notebook containing data analysis, modelling, evaluation and Part 5 holdout experiment |
| `SIT720_8.1D_Sydney_Housing_Data_Cleaned.csv` | Cleaned housing dataset used for modelling |
| `SIT720_8.1D_app.py` | Streamlit application source code |
| `sydney_house_price_model.joblib` | Saved trained Linear Regression pipeline used by the deployed application |
| `requirements.txt` | Python dependencies required to run the application |
| `README.md` | Project documentation |

## 6. How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YoggJoshi7/SIT720-Sydney-Housing-Price-Prediction.git
cd SIT720-Sydney-Housing-Price-Prediction
pip install -r requirements.txt
streamlit run SIT720_8.1D_app.py

---

### 7. Application

```markdown
## Web Application

The final trained model was deployed using Streamlit Community Cloud.

The application allows users to enter:

- Suburb
- Sale method
- Number of bedrooms
- Number of bathrooms
- Parking spaces
- Land size
- Sale year

The application then returns an estimated property sale price.

**Live Application:**  
https://sit720-sydney-housing-price-prediction.streamlit.app/

## 8. Model Performance

| Model | CV RMSE (AUD) | CV R² | Test RMSE (AUD) | Test R² |
|---|---:|---:|---:|---:|
| Linear Regression | 498,207 | 0.764 | 290,871 | 0.912 |
| Random Forest | 530,751 | 0.750 | 346,220 | 0.875 |
| Ridge Regression | 571,510 | 0.713 | 446,918 | 0.792 |

Linear Regression was selected based on the lowest mean cross-validation RMSE.

## 9. ML, LLM and Human Comparison

Ten properties were held out separately from the Part 3 test set and used to compare:

- The selected machine learning model
- A large language model estimate
- A human estimate

| Approach | MAE (AUD) | RMSE (AUD) |
|---|---:|---:|
| Machine Learning | 357,028 | 488,041 |
| LLM | 320,850 | 459,718 |
| Human | 415,850 | 535,631 |

These results are based on a small sample of ten properties and therefore should not be interpreted as evidence of general superiority outside this experiment.

## Limitations

The dataset is limited to 100 properties across three Sydney suburbs and primarily represents houses. Several potentially influential variables, including building size, property condition, year built, renovations, exact location, views and detailed property characteristics, were unavailable or incomplete.

The dataset also contains an uneven distribution of sale dates, with most observations concentrated in 2025 and 2026. Consequently, the model should not be interpreted as a general-purpose Sydney property valuation system.

Predictions should be treated as estimates rather than professional property valuations.

## Reproducibility

The repository contains the cleaned dataset, complete Jupyter notebook, trained model artifact, application source code and dependency requirements.

The corrected notebook contains the final Part 5 methodology, where the ten held-out properties are kept separate from the Part 3 test set.



