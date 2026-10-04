# dispatch/intel — Delivery Analytics

A Streamlit data analytics and visualization mini-project for a **food-delivery operations** problem.

## Dataset

This project uses a **fully synthetic, reproducible dataset** generated inside `app.py`.

- 12,000 order records
- 16 variables
- Target: `delivery_time_min`
- Domain: food delivery / logistics
- Seed: `2407`

Each row represents a delivery order. The generator models realistic relationships between distance, restaurant preparation time, traffic, weather, rider experience, peak hours and delivery time.

No external dataset download is required.

## Cleaning

The raw synthetic dataset intentionally contains missing values in:

- `traffic_index`
- `restaurant_rating`
- `prep_time_min`

Missing numerical values are filled with the **median** of their column. Complete duplicate rows are checked and removed if present.

For machine learning:

- numerical features → median imputation + standardization
- categorical features → most-frequent imputation + one-hot encoding
- 80% chronological training split / 20% test split

## Models

Two regression models are compared:

1. Ridge Regression
2. Gradient Boosting Regressor

The app evaluates both using **MAE, RMSE and R²** and selects the strongest model based primarily on R² while checking the error metrics.

## App Flow

**Briefing → The Dataset → Cleaning Desk → Patterns → Model Trials → Estimator**

The interface intentionally uses a different visual direction from the other DAV project: editorial layout, warm paper background, black rules, red/blue accents, compact typography and horizontal navigation.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
