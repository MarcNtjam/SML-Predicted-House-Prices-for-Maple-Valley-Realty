# 🏡 House Price Prediction — Maple Valley Realty

Supervised machine learning (regression) to estimate residential property prices from features like size, bedrooms, location, and nearby-school ratings — turning historical sales data into fair, data-driven price estimates. 📈

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-modelling-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-viz-11557C?logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-viz-4C72B0)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)

---

## 📌 Overview

**Maple Valley Realty** is a regional real estate agency specialising in residential sales and rentals across the Pacific Northwest. To support its shift toward a data-driven approach, this project builds a **linear regression model** that predicts a property's selling price from its physical attributes and local market signals.

Accurate price estimates let the agency:

- **Advise sellers** on competitive, optimal pricing strategies
- **Give buyers** fair market-value estimates they can trust
- **Help investors** spot undervalued properties

## 🎯 Business Problem

Real estate prices depend on many interacting factors — location, size, amenities, and market trends. Mispricing leads to lost sales and dissatisfied clients. By leveraging historical sales data, this project delivers a reliable, interpretable model that answers a simple question: **what should this property sell for, and which features drive that price?**

## 🗂️ Dataset

The dataset (`house_price_prediction_dataset.csv`) contains **1,500 property records** with **12 columns**. `SellingPrice` is the target variable.

| Feature | Description |
|---|---|
| `PropertyID` | Unique identifier for each property *(dropped during preprocessing)* |
| `Location` | Neighbourhood or area (Downtown, Suburbs, Beachfront, …) |
| `PropertyType` | Type of property (Single Family, Condo, Townhouse) |
| `SizeInSqFt` | Total area of the property in square feet |
| `Bedrooms` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |
| `YearBuilt` | Year the property was constructed |
| `GarageSpaces` | Number of garage spaces available |
| `LotSize` | Lot size in square feet |
| `NearbySchools` | Average rating of nearby schools (1–10 scale) |
| `MarketTrend` | Market direction for the area (1 = increasing, 0 = stable, −1 = decreasing) |
| `SellingPrice` | **Actual selling price of the property (target)** |

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data wrangling | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | scikit-learn (`LinearRegression`, `train_test_split`, preprocessing, metrics) |
| Environment | Jupyter Notebook |

## 🔄 Project Workflow

1. **Data Preparation** — dropped the non-predictive `PropertyID`; imputed missing values in `SizeInSqFt`, `Bedrooms`, `Bathrooms`, and `LotSize` (~5% each) with the **median** (robust to outliers).
2. **Encoding & Scaling** — label-encoded the categorical `Location` and `PropertyType`; applied **Min-Max scaling** to numerical features (chosen after histograms showed a broadly uniform distribution).
3. **Train/Test Split** — 80% training / 20% testing (`random_state=42`).
4. **Model Development** — trained a **Linear Regression** model on the training set.
5. **Evaluation** — assessed performance with MAE, RMSE, and R², and ran residual analysis.
6. **Business Insights** — interpreted model coefficients to rank the key drivers of price.

## 📊 Model Performance

| Metric | Score | Interpretation |
|---|---|---|
| **MAE** | **$67,620.67** | On average, predictions are off by ~$67.6K |
| **RMSE** | **$90,705.03** | Larger errors penalised more heavily — room to improve on outliers |
| **R²** | **0.867** | The model explains ~**87%** of the variance in selling prices |

Residuals were **approximately normally distributed and centred on zero**, indicating the model has no systematic bias (it doesn't consistently over- or under-predict).

## 💡 Key Findings & Business Insights

Feature coefficients (sorted by influence on predicted price):

| Feature | Coefficient | Direction |
|---|---:|:---:|
| `SizeInSqFt` | +548,093 | 🔼 |
| `Bedrooms` | +164,040 | 🔼 |
| `Bathrooms` | +65,965 | 🔼 |
| `NearbySchools` | +49,469 | 🔼 |
| `GarageSpaces` | +40,574 | 🔼 |
| `PropertyType` | +21,643 | 🔼 |
| `YearBuilt` | +21,140 | 🔼 |
| `MarketTrend` | +11,910 | 🔼 |
| `Location` | −14,602 | 🔽 |
| `LotSize` | −401,929 | 🔽 |

*(Numerical features were Min-Max scaled, so a coefficient reflects the price change across that feature's full range — read the table as relative importance and direction rather than raw dollar-per-unit.)*

**Takeaways:**

- 🏠 **Size is king.** `SizeInSqFt` is by far the strongest driver of selling price — bigger homes command substantially higher prices.
- 🛏️ **Rooms add real value.** More bedrooms and bathrooms both push prices up meaningfully.
- 🎓 **Schools matter.** Higher `NearbySchools` ratings lift prices, confirming the premium buyers place on family-friendly neighbourhoods.
- 📈 **Market & recency help.** Positive `MarketTrend` and newer `YearBuilt` nudge prices upward, though with smaller effects.
- ⚠️ **Watch the counter-intuitive signs.** `LotSize` and `Location` carry negative coefficients — likely reflecting non-linear effects, area-specific demand, or the label-encoding of location. These are good candidates for deeper feature engineering.

## 📁 Repository Structure

```
├── Project-SML1-Regression.ipynb            # Main analysis notebook
├── house_price_prediction_dataset.csv       # Dataset (1,500 property records)
├── Case Study - Regression.pdf              # Project brief
└── README.md
```

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/MapleValley-HousePrice-Regression.git
cd MapleValley-HousePrice-Regression

# 2. (Optional) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 4. Launch the notebook
jupyter notebook Project-SML1-Regression.ipynb
```

> **Note:** update the dataset path in the notebook to point at your local copy of `house_price_prediction_dataset.csv` (e.g. place it in a `data/` folder and load with a relative path).

## 🔮 Future Improvements

- Try regularised and non-linear models (Ridge, Lasso, Random Forest, Gradient Boosting) and compare against the linear baseline.
- One-hot encode `Location` / `PropertyType` instead of label encoding to avoid implying a false ordering.
- Engineer new features (e.g. property age from `YearBuilt`, price-per-sqft) and add cross-validation for more robust performance estimates.
- Incorporate external signals such as interest rates or neighbourhood demographics to close the remaining ~13% of unexplained variance.

## 👤 Author

**Marc Aurel Ntjam Minkeng** — *Data Science*
> Update with your GitHub profile and contact links.

---

*Supervised Learning · Regression · Real Estate — a 10Alytics case study.*

