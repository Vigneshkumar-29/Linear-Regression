# Linear Regression — Ice Cream Sales 🍦

A small machine-learning project that predicts ice cream sales from temperature. It demonstrates **why simple linear regression fails on non-linear data** and how **polynomial regression** fixes it, using scikit-learn.

## Project Structure

```
.
├── linear_regression.ipynb     # Main analysis notebook
├── Dataset/
│   └── Ice_cream selling data.csv
└── README.md
```

## Dataset

`Dataset/Ice_cream selling data.csv` contains 49 rows with two columns:

| Column | Type | Description |
|---|---|---|
| `Temperature (°C)` | float64 | Daily temperature (feature) |
| `Ice Cream Sales (units)` | float64 | Units sold (target) |

The relationship between temperature and sales is **non-linear** (curved, roughly quadratic), which is the point of the exercise.

## Requirements

- Python 3.8+
- pandas
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install pandas matplotlib scikit-learn
```

## How to Run

1. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

2. Open `linear_regression.ipynb` and run all cells (**Kernel → Restart & Run All**).

The notebook loads the CSV via a relative path (`Dataset/Ice_cream selling data.csv`), so run it from the project root (the directory containing `Dataset/`).

## What the Notebook Does

1. **Load & inspect** — `df.head()`, `df.info()`, and a correlation matrix.
2. **Visualize** — scatter plot of temperature vs. sales; the curved shape shows a straight line won't fit well.
3. **Linear regression baseline** — fit `LinearRegression` on an 80/20 train/test split and evaluate with R².
4. **Polynomial regression** — expand features with `PolynomialFeatures(degree=2)` (using `fit_transform` on train, `transform` on test to avoid leakage), refit, and re-evaluate.

### Results (from the executed notebook)

| Model | Train R² | Test R² |
|---|---|---|
| Linear regression | 0.066 | −0.575 |
| Polynomial (degree 2) | 0.941 | 0.843 |

Plain linear regression explains almost none of the variance (and generalizes worse than predicting the mean), while the degree-2 polynomial fits the data well — a clear illustration of matching model complexity to the data's shape.

## Key Takeaways

- Always plot your data before choosing a model.
- A low/negative R² from a linear fit on curved data is a *model mismatch*, not a data problem.
- Fit the feature transformer on training data only, then transform the test set.
