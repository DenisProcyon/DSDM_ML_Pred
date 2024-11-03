import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import zscore

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor

def get_data(type_data: str) -> pd.DataFrame:
    file_path = Path().parent / f'data/{type_data}.csv'

    return pd.read_csv(file_path)

def to_numerical(column: pd.Series) -> pd.Series:
    unique_values = column.dropna().unique()  
    mapping = {value: idx for idx, value in enumerate(unique_values, 1)}

    return column.map(mapping)

def split_doors(data: pd.DataFrame) -> pd.DataFrame:
    data["door_a"], data["door_b"] = zip(*data["door"].apply(lambda x: x.split(" - ") if isinstance(x, str) else (float("nan"), float("nan"))))

    data["door_a"] = to_numerical(data["door_a"])
    data["door_b"] = to_numerical(data["door_b"])

    data = data.drop(columns=["door"])

    return data

def fill_nan_values(data: pd.DataFrame) -> pd.DataFrame:
    interpolated_data = data.fillna(data.median())

    interpolated_data.isna().sum()

    return interpolated_data

def clean_data_from_anomalies(data: pd.Series) -> pd.DataFrame:
    data = data.apply(lambda x: -x if x < 0 else x)
    
    return data

def delete_outliers(data: pd.Series) -> pd.DataFrame:
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers_condition = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
    data_with_nan = data.mask(outliers_condition)
    return data_with_nan

def create_relation(data: pd.DataFrame, columns: list[str]):
    data[f'{"_".join(columns)}_relation'] = data[columns[0]] / data[columns[1]]

    return data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(columns=["id"])

    categorical_columns = [
        "has_pool",
        "orientation",
        "is_furnished",
        "accepts_pets",
        "has_ac",
        "neighborhood"
    ]

    outliers_columns = ['num_rooms', 'num_baths', 'square_meters', 'year_built', 'num_crimes', 'num_supermarkets', 'price']

    for column in categorical_columns:
        data[column] = to_numerical(column=data[column])

    data = split_doors(data)

    for column in outliers_columns:
        data[column] = clean_data_from_anomalies(data[column])
        data[column] = delete_outliers(data[column])

    data = fill_nan_values(data)

    # data = create_relation(data=data, columns=["num_rooms", "num_baths"])
    # data = create_relation(data=data, columns=["square_meters", "num_rooms"])
 
    data["num_supermarkets"] = data["num_supermarkets"].astype(float)
    
    print(data.isna().sum())

    data = data.drop(columns=["accepts_pets", "has_ac", "orientation"])

    return data

data = get_data(type_data="train")

train, test = train_test_split(data, test_size=0.3)
train = preprocess_data(data=train)

X_train = train.drop(columns=["price"])
y_train = train["price"]

X_test = preprocess_data(test)
X_test = X_test.drop(columns=["price"])
y_test = test["price"]

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f"Linear: MSE: {mse_linear}")
print(f"Coef: {linear_model.coef_}")
print(f"Intercept: {linear_model.intercept_}\n")

alphas = [0.1, 0.5, 1.0, 5.0, 10.0]

for alpha in alphas:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    
    y_pred_lasso = lasso_model.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    
    print(f"Lasso  (Alpha: {alpha}): MSE: {mse_lasso}")
    print(f"Coef: {lasso_model.coef_}")
    print(f"Intercept: {lasso_model.intercept_}\n")


