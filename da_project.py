# Necessary libraries for project
import matplotlib.ticker as ticker
import missingno as msno
import numpy as np
import pandas as pd
import contextily as ctx
import seaborn as sns
import sklearn
import geopandas as gpd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
test = pd.read_csv('test.csv')
data = pd.read_csv('train.csv')

# Missing values plot function (to avoid repeat code)
def missing_values(dataframe):

    # Create figure and axis with a specified size
    fig, ax = plt.subplots(1, 1, figsize=(7, 81))

    # Generate the missing data bar plot with custom color
    msno.bar(dataframe, ax=ax, color="#3498db")

    # Enhance plot aesthetics
    ax.set_title("Missing Data Overview", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Columns", fontsize=10)
    ax.set_ylabel("Count of Non-missing Values", fontsize=10)

    # Reduce font size for bar labels (values on top of bars) and side index labels
    ax.tick_params(axis="both", which="major", labelsize=7)  # Shrinks x and y ticks

    # Remove top and right spines for a cleaner look
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    plt.show()

missing_values(data)  # Initial missing values before imputation

"""
The following missing data was noticed:

MiscFeature (54/1460)
Fence (281)
PoolQC (7)
Garage Related (1379)  -> IMPUTED
FireplaceQu (770)
Electrical (1459) -> IMPUTED
Basement Related (1422-1423) -> IMPUTED
MasVnrArea (1452) -> IMPUTED
MasVnrType (588)
Alley (91)
LotFrontage (1201) -> IMPUTED
"""

# Class-based Imputation for columns with a low number of missing values

data_copy = data.copy()
data_copy["class"] = pd.cut(
    data_copy["MSSubClass"],
    bins=[0, 20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 160, 180, 190],
    labels=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8","C9", "C10", "C11", "C12", "C13", "C14", "C15"],
)

numeric_cols = ["MasVnrArea", "LotFrontage"]
string_cols = ["GarageCond", "GarageQual", "GarageFinish", "GarageYrBlt", "GarageType",
        "BsmtFinType1", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtExposure", "Electrical"]

# Median imputation for numeric columns
for col in numeric_cols:
    group_medians = data_copy.groupby("class", observed=True)[col].transform("median")
    data_copy[col] = data_copy[col].fillna(group_medians)

# Mode imputation for string columns
for col in string_cols:
    group_modes = (
        data_copy.groupby("class", observed=True)[col]
        .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    data_copy[col] = data_copy[col].fillna(group_modes)

"""
For the six remaining columns of missing data, these observations were made:
(1) If MasVnrArea = 0 then MasVnrArea = None
(2) If Fireplaces = 0 then FireplaceQu = NA
(3) If PoolArea = 0 then PoolQC = NA
(4) Missing values are related to things that don't exist (like houses not having a fireplace or 
an alley etc), so they should be left with "None"
"""

# Adding the rest of the missing values (filled with "None")
cols = ["FireplaceQu", "Alley", "PoolQC", "MiscFeature", "Fence", "MasVnrType"]
for col in cols:
    data_copy[col] = data_copy[col].fillna("None")

# New plot to show the values have been filled
missing_values(data_copy)