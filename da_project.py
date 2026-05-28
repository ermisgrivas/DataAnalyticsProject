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
Garage Related (1379)
FireplaceQu (770)
Electrical (1459)
Basement Related (1422-1423)
MasVnrArea (1452)
MasVnrType (588)
Alley (91)
LotFrontage (1201)
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
(4) If MiscVal = 0 then MiscFeature = NA etc etc.
So, missing values are related to things that don't exist (like houses not having a fireplace or 
an alley etc), so they should be left with "None"
"""

# Adding the rest of the missing values (filled with "None")
cols = ["FireplaceQu", "Alley", "PoolQC", "MiscFeature", "Fence", "MasVnrType"]
for col in cols:
    data_copy[col] = data_copy[col].fillna("None")

missing_values(data_copy) # New plot to show the values have been filled


# IQR Outlier Detection for numeric columns
def find_outliers_iqr(data, feats=None, factor=1.5):
    # Select data to use
    if feats is not None:
        data_subset = data[feats]
    else:
        # Auto-select numeric columns
        data_subset = data.select_dtypes(include="number")

    if data_subset.empty:
        raise ValueError("No numeric columns to compute outliers.")

    # Calculate Q1, Q3, and IQR
    q1 = data_subset.quantile(0.25)
    q3 = data_subset.quantile(0.75)
    iqr = q3 - q1

    # Compute bounds
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    # Find rows where any value is an outlier
    outliers_mask = (data_subset < lower_bound) | (data_subset > upper_bound)
    outliers_rows = data_subset[outliers_mask.any(axis=1)].index

    return outliers_rows, outliers_mask

outlier_rows, outlier_mask = find_outliers_iqr(data_copy)

# Looking for strong outlier rows (rows with 5+ outliers)
data_copy["Outlier_Count"] = outlier_mask.sum(axis=1)
strong_outliers = data_copy[data_copy["Outlier_Count"] > 5]

"""
Worth noting at this point that after testing 18 strong outlier rows were detected.
A small enough number to where we will conduct a manual inspection to determine which
should be removed.
"""

# Show outlier columns to help manual inspection
outlier_columns = []
for idx in strong_outliers.index:
    cols = outlier_mask.columns[outlier_mask.loc[idx]]
    outlier_columns.append(", ".join(cols))

strong_outliers["Outlier_Columns"] = outlier_columns

# Exporting strong outlier rows to csv to proceed with manual inspection
strong_outliers.to_csv("outliers.csv", index= False)

# 4 example box plots to demonstrate outliers
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for feat, axis in zip(
    ["SalePrice", "GrLivArea", "MasVnrArea", "GarageArea"],
    axes.flatten(),
    strict=True,
):
    sns.boxplot(data=data_copy, y=feat, ax=axis)
    axis.set_title(f"Boxplot of {feat}")

plt.tight_layout()
plt.show()

"""
After manual inspection we have determined the following rows to be removed:
(1) ID = 1299. Amazing home according to most metrics (10 overall quality, huge living area)
    but sold very cheaply.
(2) ID = 524. For similar reasons
(3) ID = 636. 8 bedroom, 14 room apartment
"""

data_copy = data_copy.drop([1299, 524, 636])

data_copy.corr(method="pearson", numeric_only=True)
# Calculate the correlation matrix
corr = data.corr(method="pearson", numeric_only=True).round(2)

# Create the mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Create the heatmap with the mask
plt.figure(figsize=(20, 16))
sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    vmin=-1,
    vmax=1,  # Ensure that color scaling is consistent
    cbar_kws={"label": "Correlation Coefficient"},
    annot_kws={"size": 10},  # Adjust annotation size
    mask=mask,  # Apply the mask to hide the upper triangle
)

# Title and labels for context
plt.title("Correlation Matrix of Numerical Features", fontsize=16, fontweight="bold")
plt.xlabel("Features", fontsize=12)
plt.ylabel("Features", fontsize=12)

# Rotate the axis labels for better readability
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0, ha="right")

plt.tight_layout()
plt.show()

"""
The following observations are made:
(1) Strong correlation between GarageCars and GarageArea (combine features).
    Same with 1stFlrSF and BsmtSF
(2) 3ssn porch has very low correlation (<+-0.1) to everything (remove feature)
    same with miscval, mosold
(3) Quality and size (living area, garages) most important for price
(4) ID has no reason to exist, we can use the array index instead

Leading us to the following steps:
(1) New features: GarageAreaPerCar, LowerFloorsSF
(2) Removed features: 3ssn porch, miscval, mosold, ID
"""

data_copy["GarageAreaPerCar"] = data_copy["GarageArea"] / data_copy["GarageCars"].replace(0, np.nan)
data_copy["GarageAreaPerCar"] = data_copy["GarageAreaPerCar"].fillna(0)
data_copy = data_copy.drop("ID", axis=1)
data_copy = data_copy.drop("3SsnPorch", axis=1)
data_copy = data_copy.drop("MiscVal", axis=1)
data_copy = data_copy.drop("MoSold", axis=1)
