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
train = pd.read_csv('train.csv')

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

missing_values(train)  # Initial missing values before imputation

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

data = train.copy()
data["Class"] = pd.cut(
    data["MSSubClass"],
    bins=[0, 20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 160, 180, 190],
    labels=[1, 2, 3, 4, 5, 6, 7, 8,9, 10, 11, 12, 13, 14, 15],
)

data = data.drop("MSSubClass", axis=1) # Since classes have been binned

# Median imputation for numeric columns
cols = ["MasVnrArea", "LotFrontage"]
for col in cols:
    group_medians = data.groupby("Class", observed=True)[col].transform("median")
    data[col] = data[col].fillna(group_medians)

# Mode imputation for string columns
cols = ["GarageCond", "GarageQual", "GarageFinish", "GarageYrBlt", "GarageType",
        "BsmtFinType1", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtExposure", "Electrical"]
for col in cols:
    group_modes = (
        data.groupby("Class", observed=True)[col]
        .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    data[col] = data[col].fillna(group_modes)

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
    data[col] = data[col].fillna("None")

missing_values(data) # New plot to show the values have been filled

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

outlier_rows, outlier_mask = find_outliers_iqr(data)

# Looking for strong outlier rows (rows with 5+ outliers)
data["Outlier_Count"] = outlier_mask.sum(axis=1)
strong_outliers = data[data["Outlier_Count"] > 5]

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

"""
Exporting strong outlier rows to csv to proceed with manual inspection (will be commented as
outliers.csv isn't part of the project and was only used for initial help

strong_outliers.to_csv("outliers.csv", index= False)
"""

# 4 example box plots to demonstrate outliers
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for feat, axis in zip(
    ["SalePrice", "GrLivArea", "MasVnrArea", "GarageArea"],
    axes.flatten(),
    strict=True,
):
    sns.boxplot(data=data, y=feat, ax=axis)
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

data = data.drop([1299, 524, 636])
data = data.drop(["Outlier_Count"], axis=1) # No longer needed

data.corr(method="pearson", numeric_only=True)
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

data["GarageAreaPerCar"] = data["GarageArea"] / data["GarageCars"].replace(0, np.nan)
data["GarageAreaPerCar"] = data["GarageAreaPerCar"].fillna(0)
data = data.drop("Id", axis=1)
data = data.drop("3SsnPorch", axis=1)
data = data.drop("MiscVal", axis=1)
data = data.drop("MiscFeature", axis=1) # Since MiscFeature was imputed based on MiscVal earlier
data = data.drop("MoSold", axis=1)
data = data.drop("Utilities", axis=1) # Upon observing it has the same result 1459/1460 times

"""
Start of encoding
"""

# All Quality/Condition related features will be ordinally encoded.
# All other features will be one-hot encoded

# Ordering the categories so that "None" gets dropped
data["MasVnrType"] = pd.Categorical(
    data["MasVnrType"],
    categories=["None", "BrkFace", "Stone", "BrkCmn"]
)

data["Alley"] = pd.Categorical(
    data["Alley"],
    categories=["None", "Grvl", "Pave"]
)

data["Fence"] = pd.Categorical(
    data["Fence"],
    categories=["None", "GdPrv", "GdWo", "MnPrv", "MnWw"]
)

# One-hot encoding
cols = ["Street", "CentralAir", "Alley", "MSZoning", "LotShape", "LandContour", "LotConfig",
        "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
        "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation",
        "Heating", "Electrical", "Functional", "GarageType", "Fence", "SaleType", "SaleCondition"]
for col in cols:
    one_hot = pd.get_dummies(data[col], dtype=int, drop_first=True, prefix=col)
    data = data.drop(col, axis=1)
    data = data.join(one_hot)

# Ordinal encoding for quality/condition based features
ordinal_quality_map = {
    "None": 0,
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5
}

bsmt_fin_map = {
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6
}

paved_drive_map = {
    "N": 1,
    "P": 2,
    "Y": 3
}

garage_finish_map = {
    "Unf": 1,
    "RFn": 2,
    "Fin": 3
}

bsmt_exposure_map = {
    "No": 1,
    "Mn": 2,
    "Av": 3,
    "Gd": 4
}

cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",
        "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
for col in cols:
    data[col] = data[col].map(ordinal_quality_map).astype("Int64")

cols = ["BsmtFinType1", "BsmtFinType2"]
for col in cols:
    data[col] = data[col].map(bsmt_fin_map).astype("Int64")

data["BsmtExposure"] = data["BsmtExposure"].map(bsmt_exposure_map).astype("Int64")
data["GarageFinish"] = data["GarageFinish"].map(garage_finish_map).astype("Int64")
data["PavedDrive"] = data["PavedDrive"].map(paved_drive_map).astype("Int64")

"""
End of encoding
"""

data.to_csv("data.csv", index = False) # See final results of dataset after preprocessing

"""
End of preprocessing
"""
