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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from kneed import KneeLocator
from sklearn.svm import SVC

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

data = data.drop([1298, 523, 635]) # Since index = 0 for ID = 1
data = data.drop(["Outlier_Count"], axis=1) # No longer needed

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
        "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType",
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

classification_data = data.copy() # Before HouseStyle gets one-hot encoded

# Future target variable so will be one-hot encoded separately
one_hot = pd.get_dummies(data["HouseStyle"], dtype=int, drop_first=True, prefix="HouseStyle")
data = data.drop("HouseStyle", axis=1)
data = data.join(one_hot)


"""
End of encoding
"""

data.to_csv("data.csv", index = False) # See final results of dataset after preprocessing

"""
End of preprocessing. Start of clustering
"""

# Choosing natural & structural features for clustering & classification
cols = ["OverallQual", "YearBuilt", "TotalBsmtSF", "GrLivArea", "TotRmsAbvGrd", "GarageArea",
        "GarageCars", "KitchenQual", "GarageQual", "LotArea",
        "Fireplaces", "FullBath", "PoolArea", "BsmtExposure"]

X_cluster = data[cols]

# Scaling
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Find optimal k by silhouette score method
silhouette_scores = []
K = range(2, 20)

for k in K:
    # Initialise kmeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster_scaled)

    # Silhouette score
    silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))

# Find the optimal K based on the maximum silhouette score
optimal_k = K[np.argmax(silhouette_scores)]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(K, silhouette_scores, "bx-", label="Silhouette Score", markersize=8)
plt.axvline(
    x=optimal_k,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Optimal K = {optimal_k}",
)
plt.scatter(
    optimal_k,
    silhouette_scores[np.argmax(silhouette_scores)],
    color="red",
    s=100,
    zorder=5,
)

plt.title("Silhouette Analysis For Optimal K", fontsize=16, fontweight="bold")
plt.xlabel("Number of Clusters (K)", fontsize=12)
plt.ylabel("Silhouette Score", fontsize=12)
plt.xticks(K)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="best")
plt.tight_layout()
plt.show()

"""
The silhouette score plot shows that 3 clusters is the optimal route. The score afterwards rapidly 
descends as k increases.
"""

pca2d = PCA(n_components=2)
X_cluster_scaled_2d = pca2d.fit_transform(X_cluster_scaled)

# Running K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X=X_cluster_scaled)

_, ax = plt.subplots(1, 1, figsize=(6, 6))

sns.scatterplot(
    x=X_cluster_scaled_2d[:, 0],
    y=X_cluster_scaled_2d[:, 1],
    hue=kmeans.predict(X_cluster_scaled),
    palette="Set1",
    ax=ax,
    s=40,
    edgecolor="k",
)

ax.set_title("Cluster Assignments", fontsize=14, fontweight="bold")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
sns.despine(ax=ax)
plt.legend(title="Cluster", loc="upper right", bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show()

"""
Upon observation of the plot, K-Means separates the data into 3 clusters, 
based on the quality and size of each house. The outcome is obviously that 
bigger & higher quality homes end up more expensive.
"""

# Clustering via DBSCAN


# Fit NearestNeighbors model and calculate distances
nearest_neighbors = NearestNeighbors(n_neighbors=11)
neighbors = nearest_neighbors.fit(X_cluster_scaled)

distances, indices = neighbors.kneighbors(X_cluster_scaled)
distances = np.sort(distances[:, 10], axis=0)

# Find the knee point (approximate where the slope increases sharply)
knee_point = np.argmax(np.diff(distances)) + 1  # Add 1 because diff reduces length by 1

# Plot
fig = plt.figure(figsize=(7, 5))
plt.plot(distances, color="b", label="Distance to 11th nearest neighbor", linewidth=2)
plt.scatter(
    knee_point,
    distances[knee_point],
    color="red",
    s=80,
    label=f"Knee point = {knee_point}",
    zorder=5,
)
plt.axvline(knee_point, color="red", linestyle="--", linewidth=1.5)

plt.title("k-NN Distance Plot (for DBSCAN)", fontsize=14, fontweight="bold")
plt.xlabel("Sample Index (sorted)", fontsize=12)
plt.ylabel("Distance to 11th Nearest Neighbor", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Locate eps
i = np.arange(len(distances))
knee = KneeLocator(
    i,
    distances,
    S=1,
    curve="convex",
    direction="increasing",
    interp_method="polynomial",
)

knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")
plt.axhline(y=distances[knee.knee], color="red", linestyle="--", linewidth=1.5)
plt.show()

# Apply DBSCAN
dbscan = DBSCAN(eps=distances[knee.knee], n_jobs=-1)
dbscan.fit(X_cluster_scaled)

fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout="constrained")

sns.scatterplot(
    x=X_cluster_scaled_2d[:, 0],
    y=X_cluster_scaled_2d[:, 1],
    hue=dbscan.labels_,
    palette="Set1",
    ax=ax,
    legend="full",
    s=50,
)
ax.set_title("DBSCAN Clustering", fontsize=14, fontweight="bold")
sns.despine()
plt.show()

"""
Observation: DBScan also produced 3 clusters. 
Blue (cluster 0) is the dominant cluster. The other 2 clusters (1,2) indicate small subgroups while
-1 indicates noise/outliers.
"""

# Identify clusters
fig, ax = plt.subplots(4, 5, figsize=(10, 10), sharex=True, sharey=True)
ax_flat = ax.flatten()

for i, cluster_id in enumerate(set(dbscan.labels_)):
    mask = dbscan.labels_ == cluster_id
    ax_flat[i].scatter(
        X_cluster_scaled_2d[mask, 0],
        X_cluster_scaled_2d[mask, 1],
        color=plt.cm.tab20(cluster_id) if cluster_id != -1 else "white",
    )
    ax_flat[i].set_title(f"Cluster ID: {cluster_id}")

plt.show()


"""
End of clustering. Beginning of classification
"""

# Inspection of the target variable's distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=classification_data,
    x="HouseStyle",
    order=classification_data["HouseStyle"].value_counts().index
)

plt.title("Distribution of HouseStyle")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Training a Bayesian classifier based on the earlier-selected features
X = classification_data[cols]
y = classification_data["HouseStyle"] # Target variable

# Test-train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict
y_pred_nb = nb.predict(X_test)

# Evaluate
print("NB Accuracy:", accuracy_score(y_test, y_pred_nb))

print(classification_report(y_test, y_pred_nb))

"""
NB is not a strong classifier by any metric, as we're seeing low accuracy and F1-scores across the board.
It is surprisingly good at predicting the least dominant classes of the dataset, but the model does a 
poor job at reflecting 1Story and 2Story's level of dominance.
"""

# Training an SVM classifier (scaled data is required here)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    random_state=42
)

svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

"""
SVM accuracy is dramatically improved (nearly 30% compared to NB), as are the F1-scores.
"""

# Confusion matrices
cm_nb = confusion_matrix(y_test, y_pred_nb, labels=nb.classes_)
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=svm.classes_)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(
    cm_nb,
    annot=True,
    fmt="d",
    ax=ax[0],
    xticklabels=nb.classes_,
    yticklabels=nb.classes_
)

ax[0].set_title("Naive Bayes")

sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    ax=ax[1],
    xticklabels=svm.classes_,
    yticklabels=svm.classes_
)

ax[1].set_title("SVM")

plt.tight_layout()
plt.show()

"""
From the Confusion Matrix comparison, it's evident that SVM much more accurately predicts the 
two dominant classes (1Story and 2Story). Since those account for nearly 80% of the test dataset,
it leads to an overall higher accuracy and makes it the stronger model overall.
"""
