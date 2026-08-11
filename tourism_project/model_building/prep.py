import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv", index_col=0)
df.drop(columns=["CustomerID"], inplace=True)

# replacing 'Fe Male' with 'Female' in Gender columns to avoid inconsisitency
df.replace({'Gender': {'Fe Male': 'Female'}}, inplace=True)

# replacing 'Unmarried' with 'Single' in MaritalStatus as Unamrried and Single are synonymns and we want only one value to avoid any inconsistency
df.replace({'MaritalStatus': {'Unmarried': 'Single'}}, inplace=True)

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("TypeofContact values kept as:", sorted(X["TypeofContact"].unique()))
print("CityTier values kept as:", sorted(X["CityTier"].unique()))
print("Occupation values kept as:", sorted(X["Occupation"].unique()))
print("Gender values kept as:", sorted(X["Gender"].unique()))
print("ProductPitched values kept as:", sorted(X["ProductPitched"].unique()))
print("PreferredPropertyStar values kept as:", sorted(X["PreferredPropertyStar"].unique()))
print("MaritalStatus values kept as:", sorted(X["MaritalStatus"].unique()))
print("Passport values kept as:", sorted(X["Passport"].unique()))
print("OwnCar values kept as:", sorted(X["OwnCar"].unique()))
print("Designation values kept as:", sorted(X["Designation"].unique()))
