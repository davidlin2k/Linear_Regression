
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename, names=headers)

df.replace('?', np.nan, inplace=True)

#replace the missing categorical values with the most frequent value
df["num-of-doors"].replace(np.nan, df["num-of-doors"].value_counts().idxmax(), inplace=True)

#replace the missing numerical values with the mean
missing_values= df.isnull()
for clm in missing_values.columns:
    if True in missing_values[clm].values:
        df[clm].replace(np.nan, df[clm].astype("float").mean(), inplace=True)
        
#change the wrong types
for clm in ["price", "horsepower", "peak-rpm"]:
    df[clm] = df[clm].astype("int")
    
for clm in ["stroke", "bore", ]:
    df[clm] = df[clm].astype("float")

#check correlated features
item = []

for clm in df.columns:
    if (df[clm].dtype == "int32" or df[clm].dtype == "int64" or df[clm].dtype == "float64") and clm != "price":
        pearson_coef, p_value = stats.pearsonr(df[clm], df["price"])
        
        #plot the graph for each feature to check if it's correlated with price
        fig1, ax1 = plt.subplots()
        sns.regplot(x=clm, y="price", data=df, ax=ax1)

        if (pearson_coef > 0.6 or pearson_coef < -0.6) and p_value < 0.001:
            item.append(clm)
            
lm = LinearRegression()

lm.fit(df[item], df["price"])

pred = lm.predict(df[item])

print(r2_score(df['price'], pred))

for x in range(0, len(pred)):
    print("Predict: " + str(pred[x]) + "    \tActual: " + str(df["price"][x]))

fig1, ax1 = plt.subplots()
sns.distplot(y_test, hist=False, color="r", ax=ax1)
sns.distplot(pred, hist=False, color="b", ax=ax1)
