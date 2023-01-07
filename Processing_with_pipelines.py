import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import geopandas


#drops the first column that has no name
class DropNames(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(["Unnamed: 0", "Country Code", "Series Name", "Series Code"], axis=1)

class GroupBy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.groupby('Country Name').mean()
        return X
    

data = pd.read_csv("ids_last_5y.csv")
df = pd.DataFrame(data)

pipe = Pipeline([
    ("dropper", DropNames()),
    ("groupper", GroupBy()) 
])
result = pd.DataFrame(pipe.fit_transform(df))
print(result)

