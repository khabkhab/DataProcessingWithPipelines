import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import geopandas


#drops the first column that has no name and delete rows where "NaN" is present
class DropNames(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.drop(["Unnamed: 0", "Country Code", "Series Name", "Series Code"], axis=1)
        X.dropna(how='any')
        return X
#group the dataframe by Country name and sums/mean = sum()/mean() the dept
class GroupBy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.groupby('Country Name').sum()
        return X
    
#combining pipeline 
pipe = Pipeline([
    ("dropper", DropNames()),
    ("groupper", GroupBy()) 
])   
if __name__ == "__main__":
    data = pd.read_csv("ids_last_5y.csv")
    df = pd.DataFrame(data)
    #converts ouput from pipeline back to dataframe format
    result = pd.DataFrame(pipe.fit_transform(df))
    print(result)

