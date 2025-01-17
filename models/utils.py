import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SplitTransformer(BaseEstimator, TransformerMixin):
    '''
    Split each feature according to a character and expand them
    '''
    def __init__(self, by=","):        
        self.by = by        

    def fit(self, X, y=None):
        self._number_splits = X.apply(lambda x : x.str.count(',').max(),axis=0).astype(int)
        return self

    def transform(self, X, y=None):
        return_df = pd.DataFrame()
        for col, j in zip(X.columns, self._number_splits):
            splitted = X[col].str.split(self.by, n=j, expand=True)
            new_col = [col+"-"+str(i) if i!=0 else col for i in range(j+1)]
            return_df[new_col] = splitted
        return return_df



def split_columns(df : pd.DataFrame, cols : list) -> pd.DataFrame:
    '''
    Split df[cols] columns according to ","
    Return the splitted columns expanded and in the order present in cols
    '''
    return_df = pd.DataFrame()
    nb_splits = df[cols].apply(lambda x : x.str.count(',').max(),axis=0).astype(int)
    for col,j in zip(cols,nb_splits):
        splitted = df[col].str.split(",",n=j, expand=True)
        new_col = [col+"-"+str(i) if i!=0 else col for i in range(j+1)]
        return_df[new_col] = splitted
    return return_df