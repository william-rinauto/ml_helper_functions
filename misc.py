import numpy as np
import pandas as pd

# function that takes X dataframe as input and y series and calcs cor of all x with y 
def correlation(X, y):
    corrs = []
    for col in X.columns:
        cor = np.corrcoef(X[col], y)[0, 1]
        corrs.append((col, cor))
    corrs.sort(key=lambda x: np.abs(x[1]), reverse=True)

    sorted = pd.Series(dict(corrs)).sort_values(ascending=False)

    return sorted


