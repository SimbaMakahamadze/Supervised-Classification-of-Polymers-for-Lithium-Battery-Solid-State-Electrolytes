
from helper import get_fingerprints_batch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def corr():
    X, y = get_fingerprints_batch(512)
    
    feature_df = pd.read_csv("forest_feature_importance.csv")
    important_feature_indices = feature_df['support']
    data = pd.DataFrame(X)
    y_col = pd.Series(y)
    data = pd.concat([data, y_col], axis=1)
    important_feature_df = data.loc[:, important_feature_indices]
    
    corr_matrix = important_feature_df.corr()
    sn.heatmap(corr_matrix, annot=True)
    plt.savefig('feature_correlation_heatmap')
    