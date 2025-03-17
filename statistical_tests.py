import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


def chi_squared_test(df, col1, col2):
    contingiency_table = pd.crosstab(df[col1], df[col2])
    # Perform the chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingiency_table)
    v = cramers_v(contingiency_table)
    return chi2, p, v

def cramers_v(contingiency_table):
    chi2 = chi2_contingency(contingiency_table)[0]
    n = contingiency_table.sum().sum()
    k = min(contingiency_table.shape) - 1
    return np.sqrt(chi2 / (n * k))
