import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def get_skewness(df):
    return df.apply(lambda x: x.skew()).abs()


class DataTransformer:
    def __init__(self, df, threshold=0.5):
        self.raw_df = df
        self.transformed_df = self.raw_df.copy()
        self.scaled_df = self.raw_df.copy()
        self.skewed_cols = None
        self.threshold = threshold

    def transform_data(self):
        skewed_features = get_skewness(self.raw_df)
        self.skewed_cols = skewed_features[skewed_features > self.threshold].index

        for col in self.skewed_cols:
            self.transformed_df[col] = np.log1p(self.raw_df[col])

    def visualize(self, scaled=False):
        num_features = len(self.skewed_cols)
        fig, axs = plt.subplots(num_features, 2 if scaled == False else 3, figsize=(15, 2*num_features))

        if num_features == 1:
            axs = np.expand_dims(axs, axis=0)

        for i, col in enumerate(self.skewed_cols):
            axs[i, 0].hist(self.raw_df[col])
            axs[i, 0].set_title(f"{col} - Original")

            axs[i, 1].hist(self.transformed_df[col])
            axs[i, 1].set_title(f"{col} - Transformed")

            if scaled:
                axs[i, 2].hist(self.scaled_df[col])
                axs[i, 2].set_title(f"{col} - Scaled")

        plt.tight_layout()
        plt.show()

    def scale_features(self):
        scaler = StandardScaler()

        self.scaled_df = pd.DataFrame(
            scaler.fit_transform(self.transformed_df),
            columns=self.transformed_df.columns
        )
