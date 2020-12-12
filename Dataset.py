from sklearn.datasets import load_boston
import pandas as pd


class Dataset:
    """Load data from sklearn module or specified file

    Args:
        config (dict): configuration dictionary

    Attributes:
        df (pd.DataFrame) - Dataframe containing loaded data
        config (dict): configuration dictionary
    """
    def __init__(self, config: dict):
        """Initialize Dataset class"""
        self.df = None
        self.config = config

        self.load_dataset()

    def load_dataset(self):
        """Load dataset from sklearn module or file

        Returns:
             None
        """
        if not self.config['from_csv']:
            X, y = load_boston(return_X_y=True)
            feature_names = load_boston().feature_names

            X_df = pd.DataFrame(X, columns=feature_names)
            y_df = pd.DataFrame(y, columns=self.config['target'])

            self.df = pd.concat([X_df, y_df], axis=1)

        else:
            self.df = pd.read_csv(self.config['source_filename'])

    def get_dataset(self, return_X_y=False):
        """Return loaded dataset

        Args:
            return_X_y (bool, optional) : boolean deciding to return feature separately from target or not

        Returns:
             self.df (pd.DataFrame): DataFrame containing loaded dataset according to return_X_y
        """
        if return_X_y:
            return (self.df.drop(columns=self.config['target']),
                    self.df[self.config['target']])
        else:
            return self.df
