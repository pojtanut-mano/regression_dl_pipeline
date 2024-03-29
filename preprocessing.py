import pandas as pd
import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

ECHELON = 1e-2


class PreProcessing:
    """Preprocess data according to configuration file

    Apply data preprocessing methods as listed below:
        - fillNaN with the given value
        - (Optional) normalize numerical value features
        - (Optional) apply PCA transformation to reduce the dimension of features
        - (Optional) apply BoxCox transformation to transform data into normal distribution-ish

    Args:
        X (pd.DataFrame): dataframe containing features
        y (pd.DataFrame): vector containing target, in our case, it needs to be real positive values
        config (dict): configuration dictionary
        dir_name (str): location that lambda used in BoxCox transformation will be stored

    Attributes:
        X (pd.DataFrame): dataframe containing features
        y (pd.DataFrame): vector containing target, in our case, it needs to be real positive values
        config (dict): configuration dictionary
        dir_name (str): location that lambda used in BoxCox transformation will be stored
        X_dummy (pd.DataFrame): features dataframe after transforming categorical features into dummies
        X_train (pd.DataFrame): dataframe containing features used in training step
        y_train (pd.DataFrame): dataframe containing target used in training step
        X_test (pd.DataFrame): dataframe containing features used in test step
        y_test (pd.DataFrame): dataframe containing target used in test step
    """
    def __init__(self, X, y, config, dir_name):
        self.config = config
        self.X = X
        self.y = y
        self.dir_name = dir_name

        self.X_dummy = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.object_cols = list(self.X.columns[self.X.dtypes == 'object'])
        self.numeric_cols = self.find_complement(self.X.columns, self.object_cols)

        # Transform X
        if self.config['fillNaN']:
            self.X = self.fill_nan(self.X)

        if self.config['normalization']:
            self.X = self.normalize(self.X)

        # if self.config['remove_corr_cols']:
        #     self.X = self.remove_corr_cols(self.X, self.config['corr_threshold'])

        if self.config['PCA']:
            self.X = self.PCA_transformation(self.X, self.config['PCA_threshold'])

        # Transform y
        if self.config['BoxCox']:
            self.y = self.BoxCox_transformation(self.y, self.dir_name)

        # Get dummy variables
        self.X_dummy = self.get_dummies(self.X)

        # Scale y
        self.y_scaled, self.y_scaler = self.transform_y(self.y)

        # Split train test
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(
            self.X_dummy, self.y)
        print('Shape of train and test')
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)

    def find_complement(self, main_iter, sub_iter):
        """Return the elements in the list that aren't in the another list

        Args:
            main_iter (iterable): iterable containing elements that will be retained
            sub_iter (iterable): iterable containing elements that won't be included in result list

        Returns:
            (list): containing elements that aren't in the sub_list
        """
        return [i for i in main_iter if i not in sub_iter]

    def fill_nan(self, df):
        """Fill object cols with most occurrences

        Args:
            df (pd.DataFrame): dataframe

        Returns:
            df (pd.DataFrame): fill dataframe with median or mean according to skewness of distribution
        """
        for col in self.object_cols:
            temp_occur = df[col].value_counts().keys()[0]
            df[col] = df[col].fillna(temp_occur)

        # Fill numeric cols
        fill_mean_cols = df[self.numeric_cols].columns[(df[self.numeric_cols].skew().abs() < ECHELON).values]
        fill_median_cols = df[self.numeric_cols].columns[(df[self.numeric_cols].skew().abs() >= ECHELON).values]

        fill_mean_values = df[fill_mean_cols].mean()
        fill_median_values = df[fill_median_cols].median()

        df[fill_mean_cols] = df[fill_mean_cols].fillna(fill_mean_values)
        df[fill_median_cols] = df[fill_median_cols].fillna(fill_median_values)

        return df

    def normalize(self, df):
        """Transform each feature into distribution with mean 0 and variance 1

        Args:
            df (pd.DataFrame): DataFrame containing features

        Returns:
            df (pd.DataFrame): transformed DataFrame
        """
        scaler = MinMaxScaler()
        df[self.numeric_cols] = scaler.fit_transform(df[self.numeric_cols])

        return df

    def PCA_transformation(self, X, threshold):
        """Select principal components with explained variance above the threshold

        Args:
            X (pd.DataFrame): DataFrame containing features
            threshold (float): value between 0 and 1

        Returns:
            X (pd.DataFrame): DataFrame that will be transformed
        """
        for i in range(X.shape[1]):
            pca = PCA(n_components=i+1)
            principal_comp = pca.fit_transform(X)
            sum_explained_var = np.sum(pca.explained_variance_ratio_)

            if sum_explained_var >= threshold:
                return pd.DataFrame(principal_comp, columns=["principal component " + str(j) for j in range(1, i+2)])

    def BoxCox_transformation(self, y, dir_name):
        """Transform the target by BoxCox transformation which will transform the target into near normal distribution

        Args:
            y (pd.DataFrame): target

        Returns:
            y (np.ndarray): transformed (normalized) target
        """
        boxcox_transformer = PowerTransformer(method='box-cox')
        bc_y = boxcox_transformer.fit_transform(y)

        pickle.dump(boxcox_transformer, open(os.path.join(dir_name, 'box-cox.pkl'), 'wb'))
        return pd.DataFrame(bc_y, columns=['target'])

    def get_dummies(self, df):
        """Transform columns that are categorical variable into dummies variable

        Args:
            df: (pd.DataFrame) containing features

        Returns:
            df: (pd.DataFrame) containing features after transforming some of them into dummies
        """
        if not self.object_cols:
            print("There are no variables that are categorical variables\n"
                  "Returning the original DataFrame...\n")
            return df

        dummy_df = pd.get_dummies(df, columns=self.object_cols)
        return dummy_df

    def transform_y(self, df):
        """Transform targets by using Standardscaler

        Args:
            df (pd.DataFrame): contain targets
        Returns:
            df_scaled (pd.DataFrame): contain transformed targets
        """
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        return df_scaled, scaler

    def split_train_test(self, X, y):
        """Split dataset into train and test set

        Args:
            X (pd.DataFrame): containing features
            y (pd.DataFrame, np.ndarray): containing target
        Returns:
            X_train (pd.DataFrame): train predictor variables
            X_test (pd.DataFrame): test predictor variables
            y_train (pd.DataFrame): train target
            y_test (pd.DataFrame): test target
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], random_state=self.config['seed'],
            shuffle=self.config['shuffle']
        )
        return X_train, X_test, y_train, y_test

    def get_dataset(self):
        """Return transformed dataset

        Returns:
            X (pd.DataFrame): transformed X
            y (pd.DataFrame): transformed y
        """
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_target_transformer(self):
        """Return parameter used to transform targets

        Returns:
            scaler: parameter for transformer
        """
        return self.y_scaler