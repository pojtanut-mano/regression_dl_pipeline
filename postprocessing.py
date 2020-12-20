import os
import pandas as pd


class Postprocessing:
    """Postprocess training and test result from model

    Args:
        model: trained Pytorch model

    Attributes:
         _model: trained Pytorch model
    """

    def __init__(self, model):
        self._model = model

    def predict(self, data):
        """predict the result according to trained model

        Args:
            data (torch.tensor): data to be run through model to get result
        Returns:
            output (torch.tensor): prediction result
        """
        output = self._model(data)
        return output

    def save_csv(self, X_train, X_test, dir_name):
        """Save prediction from model in csv format

        Args:
            X_train (torch.tensor): training set features
            X_test (torch.tensor): test set features
            dir_name (str): directory location

        Returns:
            None
        """
        np_training_pred = self.predict(X_train).detach().cpu().numpy().ravel()
        np_test_pred = self.predict(X_test).detach().cpu().numpy().ravel()

        training_pred = pd.DataFrame({'Train_result': np_training_pred}, index=range(len(np_training_pred)))
        test_pred = pd.DataFrame({'Test result': np_test_pred}, index=range(len(np_test_pred)))

        training_pred.to_csv(os.path.join(os.getcwd(), dir_name, 'result_training.csv'), index=False)
        test_pred.to_csv(os.path.join(os.getcwd(), dir_name, 'result_test.csv'), index=False)
