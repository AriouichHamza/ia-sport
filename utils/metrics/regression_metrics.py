
import numpy as np
from sklearn.metrics import r2_score
from typing import Any

class RegressionMetrics:
    def __init__(self):
        self.EPSILON = 1e-10

    def _error(self, actual: np.ndarray, predicted: np.ndarray):
        """ Simple error """
        return actual - predicted

    def mae(self, actual: np.ndarray, predicted: np.ndarray):
        """ Mean Absolute Error """
        return np.mean(np.abs(self._error(actual, predicted)))

    def mse(self,actual: np.ndarray, predicted: np.ndarray):
        """ Mean Squared Error """
        return np.mean(np.square(self._error(actual, predicted)))

    def leastSquare(self,actual, predicted):
        errors  = self._error(actual, predicted)
        errrors_squared  = np.square(errors)
        dataset_length = len(actual)
        return errrors_squared / dataset_length

    def wape(self,actual: np.ndarray, predicted: np.ndarray):
        """ Weighted Absolute Percentage Error """
        return self.mae(actual, predicted) / (np.mean(actual) + self.EPSILON )


    #average r2 of all series(seperately)
    def  r2_avr(self,y_true: np.ndarray, y_pred: np.ndarray):  
        #y_true & y_pred : #(num_series,batch,samples)
        num_series = len(y_true)
        avr_r2 = 0
        for i in range(num_series):
            avr_r2 +=r2_score(y_true[i], y_pred[i])
        # print(str(i) + " " + str(avr_r2) )

        return avr_r2/num_series

    def  r2_all(self,y_true: np.ndarray, y_pred: np.ndarray):       
        return  r2_score(y_true, y_pred)
        return avr_r2/num_series
        #return  r2_score(y_true, y_pred)

    def rmse(self,y_true: np.ndarray,y_pred: np.ndarray):
        return np.sqrt(self.mse(y_true,y_pred))
    
    def nRMSE_mean(self,y_true: np.ndarray,y_pred: np.ndarray):
            """
            Compute RMSE and normalized RMSE metrics.
            
            Parameters:
                y_true : array-like of shape (n_samples,)
                    True target values.
                y_pred : array-like of shape (n_samples,)
                    Predicted values.
            Returns:
                nRMSE_mean (%)
            """
            nrmse_mean = (self.rmse(y_true,y_pred) / np.mean(y_true)) * 100            
            return np.round(nrmse_mean,2)
    
    def nRMSE_range(self,y_true: np.ndarray,y_pred: np.ndarray):
            """
            Compute RMSE and normalized RMSE metrics.
            
            Parameters:
                y_true : array-like of shape (n_samples,)
                    True target values.
                y_pred : array-like of shape (n_samples,)
                    Predicted values.
            Returns:
                 nRMSE_range (%)
            """
            nrmse_mean = (self.rmse(y_true,y_pred) / (np.max(y_true) - np.min(y_true))) * 100            
            return np.round(nrmse_mean,2)

    
    def _percentage_error(self,actual: np.ndarray, predicted: np.ndarray):
        """
        Percentage error

        Note: result is NOT multiplied by 100
        """
        return self._error(actual, predicted) / (actual + self.EPSILON)

    def mape(self,actual: np.ndarray, predicted: np.ndarray):
        return np.mean(np.abs(self._percentage_error(actual, predicted)))

    def displayMetrics(self,y_true:np.ndarray,y_pred:np.ndarray)->dict[str,Any]:
        print('mse  : {}'.format(self.mse(y_true, y_pred)))
        print( 'mae  : {}'.format(self.mae(y_true,y_pred)))
        print('rmse  : {}'.format(self.rmse(y_true,y_pred)))
        print('wape  : {}'.format(self.wape(y_true,y_pred)))
        print('mape  : {}'.format(self.mape(y_true,y_pred)))
        # print('avg r2  : {}'.format(self.r2_avr(y_true,y_pred)))
        err_dic= {
            "mse": self.mse(y_true, y_pred),
            "mae":  self.mae(y_true,y_pred),
            "rmse": self.rmse(y_true,y_pred),
            "wape": self.wape(y_true,y_pred),
            "mape": self.mape(y_true,y_pred),
            # "r2": self.r2_avr(y_true,y_pred)
        }
        # print("err_fic", err_dic)
        return err_dic
    

def calculate_normalized_results(self, y_array,pred_array):
    #normalized errors
    norm_err_dic= {
            "mse": self.mse(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) ),
            "mae": self.mae(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) ),
            "rmse": self.rmse(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) ),
            "wape":self. wape(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) ),
            "mape": self.mape(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) ),
            "r2": self.r2_avr(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) )
        }
    return norm_err_dic

class ClassificationMetrics:
    def __init__(self):
        self.EPSILON = 1e-10  # To avoid division by zero

    def accuracy(self, actual: np.ndarray, predicted: np.ndarray):
        """ Classification Accuracy """
        return np.mean(actual == predicted)

    def precision(self, actual: np.ndarray, predicted: np.ndarray, average="binary", positive_label=1):
        """
        Precision metric.
        average: 'binary', 'macro', 'micro'
        """
        if average == "binary":
            tp = np.sum((predicted == positive_label) & (actual == positive_label))
            fp = np.sum((predicted == positive_label) & (actual != positive_label))
            return tp / (tp + fp + self.EPSILON)

        classes = np.unique(actual)
        precisions = []
        for cls in classes:
            tp = np.sum((predicted == cls) & (actual == cls))
            fp = np.sum((predicted == cls) & (actual != cls))
            precisions.append(tp / (tp + fp + self.EPSILON))

        if average == "macro":
            return np.mean(precisions)
        elif average == "micro":
            tp = sum(np.sum((predicted == cls) & (actual == cls)) for cls in classes)
            fp = sum(np.sum((predicted == cls) & (actual != cls)) for cls in classes)
            return tp / (tp + fp + self.EPSILON)
        else:
            raise ValueError("average must be one of ['binary', 'macro', 'micro']")

    def recall(self, actual: np.ndarray, predicted: np.ndarray, average="binary", positive_label=1):
        """
        Recall metric.
        average: 'binary', 'macro', 'micro'
        """
        if average == "binary":
            tp = np.sum((predicted == positive_label) & (actual == positive_label))
            fn = np.sum((predicted != positive_label) & (actual == positive_label))
            return tp / (tp + fn + self.EPSILON)

        classes = np.unique(actual)
        recalls = []
        for cls in classes:
            tp = np.sum((predicted == cls) & (actual == cls))
            fn = np.sum((predicted != cls) & (actual == cls))
            recalls.append(tp / (tp + fn + self.EPSILON))

        if average == "macro":
            return np.mean(recalls)
        elif average == "micro":
            tp = sum(np.sum((predicted == cls) & (actual == cls)) for cls in classes)
            fn = sum(np.sum((predicted != cls) & (actual == cls)) for cls in classes)
            return tp / (tp + fn + self.EPSILON)
        else:
            raise ValueError("average must be one of ['binary', 'macro', 'micro']")

    def f1_score(self, actual: np.ndarray, predicted: np.ndarray, average="binary", positive_label=1):
        """
        F1 Score metric.
        average: 'binary', 'macro', 'micro'
        """
        if average == "binary":
            precision = self.precision(actual, predicted, average="binary", positive_label=positive_label)
            recall = self.recall(actual, predicted, average="binary", positive_label=positive_label)
            return 2 * (precision * recall) / (precision + recall + self.EPSILON)

        classes = np.unique(actual)
        f1_scores = []
        for cls in classes:
            p = self.precision(actual, predicted, average="binary", positive_label=cls)
            r = self.recall(actual, predicted, average="binary", positive_label=cls)
            f1_scores.append(2 * (p * r) / (p + r + self.EPSILON))

        if average == "macro":
            return np.mean(f1_scores)
        elif average == "micro":
            precision_micro = self.precision(actual, predicted, average="micro")
            recall_micro = self.recall(actual, predicted, average="micro")
            return 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + self.EPSILON)
        else:
            raise ValueError("average must be one of ['binary', 'macro', 'micro']")

    def confusion_matrix(self, actual: np.ndarray, predicted: np.ndarray, labels=None):
        """
        Confusion Matrix
        Rows = Actual classes
        Columns = Predicted classes
        """
        if labels is None:
            labels = np.unique(np.concatenate([actual, predicted]))

        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_index = {label: idx for idx, label in enumerate(labels)}

        for a, p in zip(actual, predicted):
            matrix[label_to_index[a], label_to_index[p]] += 1

        return matrix, labels

