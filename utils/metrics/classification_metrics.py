
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class ClassificationMetrics:
    def __init__(self):
        pass

     #or sklearn.metrics.accuracy_score
    def accuracy(self, actual: np.ndarray, predicted: np.ndarray):
        """Accuracy = correct predictions / total"""
        actual, predicted = self._check_inputs(actual, predicted)
        return np.mean(actual == predicted)
    


    def cls_report(self, actual: np.ndarray, predicted: np.ndarray):
        return classification_report(actual, predicted,  zero_division=0)

    def confusion_matrix(self, actual: np.ndarray, predicted: np.ndarray):
        return confusion_matrix(actual,predicted)

    def _check_inputs(self, actual: np.ndarray, predicted: np.ndarray):
        """Ensure inputs are valid and same shape"""
        if actual.shape != predicted.shape:
            raise ValueError("Shapes of actual and predicted must match")
        return actual, predicted


    
   
    def precision(self, actual: np.ndarray, predicted: np.ndarray, average="binary", positive_label=1):
        actual, predicted = self._check_inputs(actual, predicted)
        labels = np.unique(actual)

        if average == "binary":
            tp = np.sum((predicted == positive_label) & (actual == positive_label))
            fp = np.sum((predicted == positive_label) & (actual != positive_label))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        elif average == "macro":
            precisions = []
            for label in labels:
                tp = np.sum((predicted == label) & (actual == label))
                fp = np.sum((predicted == label) & (actual != label))
                precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            return np.mean(precisions)

        elif average == "micro":
            tp = np.sum(predicted == actual)
            fp = np.sum(predicted != actual)
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self, actual: np.ndarray, predicted: np.ndarray, average="binary", positive_label=1):
        actual, predicted = self._check_inputs(actual, predicted)
        labels = np.unique(actual)

        if average == "binary":
            tp = np.sum((predicted == positive_label) & (actual == positive_label))
            fn = np.sum((predicted != positive_label) & (actual == positive_label))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        elif average == "macro":
            recalls = []
            for label in labels:
                tp = np.sum((predicted == label) & (actual == label))
                fn = np.sum((predicted != label) & (actual == label))
                recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            return np.mean(recalls)

        elif average == "micro":
            tp = np.sum(predicted == actual)
            fn = np.sum(predicted != actual)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1_score(self, actual: np.ndarray, predicted: np.ndarray, average="binary", positive_label=1):
        if average == "binary":
            p = self.precision(actual, predicted, average="binary", positive_label=positive_label)
            r = self.recall(actual, predicted, average="binary", positive_label=positive_label)
            return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        elif average == "macro":
            labels = np.unique(actual)
            f1s = []
            for label in labels:
                p = self.precision(actual, predicted, average="binary", positive_label=label)
                r = self.recall(actual, predicted, average="binary", positive_label=label)
                f1s.append((2 * p * r) / (p + r) if (p + r) > 0 else 0.0)
            return np.mean(f1s)

        elif average == "micro":
            return self.precision(actual, predicted, average="micro")

    def confusion_matrix(self, actual: np.ndarray, predicted: np.ndarray, labels=None):
        """
        Confusion Matrix
        Rows = actual classes
        Columns = predicted classes
        """
        actual, predicted = self._check_inputs(actual, predicted)

        if labels is None:
            labels = np.unique(np.concatenate([actual, predicted]))

        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        label_to_index = {label: idx for idx, label in enumerate(labels)}

        for a, p in zip(actual, predicted):
            i = label_to_index[a]
            j = label_to_index[p]
            matrix[i, j] += 1

        return matrix, labels

