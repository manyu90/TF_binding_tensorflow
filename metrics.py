from __future__ import absolute_import, division, print_function
import numpy as np
from collections import OrderedDict
from sklearn.metrics import auc, log_loss, precision_recall_curve, roc_auc_score





def positive_accuracy(labels, predictions, threshold=0.5):
    return 100 * (predictions[labels] > threshold).mean()


def negative_accuracy(labels, predictions, threshold=0.5):
    return 100 * (predictions[~labels] < threshold).mean()


def balanced_accuracy(labels, predictions, threshold=0.5):
    return (positive_accuracy(labels, predictions, threshold) +
            negative_accuracy(labels, predictions, threshold)) / 2


def auROC(labels, predictions):
    return roc_auc_score(labels, predictions)


def auPRC(labels, predictions):
    precision, recall = precision_recall_curve(labels, predictions)[:2]
    return auc(recall, precision)


def recall_at_precision_threshold(labels, predictions, precision_threshold):
    precision, recall = precision_recall_curve(labels, predictions)[:2]
    return 100 * recall[np.searchsorted(precision - precision_threshold, 0)]


class ClassificationResult(object):

    def __init__(self, labels, predictions, task_names=None):
        
        self.results = OrderedDict((
            
            ('Balanced accuracy', balanced_accuracy(
                labels,predictions)),
            ('auROC', auROC(labels,predictions)),
            ('auPRC', auPRC(labels,predictions)),
            ('Recall at 5% FDR', recall_at_precision_threshold(
                labels,predictions, 0.95)),
            ('Recall at 10% FDR', recall_at_precision_threshold(
                labels,predictions, 0.9)),
            ('Recall at 20% FDR', recall_at_precision_threshold(
                labels,predictions, 0.8)),
            ('Num Positives', labels.sum()),
            ('Num Negatives', (1 - labels).sum())))
        
    def __str__(self):
        return ('Balanced Accuracy: {:.2f}%\t '
            'auROC: {:.3f}\t auPRC: {:.3f}\n\t'
            'Recall at 5%|10%|20% FDR: {:.1f}%|{:.1f}%|{:.1f}%\t '
            'Num Positives: {}\t Num Negatives: {}'.format(*self.results.values()))
            

    