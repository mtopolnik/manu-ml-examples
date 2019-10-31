import datetime
import sklearn

def auc(true_labels, predicted_probabilities, positive_label):
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_labels, predicted_probabilities, pos_label = positive_label)
	auc = sklearn.metrics.auc(fpr, tpr)
	return auc

def get_timestamp():
	return datetime.datetime.now().strftime('%Y%m%d_%H%M')

