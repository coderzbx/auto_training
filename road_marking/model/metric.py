"""
Evaluation metrics.
"""


import logging
import numpy as np
import mxnet as mx


class SegMetric(mx.metric.EvalMetric):
    """Segmentation metric, including pixel-acc, mean-acc, mean-iou
    """

    def __init__(self, output_names=None, label_names=None,
            use_ignore=True, ignore_label=255):
        """Initializer for segmentation metric
        """
        self._nclass = 0
        super(SegMetric, self).__init__(
                'SegMetric', output_names, label_names)
        self._names = ['pixel-acc', 'accs', 'ious', 'mean-acc', 'mean-iou']
        self._use_ignore = use_ignore
        self._ignore_label = ignore_label

    def set_nclass(self, number_of_classes):
        self._nclass = number_of_classes
        self._tp = np.zeros(self._nclass)
        self._fp = np.zeros(self._nclass)
        self._fn = np.zeros(self._nclass)
        self._num_inst = np.zeros(self._nclass)

    def reset(self):
        """Reset metrics.
        """
        self._tp = np.zeros(self._nclass)
        self._fp = np.zeros(self._nclass)
        self._fn = np.zeros(self._nclass)
        self._num_inst = np.zeros(self._nclass)

    def update(self, labels, preds):
        """Update metrics.
        """
        for pred, label in zip(preds, labels):
            label = label.asnumpy().ravel()
            pred = pred.asnumpy().ravel()
            # TODO: whether to drop ignore label
            if self._use_ignore:
                mask = label != self._ignore_label
                label = label[mask]
                pred = pred[mask]
            for i in xrange(self._nclass):
                self._tp[i] += ((label == i) & (pred == i)).sum()
                self._fp[i] += ((label != i) & (pred == i)).sum()
                self._fn[i] += ((label == i) & (pred != i)).sum()
                self._num_inst[i] += (label == i).sum()

    def get(self):
        """Get current state of metrics.
        """
        pixel_acc = self._tp.sum() / self._num_inst.sum()
        accs = np.divide(self._tp, self._num_inst,
                out=np.full_like(self._tp, np.nan), where=self._tp != 0)
        logging.info('accs:\n{}'.format(accs))
        ious = np.divide(self._tp, self._tp + self._fp + self._fn,
                out=np.full_like(self._tp, np.nan), where=self._tp != 0)
        logging.info('ious:\n{}'.format(ious))
        values = [pixel_acc, accs, ious,
                  accs[np.logical_not(np.isnan(accs))].mean(),
                  ious[np.logical_not(np.isnan(ious))].mean()]
        return (self._names, values)

