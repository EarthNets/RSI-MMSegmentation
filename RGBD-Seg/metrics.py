import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class M_IOU(object):
    def __init__(self, n_classes, ignore_index=255):
        self.n_classes = n_classes
        self.hist = torch.zeros(n_classes, n_classes).to(device).detach()
        self.ignore_index = ignore_index

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        logits = logits.clone().detach()
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # avoid void
        keep = targets != self.ignore_index

        self.hist += torch.bincount(targets[keep] * self.n_classes + preds[keep], minlength=self.n_classes ** 2).view(self.n_classes, self.n_classes).float()

    def compute(self):
        ious = self.hist.diag() / (1e-9 + self.hist.sum(dim=0) + self.hist.sum(dim=1) - self.hist.diag())
        miou = ious.mean()
        return miou.item()


class ConfusionMatrix(object):
    """Calculates confusion matrix for multi-class data.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must contain logits and has the following shape (batch_size, num_categories, ...)
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
        with or without the background class. During the computation, argmax of `y_pred` is taken to determine
        predicted classes.

    Args:
        num_classes (int): number of classes. See notes for more details.
        average (str, optional): confusion matrix values averaging schema: None, "samples", "recall", "precision".
            Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
            samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
            represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
            diagonal values represent class precisions.

    Note:
        In case of the targets `y` in `(batch_size, ...)` format, target indices between 0 and `num_classes` only
        contribute to the confusion matrix and others are neglected. For example, if `num_classes=20` and target index
        equal 255 is encountered, then it is filtered out.

    """

    def __init__(self, num_classes, average=None):
        if average is not None and average not in ("samples", "recall", "precision"):
            raise ValueError("Argument average can None or one of ['samples', 'recall', 'precision']")

        self.num_classes = num_classes
        self._num_examples = 0
        self.average = average
        self.confusion_matrix = None

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes,
                                            dtype=torch.int64, device='cpu')
        self._num_examples = 0

    def _check_shape(self, output):
        y_pred, y = output

        if y_pred.ndimension() < 2:
            raise ValueError("y_pred must have shape (batch_size, num_categories, ...), "
                             "but given {}".format(y_pred.shape))

        if y_pred.shape[1] != self.num_classes:
            raise ValueError("y_pred does not have correct number of categories: {} vs {}"
                             .format(y_pred.shape[1], self.num_classes))

        if not (y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y_pred must have shape (batch_size, num_categories, ...) and y must have "
                             "shape of (batch_size, ...), "
                             "but given {} vs {}.".format(y.shape, y_pred.shape))

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

    def update(self, output):
        self._check_shape(output)
        y_pred, y = output

        self._num_examples += y_pred.shape[0]

        # target is (batch_size, ...)
        y_pred = torch.argmax(y_pred, dim=1).flatten()
        y = y.flatten()

        target_mask = (y >= 0) & (y < self.num_classes)
        y = y[target_mask]
        y_pred = y_pred[target_mask]

        indices = self.num_classes * y + y_pred
        m = torch.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

    def _get_double_cm(self):
        cm = self.confusion_matrix.type(torch.DoubleTensor)
        if self.average:
            if self.average == "samples":
                return cm / self._num_examples
            elif self.average == "recall":
                return cm / (cm.sum(dim=1) + 1e-15)
            elif self.average == "precision":
                return cm / (cm.sum(dim=0) + 1e-15)

        return cm

    def iou(self, ignore_index=None):
        cm = self._get_double_cm()

        iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)

        if ignore_index is not None:
            indices = list(range(len(iou)))
            indices.remove(ignore_index)
            return iou[indices]

        return iou

    def miou(self, ignore_index=None):
        return self.iou(ignore_index=ignore_index).mean()

    def accuracy(self):
        cm = self._get_double_cm()
        return cm.diag().sum() / (cm.sum() + 1e-15)
