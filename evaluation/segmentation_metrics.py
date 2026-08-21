import numpy as np

"""PA、MCA、MIoU、FWIU和混淆矩阵"""
class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """返回分割精度评价结果。

            - 总体像素精度
            - 平均类别精度
            - 平均交并比
            - 频率加权交并比
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        class_totals = hist.sum(axis=1)
        acc_cls = np.divide(
            np.diag(hist),
            class_totals,
            out=np.full(self.n_classes, np.nan, dtype=np.float64),
            where=class_totals > 0,
        )
        mean_acc_cls = np.nanmean(acc_cls)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iu = np.divide(
            np.diag(hist),
            union,
            out=np.full(self.n_classes, np.nan, dtype=np.float64),
            where=union > 0,
        )
        mean_iu = np.nanmean(iu)
        # 各类别像素数占全部像素数的比例
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Pixel Acc: ': acc,
                'Class Accuracy: ': acc_cls,
                'Mean Class Acc: ': mean_acc_cls,
                'Freq Weighted IoU: ': fwavacc,
                'Mean IoU: ': mean_iu,
                'confusion_matrix': self.confusion_matrix}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
