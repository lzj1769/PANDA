import torch
import torch.nn as nn


def cohen_kappa_loss(y_true, y_pred, row_label_vec, col_label_vec,
                     weight_mat, eps=1e-6, dtype=torch.float)
    labels = torch.matmul(y_true, col_label_vec)
    weight = tf.pow(tf.tile(labels, [1, tf.shape(y_true)[1]]) - tf.tile(row_label_vec, [tf.shape(y_true)[0], 1]), 2)
    weight /= tf.cast(tf.pow(tf.shape(y_true)[1] - 1, 2), dtype=dtype)
    numerator = tf.reduce_sum(weight * y_pred)

    denominator = tf.reduce_sum(
        tf.matmul(
            tf.reduce_sum(y_true, axis=0, keepdims=True),
            tf.matmul(weight_mat, tf.transpose(tf.reduce_sum(y_pred, axis=0, keepdims=True)))
        )
    )

    denominator /= tf.cast(tf.shape(y_true)[0], dtype=dtype)

    return tf.math.log(numerator / denominator + eps)

class CohenKappaLoss(nn.Module):
    def __init__(self, reduction='mean', num_classes=None):
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, input, target):
