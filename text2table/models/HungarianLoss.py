import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from scipy.optimize import linear_sum_assignment

# This file intends to be used as a loss function for the hierarchical LED model. The loss function is based on the implementation of
# the Cross Entropy Loss on pytorch. By computing cross entropy loss between all predictions and targets, the hungarian algorithm is 
# used to find the optimal assignment between the predictions and targets that minimizes the cross entropy loss. The optimal assignment is 
# then used to compute the total loss, and this total loss is returned as the training loss for current step.


class HungarianLoss(CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(HungarianLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target):
        """
        Args:
            input (Tensor): A 2D tensor with shape (batch_size, num_classes) where the num_classes is the number of classes
                in the dataset.
            target (Tensor): A 1D tensor with shape (num_classes) where the num_classes is the number of classes in the dataset.
        """
        # input and target are both 3D tensors with shape (batch_size, num_classes, num_classes)
        # input is the output of the model, and target is the ground truth.
        # The hungarian algorithm is used to find the optimal assignment between the predictions and targets that minimizes the cross entropy loss.
        # The computed total loss by the optimal assignment is returned as the training loss for current step.
        # The hungarian algorithm is implemented in the scipy.optimize.linear_sum_assignment function.
        # The hungarian algorithm is a greedy algorithm, and it is not guaranteed to find the optimal assignment.
        # The hungarian algorithm is also known as the Kuhn-Munkres algorithm.

        
        