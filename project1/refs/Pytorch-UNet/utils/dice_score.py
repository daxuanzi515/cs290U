# import torch
# from torch import Tensor


# def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
#     # Average of Dice coefficient for all batches, or for a single mask
#     assert input.size() == target.size()
#     assert input.dim() == 3 or not reduce_batch_first

#     sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

#     inter = 2 * (input * target).sum(dim=sum_dim)
#     sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
#     sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

#     dice = (inter + epsilon) / (sets_sum + epsilon)
#     return dice.mean()


# def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
#     # Average of Dice coefficient for all classes
#     return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


# def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
#     # Dice loss (objective to minimize) between 0 and 1
#     fn = multiclass_dice_coeff if multiclass else dice_coeff
#     return 1 - fn(input, target, reduce_batch_first=True)

import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, eps: float = 1e-6) -> Tensor:
    """
    Compute Dice coefficient (between 0 and 1).
    """
    assert input.size() == target.size(), f"input {input.shape} & target {target.shape} must have same shape"
    sum_dims = (-1, -2) if input.dim() == 4 else tuple(range(1, input.dim()))

    inter = torch.sum(input * target, dim=sum_dims)
    union = torch.sum(input + target, dim=sum_dims)

    dice = (2. * inter + eps) / (union + eps)
    if reduce_batch_first:
        dice = dice.mean()  # mean over batch

    return dice


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, eps: float = 1e-6) -> Tensor:
    """
    Dice coefficient for multiple classes.
    """
    # compute per-channel dice and average
    dices = []
    for channel in range(input.shape[1]):
        dices.append(dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, eps))
    return torch.mean(torch.stack(dices))


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False, eps: float = 1e-6) -> Tensor:
    """
    Dice loss (objective to minimize): 1 - Dice coefficient
    """
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True, eps=eps)
