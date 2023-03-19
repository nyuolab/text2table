import torch

# This is a metric version of corrloss
def corr_score(input, target):

    # convert input and target to torch tensors
    input = input.detach().clone()
    target = torch.tensor(target)

    # transpose input and target to fit corrcoef function
    input_ = torch.transpose(input, 0, 1)
    target_ = torch.transpose(target, 0, 1)

    # add error terms to input and target for numerical stability
    input_error = torch.rand(input_.shape, device=input_.device)
    target_error = torch.rand(target_.shape, device=target_.device)

    input_error = input_error * 2
    target_error = target_error * 2

    input_error = input_error - 1
    target_error = target_error - 1

    input_error = input_error * 1e-6
    target_error = target_error * 1e-6

    input_ = input_ + input_error
    target_ = target_ + target_error

    # compute correlation using corrcoef
    input_corr = torch.corrcoef(input_)
    target_corr = torch.corrcoef(target_)

    # add 1 to to make correlation values positive for computational reasons
    input_corr = input_corr + 1
    target_corr = target_corr + 1

    # take the absolute value of the correlation matrix
    input_corr = torch.abs(input_corr)
    target_corr = torch.abs(target_corr)

    # take the difference of the correlation matrix
    corr_diff = torch.abs(target_corr - input_corr)

    # apply a boolean matrix mask of the same size with values of 1 only above the diagonal
    corr_diff = torch.triu(corr_diff, diagonal=1)

    # sum up the values of the correlation difference matrix
    corr_diff_sum = torch.sum(corr_diff)

    # compute the denominator that scales the result by the number of correlations
    num_labels = input.shape[1]
    num_corr = (num_labels * (num_labels - 1)) / 2

    # compute the final correlation loss
    corr_s = corr_diff_sum / num_corr
    corr_s = corr_s / 2

    return corr_s.item()
