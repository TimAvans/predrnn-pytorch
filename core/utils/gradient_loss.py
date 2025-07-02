import torch

def gradient_loss(pred, gt):
    # pred, gt: [B, C, H, W]
    dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy_gt = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
    dx_gt = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
    
    grad_diff_y = torch.abs(dy_pred - dy_gt)
    grad_diff_x = torch.abs(dx_pred - dx_gt)
    return torch.mean(grad_diff_x) + torch.mean(grad_diff_y)
