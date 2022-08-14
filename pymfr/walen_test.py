import torch


def _calculate_alfvenicity(frame, windows):
    remaining_flow = (windows[:, :, 3:6] - frame.unsqueeze(1)).squeeze(1)
    alfven_velocity = windows[:, :, 6:9]
    d_flow = remaining_flow - remaining_flow.mean(dim=1).unsqueeze(1)
    d_alfven = alfven_velocity - alfven_velocity.mean(dim=1).unsqueeze(1)
    walen_slope = (d_flow * d_alfven).sum(dim=(1, 2)) / (d_alfven ** 2).sum(dim=(1, 2))
    return torch.abs(walen_slope)
