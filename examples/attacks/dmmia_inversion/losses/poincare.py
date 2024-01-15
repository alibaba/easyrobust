import torch

def poincare_loss(outputs, targets, xi=1e-4):
    # Normalize logits
    u = outputs / torch.norm(outputs, p=1, dim=-1).unsqueeze(1)
    # Create one-hot encoded target vector
    v = torch.clip(torch.eye(outputs.shape[-1])[targets] - xi, 0, 1)
    v = v.to(u.device)
    # Compute squared norms
    u_norm_squared = torch.norm(u, p=2, dim=1)**2
    v_norm_squared = torch.norm(v, p=2, dim=1)**2
    diff_norm_squared = torch.norm(u - v, p=2, dim=1)**2
    # Compute delta
    delta = 2 * diff_norm_squared / ((1 - u_norm_squared) *
                                     (1 - v_norm_squared))
    # Compute distance
    loss = torch.arccosh(1 + delta)
    return loss

