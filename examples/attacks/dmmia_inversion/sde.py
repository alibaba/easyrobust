import torch
import functools
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from tqdm import trange
from losses.poincare import poincare_loss
def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
    Returns:
      The standard deviation.
    """
    t = torch.tensor(t, device='cuda:7')
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))
def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.
    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.
    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(sigma ** t, device='cuda:7')

sigma = 5.0  # @param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

num_steps = 800  # @param {'type':'integer'}

def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std=marginal_prob_std_fn,
                           diffusion_coeff=diffusion_coeff_fn,
                           batch_size=64,
                           num_steps=num_steps,
                           device='cuda:7',
                           eps=1e-3):
    """Generate samples from score-based models with the Euler-Maruyama solver.
    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.
    Returns:
      Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 3, 32, 32, device=device) \
             * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            outputs =  score_model(x)
            targets_batch = torch.zeros(batch_size).long()
            #t = -1 * poincare_loss(outputs, targets_batch).mean()
            t = (-1 * outputs).softmax(-1)[:,0].mean()
            mean_x = x + (g ** 2)[:, None, None, None] * t * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)*2
            # Do not include any noise in the last sampling step.
    return mean_x

def main():
    device = 'cuda:7'
    from resnest.torch import resnest101
    score_model = resnest101(pretrained=False)
    checkpoint = torch.load('./Classifier_0.9999_no_val.pth',map_location='cpu')['model_state_dict']
    for key in list(checkpoint.keys()):
        if 'model.' in key:
            checkpoint[key.replace('model.', '')] = checkpoint[key]
            del checkpoint[key]
    score_model.load_state_dict(checkpoint)
    score_model.eval()
    score_model.to(device)
    
    samples = Euler_Maruyama_sampler(score_model, device=device)
    ims = tv.utils.make_grid(samples, normalize=True)
    plt.imshow(ims.cpu().numpy().transpose((1, 2, 0)))
    plt.savefig('./sde/iamge.png')
    
    
if __name__ == "__main__":
    main()
    
    

