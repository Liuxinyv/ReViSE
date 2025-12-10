import torch
import traceback


class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False, is_training=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps, training=is_training)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, device='cpu'):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing


    def step(self, model_output, timestep, sample, to_final=False):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep.reshape(-1, 1)).abs(), dim=-1, keepdim=True)
        sigma = _extract_into_tensor(self.sigmas, timestep_id, noise.shape).to(noise.device)#([1000]),([2, 1]),([2, 16, 5, 44, 80])
        sample = (1 - sigma) * original_samples + sigma * noise#([2, 16, 5, 44, 80]),
        return sample#
    

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    

    def training_weight(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        # timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        timestep_id = torch.argmin((self.timesteps - timestep.reshape(-1, 1)).abs(), dim=-1, keepdim=True)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
    
    @torch.no_grad()
    def predict_x0(self, noisy_sample: torch.Tensor, v_pred: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        根据训练目标 v ≈ (ε - x)，以及 x_t = (1-σ)x + σ ε，
        反推出 x ≈ x_t - σ·v_pred

        Args:
            noisy_sample: x_t，形状 [B,C,F,H,W]
            v_pred: 模型预测的速度场，形状同上
            timestep: 当前步的时间戳 [B] 或 [B,1]

        Returns:
            pred_x0: 估计的原始 x，形状同 noisy_sample
        """
        if isinstance(timestep, torch.Tensor):
            t_cpu = timestep.cpu()
        else:
            t_cpu = timestep
        # 和 add_noise 保持一致的最近邻查找
        timestep_id = torch.argmin((self.timesteps - t_cpu.reshape(-1, 1)).abs(), dim=-1, keepdim=True)
        sigma = _extract_into_tensor(self.sigmas, timestep_id, noisy_sample.shape).to(noisy_sample.device)

        x0 = noisy_sample - sigma * v_pred
        return x0
    
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D tensor.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    with torch.autocast(device_type="cuda", enabled=False):
        if arr.device != timesteps.device:
            # import ipdb; ipdb.set_trace();
            for line in traceback.format_stack():
                print('info', line.strip())
            res = arr.to(device=timesteps.device)
            res = res[timesteps].float()
        else:
            res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res + torch.zeros(broadcast_shape, device=timesteps.device)