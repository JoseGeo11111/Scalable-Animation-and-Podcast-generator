"""FastWan DMD sampling adapter for Wan2GP's EulerScheduler interface.

Source semantics: hao-ai-lab/FastVideo@33d81730e9fe423ab3c33dbc0451973267b80d92
  fastvideo/pipelines/basic/wan/stages/dmd.py
  fastvideo/models/utils.py:pred_noise_to_pred_video
  fastvideo/models/schedulers/scheduling_flow_match_euler_discrete.py:add_noise
  fastvideo/models/wan/definition.py:DMD_TRAINING_NOISE_SHIFT=8.0

Runtime use, after importing Wan2GP's module:
    import models.wan.any2video as wan_module
    from film15_dmd_scheduler import install
    original = install(wan_module)
Then select sample_solver='euler', sampling_steps=3, guidance scale=1.
Restore with wan_module.EulerScheduler = original. No vendor files are edited.

This implements the DMD recurrence, not an ODE integrator. Wan2GP conditioning
and initial-latent generation remain its own; this is not a bitwise claim of
whole-pipeline equivalence with FastVideo. A controlled visual A/B is required.
"""
import torch

SOURCE_COMMIT='33d81730e9fe423ab3c33dbc0451973267b80d92'

class DMDOutput:
    def __init__(self, prev_sample, pred_original_sample):
        self.prev_sample=prev_sample; self.pred_original_sample=pred_original_sample
    def __getitem__(self, index):
        if index!=0: raise IndexError(index)
        return self.prev_sample
    def __iter__(self): yield self.prev_sample

class FastWanDMDScheduler:
    is_stateful=False
    order=1
    def __init__(self, num_train_timesteps=1000, use_timestep_transform=True):
        if num_train_timesteps!=1000: raise ValueError('This adapter is for the1000-step FastWan checkpoint.')
        self.num_train_timesteps=1000
        # Official training table: float32 linspace1000..1, shift8, nearest lookup.
        base=torch.arange(1000,0,-1,dtype=torch.float32)/1000
        self._train_sigmas=8*base/(1+7*base)
        self._train_timesteps=self._train_sigmas*1000
        self.timesteps=None; self.sigmas=None
    def set_timesteps(self, num_inference_steps, device=None, shift=5.0):
        if num_inference_steps!=3: raise ValueError('FastWan DMD adapter requires exactly3steps.')
        self.num_inference_steps=3
        self.timesteps=torch.tensor([1000,757,522],dtype=torch.long,device=device)
        self.sigmas=torch.tensor([self._sigma(t) for t in (1000,757,522)]+[0.],device=device)
        return self.timesteps
    def _sigma(self,t):
        # Match official double conversion for x0 sigma lookup; table stays FP32.
        i=(self._train_timesteps.double()-float(t)).abs().argmin()
        return self._train_sigmas[i].item()
    def step(self, model_output, timestep, sample, return_dict=True, generator=None, **kwargs):
        if self.timesteps is None: raise ValueError('Call set_timesteps first.')
        t=float(timestep.item()) if torch.is_tensor(timestep) else float(timestep)
        idx=int((self.timesteps.double()-t).abs().argmin().item())
        if float(self.timesteps[idx].item())!=t: raise ValueError(f'Unexpected DMD timestep:{t}')
        # Official conversion calculates in double, then casts to prediction dtype.
        clean=(sample.double()-self._sigma(t)*model_output.double()).to(model_output.dtype)
        if idx==2:
            previous=clean
        else:
            # FastVideo draws in B,T,C,H,W. Wan2GP's latent layout is B,C,T,H,W.
            if sample.ndim!=5: raise ValueError('Expected Wan2GP B,C,T,H,W latents.')
            b,c,f,h,w=sample.shape
            rng_device=generator.device if generator is not None else sample.device
            noise=torch.randn((b,f,c,h,w),dtype=clean.dtype,device=rng_device,generator=generator)
            noise=noise.permute(0,2,1,3,4).to(sample.device)
            # add_noise uses FP32 sigma arithmetic then casts to noise dtype.
            sigma=torch.tensor(self._sigma(self.timesteps[idx+1].item()),device=sample.device,dtype=torch.float32)
            sigma=sigma.reshape(1,1,1,1,1)
            previous=((1-sigma)*clean+sigma*noise).to(noise.dtype)
        if not return_dict: return (previous,)
        return DMDOutput(previous,clean)
    def scale_model_input(self,sample,*args,**kwargs): return sample

def install(wan_any2video_module):
    """Patch only an explicitly supplied loaded module; return original to restore."""
    old=wan_any2video_module.EulerScheduler
    wan_any2video_module.EulerScheduler=FastWanDMDScheduler
    return old
