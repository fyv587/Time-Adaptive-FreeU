import torch
import torch.fft as fft
from diffusers.utils import is_torch_version
from typing import Any, Dict, Optional, Tuple


def isinstance_str(x: object, cls_name: str):
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


def Fourier_filter(x: torch.Tensor, threshold: int, scale):
    """
    x: (B,C,H,W)
    threshold: int, half-size of low-frequency square window
    scale: float or 0-dim tensor
    """
    dtype = x.dtype
    device = x.device

    x = x.to(torch.float32)

    # FFT over H,W
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape

    mask = torch.ones((B, C, H, W), device=device, dtype=x_freq.dtype)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype)


def register_upblock2d(model):
    def up_forward(self):
        def forward(hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
            for resnet in self.resnets:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

                    if is_torch_version(">=", "1.11.0"):
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                        )
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb
                        )
                else:
                    hidden_states = resnet(hidden_states, temb)

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        return forward

    for upsample_block in model.unet.up_blocks:
        if isinstance_str(upsample_block, "UpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)


def _get_dynamic_freeu_params(self, fallback_b1, fallback_b2, fallback_s1, fallback_s2):
    """
    If self.schedule exists and self._cur_t exists, use schedule(t, k) to get b/s.
    Otherwise fall back to fixed scalars.
    Returns: b1,b2,s1,s2 as 0-dim tensors.
    """
    schedule = getattr(self, "schedule", None)
    t = getattr(self, "_cur_t", None)
    k = getattr(self, "_cur_k", None)  

    device, dtype = None, None
    if schedule is not None:
        try:
            p = next(schedule.parameters())
            device, dtype = p.device, p.dtype
        except StopIteration:
            pass

    def to_scalar_tensor(v):
        if torch.is_tensor(v):
            return v
        if device is None:
            return torch.tensor(float(v))
        return torch.tensor(float(v), device=device, dtype=dtype if dtype is not None else torch.float32)

    if schedule is None or t is None:
        return (to_scalar_tensor(fallback_b1),
                to_scalar_tensor(fallback_b2),
                to_scalar_tensor(fallback_s1),
                to_scalar_tensor(fallback_s2))

    # 把 k 传进 schedule
    b1, b2, s1, s2 = schedule(t)
 
    b1 = b1.mean()
    b2 = b2.mean()
    s1 = s1.mean()
    s2 = s2.mean()

    return b1, b2, s1, s2


def register_free_upblock2d(model, b1=1.2, b2=1.4, s1=0.9, s2=0.2, schedule=None):
    """
    Works in two modes:
    1) Fixed FreeU: pass b1,b2,s1,s2 (schedule=None)
    2) Time-adaptive FreeU: pass schedule=TimeFreeUSchedule and set blk._cur_t=t externally
    """

    def up_forward(self):
        def forward(hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
            for resnet in self.resnets:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # --- Dynamic (or fixed) params ---
                b1v, b2v, s1v, s2v = _get_dynamic_freeu_params(self, self.b1, self.b2, self.s1, self.s2)

                # --------------- FreeU code -----------------------
                if hidden_states.shape[1] == 1280:
                    # scale first half channels (avoid inplace for autograd stability)
                    hidden_states = torch.cat(
                        [hidden_states[:, :640] * b1v, hidden_states[:, 640:]],
                        dim=1,
                    )
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=s1v)

                if hidden_states.shape[1] == 640:
                    hidden_states = torch.cat(
                        [hidden_states[:, :320] * b2v, hidden_states[:, 320:]],
                        dim=1,
                    )
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=s2v)
                # ---------------------------------------------------------

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

                    if is_torch_version(">=", "1.11.0"):
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                        )
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb
                        )
                else:
                    hidden_states = resnet(hidden_states, temb)

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        return forward

    for upsample_block in model.unet.up_blocks:
        if isinstance_str(upsample_block, "UpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)
            setattr(upsample_block, "b1", float(b1))
            setattr(upsample_block, "b2", float(b2))
            setattr(upsample_block, "s1", float(s1))
            setattr(upsample_block, "s2", float(s2))
            setattr(upsample_block, "schedule", schedule)


def register_crossattn_upblock2d(model):
    def up_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            for resnet, attn in zip(self.resnets, self.attentions):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states,
                        encoder_hidden_states,
                        None,
                        None,
                        cross_attention_kwargs,
                        attention_mask,
                        encoder_attention_mask,
                        **ckpt_kwargs,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        return forward

    for upsample_block in model.unet.up_blocks:
        if isinstance_str(upsample_block, "CrossAttnUpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)


def register_free_crossattn_upblock2d(model, b1=1.2, b2=1.4, s1=0.9, s2=0.2, schedule=None):
    def up_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            for resnet, attn in zip(self.resnets, self.attentions):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # --- Dynamic (or fixed) params ---
                b1v, b2v, s1v, s2v = _get_dynamic_freeu_params(self, self.b1, self.b2, self.s1, self.s2)

                # --------------- FreeU code -----------------------
                if hidden_states.shape[1] == 1280:
                    hidden_states = torch.cat(
                        [hidden_states[:, :640] * b1v, hidden_states[:, 640:]],
                        dim=1,
                    )
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=s1v)

                if hidden_states.shape[1] == 640:
                    hidden_states = torch.cat(
                        [hidden_states[:, :320] * b2v, hidden_states[:, 320:]],
                        dim=1,
                    )
                    res_hidden_states = Fourier_filter(res_hidden_states, threshold=1, scale=s2v)
                # ---------------------------------------------------------

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states,
                        encoder_hidden_states,
                        None,
                        None,
                        cross_attention_kwargs,
                        attention_mask,
                        encoder_attention_mask,
                        **ckpt_kwargs,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )[0]

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        return forward

    for upsample_block in model.unet.up_blocks:
        if isinstance_str(upsample_block, "CrossAttnUpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)
            setattr(upsample_block, "b1", float(b1))
            setattr(upsample_block, "b2", float(b2))
            setattr(upsample_block, "s1", float(s1))
            setattr(upsample_block, "s2", float(s2))
            setattr(upsample_block, "schedule", schedule)