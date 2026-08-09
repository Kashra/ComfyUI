"""
Regression test for the MiniMax H3 VAE chunk-boundary darkening bug (#15426).

encode_temporal() used to split the input into independent clip_length
chunks. Because the encoder is built from CausalConv3d (zero-pads the
temporal front), every chunk after the first was encoded as if it were
the start of a brand new video, producing a discontinuity in the latents
at every clip_length boundary -> periodic dark frames on decode.

The fix prepends the previous chunk's real frames as context before
encoding, then keeps only the tokens belonging to the current chunk.

This test encodes a video in one continuous, unchunked pass (the "ground
truth" the model would see if VRAM allowed it) and compares it against
both the fixed encode_temporal() and a reconstruction of the old,
context-free chunking. The fixed path must land much closer to the
ground truth than the old one does.
"""
from unittest.mock import patch

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.minimax.vae as vae_mod  # noqa: E402


class _IdentityDecoderStub(torch.nn.Module):
    # ViT3DDecoder hardcodes num_layers=36 and isn't parameterized by the
    # small dims used here, so a full instance would use several GB. This
    # test only exercises the encoder side, so stub the decoder out.
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def _make_tiny_vae():
    with patch.object(vae_mod, "ViT3DDecoder", _IdentityDecoderStub):
        vae = vae_mod.MiniMaxH3VideoVAE(
            ch=32,
            ch_mult=(1, 2),
            num_res_blocks=1,
            space_down=(2, 1),
            time_down=(2, 1),
            z_channels=4,
            embed_dim=4,
            clip_length=6,
            token_drop=1,
            tiling=False,
        )
    # comfy.ops leaves freshly-constructed parameters uninitialized (they're
    # normally filled by loading a checkpoint), so seed them here.
    with torch.no_grad():
        for p in vae.parameters():
            p.normal_(0, 0.02)
    return vae


def _old_encode_temporal(vae, x):
    # Pre-fix behavior: independent, non-overlapping chunks with no context.
    num_chunks = x.shape[2] // vae.clip_length
    z_list = []
    for i in range(num_chunks):
        clip_x = x[:, :, i * vae.clip_length:(i + 1) * vae.clip_length, :, :]
        z_list.append(vae._adaptive_encode(clip_x))
    z = torch.cat(z_list, dim=2)
    if vae.token_drop > 0:
        z = z[:, :, :-vae.token_drop]
    return z


def test_encode_temporal_is_closer_to_unchunked_reference_than_old_chunking():
    torch.manual_seed(0)
    vae = _make_tiny_vae()
    vae.eval()

    # 3 chunks worth of frames (18), continuous content so any boundary
    # discontinuity comes from the encoder's causal padding, not the input.
    x = torch.randn(1, 3, 18, 16, 16)

    with torch.no_grad():
        z_reference = vae._adaptive_encode(x)          # single, unchunked pass
        if vae.token_drop > 0:
            z_reference = z_reference[:, :, :-vae.token_drop]

        z_fixed = vae.encode_temporal(x)                # fixed, chunked path
        z_old = _old_encode_temporal(vae, x)             # pre-fix chunked path

    assert z_reference.shape == z_fixed.shape == z_old.shape

    diff_fixed = (z_reference - z_fixed).abs().mean().item()
    diff_old = (z_reference - z_old).abs().mean().item()

    assert diff_fixed < diff_old, (
        f"encode_temporal() (mean diff {diff_fixed:.6f}) should land closer "
        f"to the unchunked reference than the old context-free chunking "
        f"(mean diff {diff_old:.6f}) -- regression of #15426."
    )
    # Sanity ceiling so a future unrelated regression still trips the test
    # even if it happens to stay below diff_old.
    assert diff_fixed < 1e-3