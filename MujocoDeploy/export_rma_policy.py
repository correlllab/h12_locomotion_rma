"""Export trained RMA Phase 1 policy + encoder from Isaac Gym checkpoint.

Takes a bundled model_<iter>.pt and produces two standalone files:
  - policy.pt   : ActorCriticRecurrent state dict + architecture config
  - encoder.pt  : EnvFactorEncoder state dict + architecture config

Usage:
  python export_rma_policy.py --ckpt ../logs/h1_2_rma/<run>/model_5000.pt --out_dir exported_weights/
"""

import os
import sys
import argparse
import torch

# Add repo root so rsl_rl and rma imports work
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rsl_rl.modules import ActorCriticRecurrent
from rma.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg


def main():
    parser = argparse.ArgumentParser(description="Export RMA policy + encoder from training checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model_<iter>.pt checkpoint")
    parser.add_argument("--out_dir", type=str, default="exported_weights", help="Output directory")

    # Architecture params (must match training config)
    parser.add_argument("--num_actor_obs", type=int, default=55, help="47 env obs + 8 z_t")
    parser.add_argument("--num_critic_obs", type=int, default=58, help="50 priv obs + 8 z_t")
    parser.add_argument("--num_actions", type=int, default=12)
    parser.add_argument("--actor_hidden_dims", type=int, nargs="+", default=[32])
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[32])
    parser.add_argument("--rnn_hidden_size", type=int, default=64)
    parser.add_argument("--rnn_num_layers", type=int, default=1)
    parser.add_argument("--et_dim", type=int, default=9)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--encoder_hidden_dims", type=int, nargs="+", default=[256, 128])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"  Keys: {list(ckpt.keys())}")
    print(f"  Iteration: {ckpt.get('iter', '?')}")

    # --- Export policy ---
    policy_cfg = dict(
        num_actor_obs=args.num_actor_obs,
        num_critic_obs=args.num_critic_obs,
        num_actions=args.num_actions,
        actor_hidden_dims=args.actor_hidden_dims,
        critic_hidden_dims=args.critic_hidden_dims,
        rnn_type="lstm",
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        activation="elu",
    )

    # Verify by loading into model
    model = ActorCriticRecurrent(**policy_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Policy loaded OK: actor_obs={args.num_actor_obs}, actions={args.num_actions}, "
          f"LSTM({args.num_actor_obs}, {args.rnn_hidden_size})")

    policy_path = os.path.join(args.out_dir, "policy.pt")
    torch.save({
        "model_state_dict": ckpt["model_state_dict"],
        "cfg": policy_cfg,
    }, policy_path)
    print(f"  Saved policy -> {policy_path}")

    # --- Export encoder ---
    encoder_cfg = dict(
        in_dim=args.et_dim,
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.encoder_hidden_dims),
    )

    encoder = EnvFactorEncoder(EnvFactorEncoderCfg(**encoder_cfg))
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    print(f"  Encoder loaded OK: {args.et_dim} -> {args.latent_dim}")

    encoder_path = os.path.join(args.out_dir, "encoder.pt")
    torch.save({
        "encoder_state_dict": ckpt["encoder_state_dict"],
        "cfg": encoder_cfg,
    }, encoder_path)
    print(f"  Saved encoder -> {encoder_path}")

    # --- Quick sanity check ---
    dummy_obs = torch.randn(1, args.num_actor_obs)
    with torch.no_grad():
        action = model.act_inference(dummy_obs)
    print(f"\n  Sanity check: dummy obs -> action shape {action.shape}, "
          f"range [{action.min().item():.3f}, {action.max().item():.3f}]")

    dummy_et = torch.randn(1, args.et_dim)
    with torch.no_grad():
        z = encoder(dummy_et)
    print(f"  Sanity check: dummy e_t -> z_t shape {z.shape}, "
          f"range [{z.min().item():.3f}, {z.max().item():.3f}]")

    print(f"\nDone! Files saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
