import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import argparse
from tqdm import tqdm
from reconstruction.mesh_generator import MeshGenerator
from rendering.sdf_renderer import SDFRenderer
from utils.spatial_hash import SpatialHashGrid
from utils.ray_sampler import BatchedRaySampler


def train(args):
    print("\n🚀 Neural Avatar Pipeline — Training Started")
    print(f"   Dataset  : {args.dataset}")
    print(f"   Epochs   : {args.epochs}")
    print(f"   Batch    : {args.batch_size}")
    print(f"   Device   : {args.device}\n")

    # ── Initialize modules ──────────────────────────
    mesh_gen    = MeshGenerator(resolution=args.resolution, device=args.device)
    renderer    = SDFRenderer(device=args.device)
    ray_sampler = BatchedRaySampler(device=args.device)
    spatial_hash= SpatialHashGrid()

    mesh_gen.summary()
    ray_sampler.summary()

    # ── Optimizer ───────────────────────────────────
    optimizer = torch.optim.Adam(
        list(mesh_gen.parameters()),
        lr=args.lr
    )

    # ── Training Loop ───────────────────────────────
    for epoch in range(1, args.epochs + 1):
        mesh_gen.train()
        epoch_loss = 0.0

        # Dummy batch for now — replace with real DataLoader
        for step in tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()

            # Random sample points in unit cube
            pts = torch.rand(args.batch_size, 3, device=args.device) * 2 - 1

            # Forward pass
            out  = mesh_gen(pts)
            sdf  = out["sdf"]

            # SDF loss — points should have near-zero SDF on surface
            loss = torch.mean(sdf ** 2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"✅ Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.6f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/epoch_{epoch:04d}.pt"
            torch.save({
                "epoch"     : epoch,
                "model"     : mesh_gen.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "loss"      : avg_loss
            }, ckpt_path)
            print(f"💾 Checkpoint saved → {ckpt_path}")

    print("\n🎉 Training Complete!")

    # Extract final mesh
    print("\n🔄 Extracting final mesh...")
    mesh_gen.eval()
    verts, faces = mesh_gen.extract_mesh()
    if verts is not None:
        print(f"✅ Mesh ready | {len(verts)} vertices | {len(faces)} faces")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Avatar Pipeline Trainer")
    parser.add_argument("--dataset",          type=str,   default="neuman")
    parser.add_argument("--data_root",        type=str,   default="data/")
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=1024)
    parser.add_argument("--steps_per_epoch",  type=int,   default=100)
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--resolution",       type=int,   default=128)
    parser.add_argument("--device",           type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
