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
from utils.visualizer import Visualizer


def train(args):
    print("\n🚀 Neural Avatar Pipeline — Training Started")
    print(f"   Dataset  : {args.dataset}")
    print(f"   Epochs   : {args.epochs}")
    print(f"   Device   : {args.device}\n")

    # Initialize all modules
    mesh_gen     = MeshGenerator(resolution=args.resolution, device=args.device)
    renderer     = SDFRenderer(device=args.device)
    ray_sampler  = BatchedRaySampler(device=args.device)
    spatial_hash = SpatialHashGrid()
    visualizer   = Visualizer(output_dir=args.output_dir)

    mesh_gen.summary()
    ray_sampler.summary()

    optimizer = torch.optim.Adam(mesh_gen.parameters(), lr=args.lr)

    all_losses = []

    for epoch in range(1, args.epochs + 1):
        mesh_gen.train()
        epoch_loss = 0.0

        for step in tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()

            # Sample random 3D points
            pts = torch.rand(args.batch_size, 3, device=args.device) * 2 - 1

            # Forward pass
            out   = mesh_gen(pts)
            sdf   = out["sdf"]
            color = out["color"]

            # SDF loss + color regularization
            sdf_loss   = torch.mean(sdf ** 2)
            color_loss = torch.mean((color - 0.5) ** 2)
            loss       = sdf_loss + 0.1 * color_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / args.steps_per_epoch
        all_losses.append(avg_loss)
        print(f"✅ Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.6f}")

        # Save outputs every N epochs
        if epoch % args.save_every == 0:

            # Save checkpoint
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/epoch_{epoch:04d}.pt"
            torch.save({
                "epoch"    : epoch,
                "model"    : mesh_gen.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss"     : avg_loss
            }, ckpt_path)
            print(f"💾 Checkpoint → {ckpt_path}")

            # Generate and save output images
            mesh_gen.eval()
            with torch.no_grad():

                # Render a test image (32x32 grid of points)
                H = W = 32
                test_pts = torch.rand(H * W, 3, device=args.device) * 2 - 1
                out      = mesh_gen(test_pts)
                rgb      = out["color"].reshape(H, W, 3)
                sdf_vals = out["sdf"].reshape(H, W)

                # Save rendered avatar
                visualizer.save_render(rgb, epoch)

                # Save depth map
                visualizer.save_depth_map(sdf_vals, epoch)

                # Save side-by-side grid
                visualizer.save_output_grid(rgb, sdf_vals, epoch)

            # Save loss curve
            visualizer.save_loss_curve(all_losses, title="Neural Avatar Training Loss")

            # Extract and visualize mesh
            print("🔄 Extracting mesh...")
            verts, faces = mesh_gen.extract_mesh()
            if verts is not None:
                visualizer.save_mesh_visualization(verts, faces, epoch)

            mesh_gen.train()

    # Final loss curve
    visualizer.save_loss_curve(all_losses)
    print(f"\n🎉 Training Complete! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Avatar Pipeline Trainer")
    parser.add_argument("--dataset",         type=str,   default="neuman")
    parser.add_argument("--data_root",       type=str,   default="data/")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=1024)
    parser.add_argument("--steps_per_epoch", type=int,   default=100)
    parser.add_argument("--lr",              type=float, default=5e-4)
    parser.add_argument("--resolution",      type=int,   default=128)
    parser.add_argument("--save_every",      type=int,   default=10)
    parser.add_argument("--output_dir",      type=str,   default="outputs")
    parser.add_argument("--device",          type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
