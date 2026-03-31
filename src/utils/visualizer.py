import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class Visualizer:
    """
    Saves all output images:
    - Rendered avatar images
    - Depth maps
    - 3D mesh visualization
    - Training loss curves
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "renders").mkdir(exist_ok=True)
        (self.output_dir / "depth_maps").mkdir(exist_ok=True)
        (self.output_dir / "mesh").mkdir(exist_ok=True)
        (self.output_dir / "loss_curves").mkdir(exist_ok=True)
        print(f"✅ Visualizer ready → {self.output_dir}")

    def save_render(self, rgb: torch.Tensor, epoch: int, step: int = 0):
        """
        Save rendered avatar image.
        Args:
            rgb   : (H, W, 3) or (N, 3) tensor of RGB values
            epoch : current epoch number
            step  : current step number
        """
        if rgb.dim() == 2:
            H = W = int(rgb.shape[0] ** 0.5)
            rgb = rgb.reshape(H, W, 3)

        img = rgb.detach().cpu().numpy()
        img = np.clip(img, 0, 1)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img)
        ax.set_title(f"Rendered Avatar — Epoch {epoch} Step {step}", fontsize=12)
        ax.axis("off")
        plt.tight_layout()

        path = self.output_dir / "renders" / f"render_epoch{epoch:04d}_step{step:04d}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"🖼️  Render saved → {path}")
        return str(path)

    def save_depth_map(self, depth: torch.Tensor, epoch: int, step: int = 0):
        """
        Save depth map visualization.
        Args:
            depth : (H, W) or (N,) tensor of depth values
            epoch : current epoch number
        """
        if depth.dim() == 1:
            H = W = int(depth.shape[0] ** 0.5)
            depth = depth.reshape(H, W)

        d = depth.detach().cpu().numpy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(d, cmap="plasma")
        ax.set_title(f"Depth Map — Epoch {epoch} Step {step}", fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()

        path = self.output_dir / "depth_maps" / f"depth_epoch{epoch:04d}_step{step:04d}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"🗺️  Depth map saved → {path}")
        return str(path)

    def save_loss_curve(self, losses: list, title: str = "Training Loss"):
        """
        Save training loss curve.
        Args:
            losses : list of loss values per epoch
            title  : plot title
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses, color="#6366f1", linewidth=2, label="Train Loss")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / "loss_curves" / "training_loss.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📈 Loss curve saved → {path}")
        return str(path)

    def save_mesh_visualization(self, verts: np.ndarray, faces: np.ndarray, epoch: int):
        """
        Save 3D mesh visualization as 4 angle views.
        Args:
            verts : (V, 3) mesh vertices
            faces : (F, 3) mesh faces
            epoch : current epoch
        """
        fig = plt.figure(figsize=(14, 4))
        angles = [0, 45, 90, 180]

        for i, angle in enumerate(angles):
            ax = fig.add_subplot(1, 4, i+1, projection="3d")
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], verts[:, 2],
                triangles=faces,
                cmap="coolwarm",
                alpha=0.85,
                linewidth=0.1
            )
            ax.view_init(elev=20, azim=angle)
            ax.set_title(f"{angle}°", fontsize=10)
            ax.axis("off")

        fig.suptitle(f"3D Mesh — Epoch {epoch} | {len(verts):,} verts · {len(faces):,} faces", fontsize=12)
        plt.tight_layout()

        path = self.output_dir / "mesh" / f"mesh_epoch{epoch:04d}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"🧊 Mesh visualization saved → {path}")
        return str(path)

    def save_output_grid(self, rgb, depth, epoch: int):
        """
        Save a side-by-side grid of render + depth map.
        """
        if isinstance(rgb, torch.Tensor):
            if rgb.dim() == 2:
                H = W = int(rgb.shape[0] ** 0.5)
                rgb = rgb.reshape(H, W, 3)
            rgb = rgb.detach().cpu().numpy()

        if isinstance(depth, torch.Tensor):
            if depth.dim() == 1:
                H = W = int(depth.shape[0] ** 0.5)
                depth = depth.reshape(H, W)
            depth = depth.detach().cpu().numpy()

        rgb   = np.clip(rgb, 0, 1)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("Rendered Avatar", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(depth, cmap="plasma")
        axes[1].set_title("Depth Map", fontsize=13)
        axes[1].axis("off")

        fig.suptitle(f"Neural Avatar Pipeline — Epoch {epoch}", fontsize=14, y=1.02)
        plt.tight_layout()

        path = self.output_dir / "renders" / f"grid_epoch{epoch:04d}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"🎨 Output grid saved → {path}")
        return str(path)
