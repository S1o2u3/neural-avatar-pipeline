import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class DatasetLoader:
    """
    Unified dataset loader for:
    - ZJU-MoCap  (human motion)
    - NeuMan     (neural human)
    - HuMMan     (multi-modal human)
    """

    SUPPORTED_DATASETS = ["zjumocap", "neuman", "humman"]

    def __init__(self, dataset_name: str, data_root: str, split: str = "train"):
        """
        Args:
            dataset_name : one of 'zjumocap', 'neuman', 'humman'
            data_root    : path to the dataset folder
            split        : 'train', 'val', or 'test'
        """
        dataset_name = dataset_name.lower()
        assert dataset_name in self.SUPPORTED_DATASETS, \
            f"Dataset must be one of {self.SUPPORTED_DATASETS}"

        self.dataset_name = dataset_name
        self.data_root    = Path(data_root)
        self.split        = split

        print(f"✅ Loading [{dataset_name.upper()}] | split={split} | root={data_root}")

        # Route to the correct loader
        if dataset_name == "zjumocap":
            self.dataset = ZJUMoCapDataset(self.data_root, split)
        elif dataset_name == "neuman":
            self.dataset = NeuManDataset(self.data_root, split)
        elif dataset_name == "humman":
            self.dataset = HuMManDataset(self.data_root, split)

    def get_loader(self, batch_size: int = 4, shuffle: bool = True):
        """Returns a PyTorch DataLoader ready for training."""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )

    def __len__(self):
        return len(self.dataset)

    def summary(self):
        print(f"\n📊 Dataset Summary")
        print(f"   Name   : {self.dataset_name.upper()}")
        print(f"   Split  : {self.split}")
        print(f"   Samples: {len(self.dataset)}")
        print(f"   Root   : {self.data_root}\n")


# ─────────────────────────────────────────────
# ZJU-MoCap Dataset
# ─────────────────────────────────────────────
class ZJUMoCapDataset(Dataset):
    """
    ZJU-MoCap: multi-view human motion dataset.
    Expected structure:
        data_root/
            CoreView_XXX/
                images/
                annots.npy
    """

    def __init__(self, data_root: Path, split: str):
        self.data_root = data_root
        self.split     = split
        self.samples   = self._load_samples()

    def _load_samples(self):
        samples = []
        scenes  = sorted(self.data_root.glob("CoreView_*"))
        for scene in scenes:
            image_dir = scene / "images"
            if image_dir.exists():
                for img in sorted(image_dir.glob("*.jpg")):
                    samples.append({"image": img, "scene": scene.name})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "image_path" : str(sample["image"]),
            "scene"      : sample["scene"],
            "dataset"    : "zjumocap"
        }


# ─────────────────────────────────────────────
# NeuMan Dataset
# ─────────────────────────────────────────────
class NeuManDataset(Dataset):
    """
    NeuMan: neural human NeRF dataset.
    Expected structure:
        data_root/
            <scene_name>/
                images/
                transforms.json
    """

    def __init__(self, data_root: Path, split: str):
        self.data_root = data_root
        self.split     = split
        self.samples   = self._load_samples()

    def _load_samples(self):
        samples = []
        scenes  = [d for d in self.data_root.iterdir() if d.is_dir()]
        for scene in scenes:
            transforms_file = scene / "transforms.json"
            if transforms_file.exists():
                with open(transforms_file) as f:
                    meta = json.load(f)
                for frame in meta.get("frames", []):
                    samples.append({
                        "scene"    : scene.name,
                        "filepath" : scene / frame.get("file_path", ""),
                        "transform": frame.get("transform_matrix", [])
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "image_path" : str(sample["filepath"]),
            "scene"      : sample["scene"],
            "transform"  : np.array(sample["transform"], dtype=np.float32),
            "dataset"    : "neuman"
        }


# ─────────────────────────────────────────────
# HuMMan Dataset
# ─────────────────────────────────────────────
class HuMManDataset(Dataset):
    """
    HuMMan: multi-modal human dataset with SMPL annotations.
    Expected structure:
        data_root/
            <subject_id>/
                kinect_color/
                smpl_params.npy
    """

    def __init__(self, data_root: Path, split: str):
        self.data_root = data_root
        self.split     = split
        self.samples   = self._load_samples()

    def _load_samples(self):
        samples  = []
        subjects = sorted(self.data_root.glob("p*"))
        for subject in subjects:
            color_dir   = subject / "kinect_color"
            smpl_params = subject / "smpl_params.npy"
            if color_dir.exists():
                for img in sorted(color_dir.glob("*.png")):
                    samples.append({
                        "image"      : img,
                        "subject"    : subject.name,
                        "smpl_params": smpl_params if smpl_params.exists() else None
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "image_path" : str(sample["image"]),
            "subject"    : sample["subject"],
            "has_smpl"   : sample["smpl_params"] is not None,
            "dataset"    : "humman"
        }
