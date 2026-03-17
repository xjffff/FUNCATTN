import trimesh
import hashlib
import os
import numpy as np

import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

import src.models as models

import trimesh
import hashlib


def _stable_int_seed(s: str, mod: int = 2**32) -> int:
    """Stable (cross-run/cross-machine) 32-bit seed from a string."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


class RNAPointCloudDataset(Dataset):
    """
    RNA point cloud segmentation dataset with uniform surface sampling.
    """

    def __init__(
        self,
        root_dir,
        train,
        k_eig,
        n_points=4096,
        sampling="fps",  # 'uniform', 'fps', 'random'
        use_cache=True,
        op_cache_dir=None,
    ):

        self.train = train
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.n_points = n_points
        self.sampling = sampling
        self.use_cache = use_cache
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 260

        self.verts_list = []
        self.faces_list = []
        self.labels_list = []

        # Load file list
        if self.train:
            with open(os.path.join(self.root_dir, "train.txt")) as f:
                this_files = [line.rstrip() for line in f]
        else:
            with open(os.path.join(self.root_dir, "test.txt")) as f:
                this_files = [line.rstrip() for line in f]

        print(f"Loading {len(this_files)} files with {sampling} sampling...")

        off_path = os.path.join(root_dir, "off")
        label_path = os.path.join(root_dir, "labels")

        for f in this_files:
            off_file = os.path.join(off_path, f)
            label_file = os.path.join(label_path, f[:-4] + ".txt")

            # 1. read mesh & label
            verts, faces = pp3d.read_mesh(off_file)
            labels = np.loadtxt(label_file).astype(int) + 1

            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces).long()
            labels = torch.tensor(labels).long()

            # 2. normalize positions
            verts = models.geometry.normalize_positions(verts)

            # 3. sampling
            V = verts.shape[0]
            n_sample = min(self.n_points, V)

            if n_sample < self.n_points:
                print(f"[WARN] mesh {f} has only {V} verts; using {V} points")

            sampled_verts, sampled_labels = self._sample_points(
                verts,
                faces,
                labels,
                n_sample,
                mesh_name=f,
            )

            # Point cloud: faces set to zeros for (compute DiffusionNet operator in point cloud format)
            pc_faces = torch.zeros((0, 3), dtype=torch.long)

            self.verts_list.append(sampled_verts)
            self.faces_list.append(pc_faces)
            self.labels_list.append(sampled_labels)

        # 4. precompute DiffusionNet operator
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = models.geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )

    def _pc_cache_path(self, mesh_name: str, n_sample: int) -> str:
        """Cache path for sampled point cloud + labels."""
        split = "train" if self.train else "test"
        subdir = os.path.join(self.cache_dir, "pc", split)
        os.makedirs(subdir, exist_ok=True)

        base = os.path.splitext(os.path.basename(mesh_name))[0]
        # Include sampling + n_sample so different settings do not collide
        return os.path.join(subdir, f"{base}__{self.sampling}__n{n_sample}.npz")

    def _sample_points(self, verts, faces, labels, n_sample, mesh_name):

        cache_file = self._pc_cache_path(mesh_name, n_sample)
        if self.use_cache and os.path.exists(cache_file):
            data = np.load(cache_file)
            sampled_verts = torch.from_numpy(data["points"]).float()
            sampled_labels = torch.from_numpy(data["labels"]).long()
            return sampled_verts, sampled_labels

        verts_np = verts.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        if self.sampling == "fps":
            # Farthest Point Sampling
            fps_mask = models.geometry.farthest_point_sampling(verts, n_sample)
            sampled_verts = verts[fps_mask]
            sampled_labels = labels[fps_mask]

        elif self.sampling == "random":
            # random sample indicies
            rand_idx = torch.randperm(verts.shape[0])[:n_sample]
            sampled_verts = verts[rand_idx]
            sampled_labels = labels[rand_idx]

        elif self.sampling == "uniform":
            # Uniformly sample points on the triangle surface (area-weighted)
            mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)

            seed = _stable_int_seed(f"{mesh_name}|uniform|{n_sample}")
            rng_state = np.random.get_state()
            np.random.seed(seed)
            try:
                sampled_points, face_idx = trimesh.sample.sample_surface(mesh, n_sample)
            finally:
                np.random.set_state(rng_state)

            sampled_verts = torch.from_numpy(sampled_points).float()

            # --- Face-based label transfer (majority vote over the 3 face vertices) ---
            # face_idx: (n_sample,) each entry is an index into faces_np
            tri_vidx = faces_np[
                face_idx
            ]  # (n_sample, 3) vertex indices of the hit triangle
            tri_labels = labels_np[tri_vidx]  # (n_sample, 3) labels at the 3 vertices

            # Majority vote (ties broken deterministically by picking the smallest label)
            # Works with many classes efficiently without looping over classes
            tri_labels_sorted = np.sort(tri_labels, axis=1)  # (n_sample, 3)
            a = tri_labels_sorted[:, 0]
            b = tri_labels_sorted[:, 1]
            c = tri_labels_sorted[:, 2]
            maj = np.where(
                a == b, a, np.where(b == c, b, a)
            )  # if all distinct -> pick smallest (a)

            sampled_labels = torch.from_numpy(maj).long()
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

        if self.use_cache:
            np.savez_compressed(
                cache_file,
                points=sampled_verts.detach().cpu().numpy().astype(np.float32),
                labels=sampled_labels.detach().cpu().numpy().astype(np.int64),
            )
        return sampled_verts, sampled_labels

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return (
            self.verts_list[idx],
            self.faces_list[idx],
            self.frames_list[idx],
            self.massvec_list[idx],
            self.L_list[idx],
            self.evals_list[idx],
            self.evecs_list[idx],
            self.gradX_list[idx],
            self.gradY_list[idx],
            self.labels_list[idx],
        )
