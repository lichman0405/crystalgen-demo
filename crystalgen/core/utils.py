import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import uuid

def save_point_cloud(array: np.ndarray, base_dir="outputs"):
    """
    Save point cloud to a timestamped subdirectory under outputs/.
    Also generates a 3D scatter plot (colored by atomic number).
    """
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    out_dir = Path(base_dir) / f"run_{now}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / "point_cloud.npy"
    np.save(npy_path, array)

    # Optional: save xyz format
    xyz_path = out_dir / "point_cloud.xyz"
    with open(xyz_path, "w") as f:
        f.write(f"{len(array)}\n\n")
        for atom in array:
            z = int(atom[3] * 100)
            x, y, z_coord = atom[:3]
            f.write(f"C {x:.5f} {y:.5f} {z_coord:.5f}\n")

    # Save visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = array[:, 3]  # normalized Z
    p = ax.scatter(array[:, 0], array[:, 1], array[:, 2], c=colors, cmap='viridis', s=15)
    ax.set_title("CrystalGen Point Cloud")
    fig.colorbar(p, ax=ax, label="Normalized Atomic Number")
    fig.savefig(out_dir / "point_cloud.png", dpi=300)
    plt.close(fig)

    print(f"[✓] Saved point cloud, xyz, and image to: {out_dir}")
    return out_dir

def normalize_atomic_number(z_array):
    return np.clip(z_array / 100.0, 0, 1)
