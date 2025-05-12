import numpy as np

class PointCloudInitializer:
    def __init__(self, num_atoms: int = 256, seed: int = None):
        self.num_atoms = num_atoms
        self.rng = np.random.default_rng(seed)

    def generate(self) -> np.ndarray:
        """
        Generates a random point cloud with shape [num_atoms, 4].
        First three columns are fractional coordinates (x, y, z) in [0, 1],
        Fourth column is a normalized atomic number in [0, 1] (original Z / 100).
        """
        coords = self.rng.random((self.num_atoms, 3))  # x, y, z in [0, 1]
        atomic_numbers = self.rng.integers(low=1, high=87, size=(self.num_atoms, 1)) / 100.0  # Z in [1, 86]
        point_cloud = np.hstack((coords, atomic_numbers))
        return point_cloud

if __name__ == "__main__":
    initializer = PointCloudInitializer(seed=42)
    point_cloud = initializer.generate()
    print("Sample point cloud:")
    print(point_cloud[:5])  # Show first 5 atoms
