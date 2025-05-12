import torch
#from initializer import PointCloudInitializer
from crystalgen.core.initializer import PointCloudInitializer
from crystalgen.core.network import CrystalNet
#from network import CrystalNet  

class ToyDiffusionGenerator:
    def __init__(self, model=None, threshold=0.5, device="cpu"):
        self.device = device
        self.threshold = threshold
        self.model = model or CrystalNet()
        self.model.to(device)
        self.model.eval()

    def generate(self, seed=42):
        # Step 1: 初始化点云
        initializer = PointCloudInitializer(seed=seed)
        cloud = initializer.generate()  # numpy array [256, 4]

        # Step 2: 转换为 tensor
        x = torch.tensor(cloud, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, 256, 4]

        # Step 3: 通过网络获取每个原子的“保留分数”
        with torch.no_grad():
            importance = self.model(x)  # [1, 256, 1]

        importance = importance.squeeze(0).squeeze(-1).cpu().numpy()  # [256]
        mask = importance > self.threshold

        # Step 4: 返回被保留的原子点云
        denoised = cloud[mask]
        return denoised, importance, mask

if __name__ == "__main__":
    generator = ToyDiffusionGenerator(threshold=0.5)
    denoised_cloud, scores, mask = generator.generate()
    print(f"Retained {len(denoised_cloud)} atoms out of 256.")
    print("First 5 atoms (after filtering):")
    print(denoised_cloud[:10])
