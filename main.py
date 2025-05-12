from crystalgen.core.diffusion import ToyDiffusionGenerator
from crystalgen.core.utils import save_point_cloud

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    # 初始化生成器（可调整 threshold）
    generator = ToyDiffusionGenerator(threshold=0.5, device="cpu")

    # 运行 toy diffusion 模拟
    denoised_cloud, scores, mask = generator.generate(seed=20240507)

    # 保存结构 + 可视化图像
    save_point_cloud(denoised_cloud)

if __name__ == "__main__":
    main()
