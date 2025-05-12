# CrystalGen：MOF晶体结构生成模块（简化扩散版）

**CrystalGen** 是一个用于生成金属有机框架（MOF）结构的轻量级原型模块，基于点云表示和简化的 Diffusion 去噪策略构建。该项目为复现《Agentic AI for Discovery of MOFs》论文中的 CrystalGen 模块而设计，具有可拓展、可视化友好、工程化良好的特点。

---

## 🧩 项目功能

- 使用 `[256, 4]` 点云矩阵初始化 MOF 原子结构
- 采用 MLP + permutation-invariant 自注意力模块提取特征
- 基于重要性评分进行原子筛选（Toy Diffusion）
- 自动保存 `.npy`、`.xyz` 数据和 `.png` 可视化图像
- 兼容未来扩展 Classifier-Free Guidance（CFG）

---

## 📁 项目结构

```
crystalgen/
├── main.py                    # 主入口，运行一键生成
├── crystalgen/
│   └── core/
│       ├── initializer.py     # 初始化随机点云 [256, 4]
│       ├── network.py         # 网络结构（MLP + attention）
│       ├── diffusion.py       # 去噪模拟器
│       ├── utils.py           # 保存与可视化
│       └── embedding.py       # 条件控制预留接口（暂未启用）
├── outputs/                   # 每次生成的结构与图像将保存在此目录
├── requirements.txt
└── README.md
```

---

## 🚀 快速开始

```bash
pip install -r requirements.txt
python main.py
```

每次运行将自动创建带时间戳的输出文件夹，例如：

```
outputs/run_20240507-153245_a7e8c5b1/
├── point_cloud.npy
├── point_cloud.xyz
└── point_cloud.png
```

---

## 🧪 模型说明

- 输入：`[256, 4]` 点云张量（xyz坐标 + 原子编号）
- 网络结构：MLP → Self-Attention → Sigmoid重要性评分
- 输出：筛选后的结构子集，保存为可视化图与数据文件

---

## 📌 后续可拓展

- 加入条件控制（如 linker / SBU 信息）
- 接入真实 DDPM 扩散模型作为替代
- 结构转化为 CIF，供 QForge 等模块使用

---

## 🔖 协议

MIT License，自由使用与扩展。
