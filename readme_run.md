### requirement

1. 用清华源

```
# 移除可能残留的 channel（如果有）
conda config --remove-key channels

# 添加清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

# 显示 channel URL
conda config --set show_channel_urls yes
```

确认生效：

```
conda config --show channels
```

输出应类似：

```
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```

2. 创建 `gave_env` 环境并确保干净（删除旧 numpy、MKL 等包）：

```
conda create -n gave_env python=3.10 nomkl
conda activate gave_env
conda remove --force numpy numpy-base
```

3. 安装科学计算依赖 + CPU PyTorch：

```
conda install numpy pandas scikit-learn matplotlib tqdm
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install datatable psutil gin-config
```

4. 验证

```
python -c "import torch; import numpy as np; print(torch.__version__); print(torch.cuda.is_available()); print(np.__version__)"
```

### change path

### mock data


