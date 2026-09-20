# WebGPU-Sets experiments for the M5 Pro

这个仓库只用来把 IPDPS 论文里 Apple 平台要重跑的实验传到 MacBook 上，不是完整的库仓库。

## 在 Mac 上怎么开始

```bash
git clone https://github.com/JiaxinLu98/Mac.git webgpu_library
cd webgpu_library
cat apple_upload/README_mac.md
```

`apple_upload/README_mac.md` 里写了要跑哪些实验、每一步的命令，以及可能出问题的地方。

## 仓库里有什么

| 目录 | 内容 |
|---|---|
| `apple_upload/` | 跑实验的脚本和中文说明 |
| `set_intersection_cpp/` | 微基准的原生程序（wgpu-native，Metal） |
| `eclat_cpp/` | ECLAT 的原生程序，以及 chess 和 kosarak 数据 |
| `src/` | 浏览器端的库和实验代码，融合实验只能在这里跑 |
| `scripts/` | 生成微基准数据、下载 FIMI 数据、转换 tidset 的脚本 |

## 数据不在仓库里，都在 Mac 上准备

```bash
bash scripts/download_fimi.sh                    # ECLAT 数据，下载到 public/fimi（webdocs 1.48 GB）
bash apple_upload/gen_micro_data_mac.sh "e2 e6"  # 微基准数组，约 4 GB
```

FIMI 数据集是公开的，脚本会自动下载并解压 webdocs。ECLAT 的脚本会先找 `eclat_cpp/data/`，找不到就用 `public/fimi/`，所以下载完直接跑即可。

## 跑完之后

把 `apple_logs/` 打包发回：

```bash
tar -czf apple_logs_$(date +%F).tar.gz apple_logs
```
