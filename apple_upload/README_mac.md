# M5 Pro 上要跑的实验

论文里 Apple 平台的所有数字都要在这台 MacBook 上重测。旧的 M4 Pro 数据来自更早的代码和测法，不能再用。

## 0. 准备（约 20 分钟，其中大部分是生成数据）

```bash
xcode-select --install          # 如果还没装命令行工具
brew install cmake node         # cmake 编译原生程序，node 跑浏览器实验
cd <解压出来的 webgpu_library>
bash apple_upload/gen_micro_data_mac.sh "e2 e6"
```

数据生成说明：
- `e6` 给微基准用，`e2` 给融合实验用，两个范围合计约 4 GB，写在 `public/data/`。
- 只想先跑微基准的话，用 `bash apple_upload/gen_micro_data_mac.sh e6`，约 2 GB。

## 1. 微基准：交集 1M–128M（必须）

```bash
bash apple_upload/run_micro_mac.sh
```

- **测的是什么：** 和 A100、RTX 3060 完全相同的协议。每个尺寸先跑 1000 ms 无关的 GPU 负载预热，再 10 次预热加 100 次计时，一共 3 个进程。时间来自 timestamp query，只算 GPU 时间。
- **结果在哪：** `apple_logs/m5pro_wgpu_micro_e6_r*.log`，每个尺寸一行 `[micro-result]` JSON，里面有每一次运行的时间。
- **顺带验证 Challenge 1：** 128M 这一档的 Lookback 需要超过 65,535 个 workgroup，会用二维 dispatch，正是当年 M4 Pro 上死锁的情形。如果 128M 正常跑完，就说明 M5 Pro 上不再出现这个问题。如果卡住不动，请直接 Ctrl-C 并告诉我。

## 2. ECLAT（必须）

```bash
bash apple_upload/run_eclat_mac.sh
```

- **数据：** 仓库里不带数据。先跑 `bash scripts/download_fimi.sh`，它会把 chess、kosarak、webdocs 下载到 `public/fimi/`（webdocs 1.48 GB，脚本自动解压）。ECLAT 脚本找不到 `eclat_cpp/data/` 时会自动改用 `public/fimi/`。
- **参数：** 和 A100 一致，3 次预热加 10 次计时，每块最多 512 MB 输入。
- **结果在哪：** `apple_logs/m5pro_wgpu_eclat_*.log`。

## 3. 融合实验：4-step 对 2-step（必须，只能在浏览器里跑）

```bash
bash apple_upload/run_browser_mac.sh fusion      # e2，交集和并集，对应论文的融合表
bash apple_upload/run_browser_mac.sh fusion-e6   # e6，只有交集，对应正文里 e6 的那句
```

- **为什么要跑：** reviewer 要求在所有平台上给出融合的加速比，Metal 是重点。
- **为什么用浏览器：** 原生程序只实现了 2-step，4-step 只有浏览器版有。
- **两个范围都要：** 论文的融合表用 e2，正文另有一句讲 e6 的交集加速比，所以两个都要测。
- **结果在哪：** `apple_logs/m5pro_chrome_fusion_*.log` 和 `m5pro_chrome_fusion_e6_*.log`，每个尺寸一行 `[fusion-result]` JSON。
- **如果无头 Chrome 不给 WebGPU：** 脚本会提示。这时手动跑：先 `npm run serve`，再用 Chrome 打开脚本里打印的那个网址，等页面输出 `[bench-done]`，把控制台内容存成文件发我。

## 4. 浏览器里的微基准和 ECLAT（必须）

```bash
bash apple_upload/run_browser_mac.sh micro
bash apple_upload/run_browser_mac.sh eclat
```

- **为什么也要跑：** 论文里 RTX 3060 和 Intel 的 WebGPU 数字都来自 Chrome，只有 A100 用原生运行时。Apple 两种都测了，写作时才能选一个口径，跨平台图也才对得齐。
- **结果在哪：** `apple_logs/m5pro_chrome_micro_*.log` 和 `m5pro_chrome_eclat_*.log`。
- **数据位置：** 浏览器从站点根目录读数据，脚本会自动把 `eclat_cpp/data/*.dat` 复制到 `public/fimi/`。
- **浏览器里的 webdocs 默认不跑：** 下载到本地之后，它在浏览器里还要先转成二进制 tidset（`node scripts/convert_fimi_to_tidsets.js`），转完再用 `DS=chess,kosarak,webdocs bash apple_upload/run_browser_mac.sh eclat`。原生那一路不需要转换，直接读 .dat。

## 跑完之后

把整个 `apple_logs/` 打包发我：

```bash
tar -czf apple_logs_$(date +%F).tar.gz apple_logs
```

## 可能出问题的地方

1. **subgroups 不支持：** DPI kernel 用了 subgroup 指令。程序启动时会打印 `Subgroups: supported / NOT supported`。如果显示不支持，微基准和 ECLAT 都会失败，请把开头几行输出发我，我改成不依赖 subgroup 的版本。
2. **缓冲区上限：** 128M 的一路输入需要 512 MB 的缓冲区。程序会打印 `maxBufferSize`，如果某个尺寸被跳过，日志里会写 `(skipped: buffer exceeds maxBufferSize)`。
3. **散热降频：** 三个进程连着跑，机器会热。请接电源，保持盖子打开，中途不要跑别的重负载程序。
4. **磁盘：** 两个范围的数据加上编译产物大约 5 GB。
