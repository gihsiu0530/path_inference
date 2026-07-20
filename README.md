# 離線路徑推論流程（Offline Trajectory Inference Pipeline）

本專案是一條**離線路徑推論（trajectory / path planning）流程**：
把原始 rosbag 轉成 224×224 的 RGB 影像、語意分割與里程計，切成 **0.5 秒**的序列，
最後用 ST-P3 模型推論車輛未來路徑，並輸出 L2 誤差統計與視覺化圖。

整條流程共四個步驟，執行順序如下：

```
原始 bag ─(Step1: bag_to_data.py)→ img/ + seg/ + odom.csv
         ─(Step2: resample.py)→ 0.5 秒序列 resample/
         ─(Step3: convert_cls4png_to_npy.py)→ seg 轉 .npy
         ─(Step4: park_L2_ASAP.py)→ 推論路徑
```

| 步驟 | 腳本 | 用途 |
|---|---|---|
| Step 1 | `bag_to_data.py` | **直接讀原始 bag**，做語意分割 + resize，輸出 `img/`、`seg/` PNG 與 `odom.csv` |
| Step 2 | `dataset/0624bkgd/resample.py` | 以 0.5 秒為間隔重採樣，產生 `resample/` |
| Step 3 | `convert_cls4png_to_npy.py` | 把彩色語意 PNG 轉成 class-id `.npy` |
| Step 4 | `park_L2_ASAP.py` | 載入模型推論路徑，輸出軌跡與 L2 誤差 |

> Step 1 的 `bag_to_data.py` 取代了舊的「`rosbag play` → `seg_real_time.py` 節點 → `rosbag record` → `extract_bag_data.py`」四步即時流程。
> 若仍需即時處理，舊做法保留於 [附錄：備選（即時）資料轉出流程](#附錄備選即時資料轉出流程)。

---

## 1. 環境需求

本流程用到**兩套不同的環境**：

### 1.1 資料轉檔環境（Step 1–3）：ROS noetic 的 `python3`

```bash
source /opt/ros/noetic/setup.bash
```

- 此環境的 `python3`（3.8）已同時具備 `rosbag`、`torch`、`transformers`、`cv2`、`numpy`，
  是執行 `bag_to_data.py` 所需的一切（也是 `seg_real_time.py` 的執行環境）。
- Step 1 首次執行會自 HuggingFace 下載模型 `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`。
- Step 2（`resample.py`，純標準函式庫）與 Step 3（`convert_cls4png_to_npy.py`，需 `numpy` + `PIL`）
  也可在此環境執行。

### 1.2 推論環境（Step 4）：conda `stp3_env`

```bash
conda env create -f environment.yml   # 首次建立
conda activate stp3_env               # 每次使用前啟用
```

- Python 3.9.23 / PyTorch / torchvision / transformers / opencv / pandas / pytorch-lightning（詳見 [environment.yml](environment.yml)）。
- **推論步驟務必在 `stp3_env` 內執行，勿用系統 python。**
- 此環境與 1.1 的 noetic 環境不同（py3.9 vs py3.8，無法互相 import `rosbag`），請分開使用。

---

## 2. 需要的 ROS Topic

`bag_to_data.py`（Step 1）只需要原始 bag 內的兩個 topic：

| 用途 | Topic 名稱 | 說明 / 參數 |
|---|---|---|
| 相機 RGB 輸入 | `/zed2i/zed_node/right_raw/image_raw_color` | `bag_to_data.py` 的 `--rgb-topic` 預設值；亦為 `seg_real_time.py` 的預設訂閱 topic（[seg_real_time.py:216](seg_real_time.py#L216)） |
| 里程計 | `/odom` | `bag_to_data.py` 的 `--odom-topic` 預設值（[extract_bag_data.py:16](extract_bag_data.py#L16)） |

> 原始 ZED bag **通常沒有深度 topic**，因此整條流程一律不使用深度（見 [注意事項](#5-注意事項--常見陷阱)）。
> 若你的相機用的是 `rgb_raw` 而非 `right_raw`，執行時加 `--rgb-topic /zed2i/zed_node/rgb_raw/image_raw_color` 覆寫即可。

### 4 類語意調色盤（分割輸出與 Step 3 轉檔必須一致）

| class id | 類別 | RGB |
|---|---|---|
| 0 | road（道路） | `(128, 64, 128)` |
| 1 | person（行人） | `(220, 20, 60)` |
| 2 | movable（可移動物） | `(0, 0, 142)` |
| 3 | static（靜態物） | `(70, 70, 70)` |

調色盤定義於 [seg_real_time.py:35-40](seg_real_time.py#L35-L40) 與 [convert_cls4png_to_npy.py:10-15](convert_cls4png_to_npy.py#L10-L15)。
`bag_to_data.py` 重用 `seg_real_time.py` 的 `colorize_cls4_rgb`，並在存檔時做 RGB→BGR 轉換，
確保磁碟 PNG 的顏色值與 `PALETTE4` 完全一致，Step 3 才不會因未知顏色而報錯。

---

## 3. 逐步流程

以下以資料根目錄 `<root>` 為例（例如 `dataset/0624bkgd`），各 video 依 `video1 ... videoN` 命名，
每個 `videoN/` 內放**恰好一個**原始 `.bag`。

### Step 1 — `bag_to_data.py`：原始 bag → `img/`、`seg/`、`odom.csv`

直接離線讀取原始 bag（不需 roscore / play / record），對每張 RGB 影像做 SegFormer 分割 + resize，
輸出結構與舊 `extract_bag_data.py` 完全相同。

- **輸入**：`<root>/video1/*.bag ... <root>/videoN/*.bag`，每個 video 資料夾恰好一個 `.bag`。
- **輸出**（寫在每個 `videoN/` 內）：
  - `img/`：resize 後 224×224 RGB PNG，檔名為時間戳 `{timestamp_ns}.png`
  - `seg/`：4 類語意分割 224×224 彩色 PNG，檔名與對應的 `img/` **同一時間戳**
  - `odom.csv`：里程計（欄位 `timestep, position_x/y/z, orientation_x/y/z/w`）

```bash
source /opt/ros/noetic/setup.bash
python3 bag_to_data.py --root <root> --start 1 --end N [--overwrite]
```

- 參數：`--root`（預設 `.`）、`--start`（1）、`--end`（9）、`--overwrite`、
  `--rgb-topic`（預設 `/zed2i/zed_node/right_raw/image_raw_color`）、`--odom-topic`（預設 `/odom`）。
- 預設不覆寫已存在的檔案，加 `--overwrite` 才會重寫。

### Step 2 — `dataset/0624bkgd/resample.py`：0.5 秒重採樣 → `resample/`

> ⚠️ **請使用 `dataset/0624bkgd/resample.py`（新版）**，不要用頂層的 `resample.py`（舊版，會強制要求 `poseimu_zero.csv` 與 `depth/`）。詳見 [注意事項](#5-注意事項--常見陷阱)。

- 以 `HALF_SECOND_NS = 500_000_000`（0.5 秒）為步長，對時間軸逐點取最近鄰樣本。
- **輸入**：`<root>/videoN` 的 `img/`、`seg/`、`odom.csv`（無深度，用 `--no-depth`）。
- **輸出**：`<root>/resample/videoN/` 內含
  - `img/`、`seg/`（複製選中的 PNG，沿用原始時間戳檔名）
  - `odom.csv`（重採樣後）
  - `resample_index.csv`（對照表：`target/img/seg/depth/odom` 時間戳）

```bash
python3 dataset/0624bkgd/resample.py \
    --root <root> --output resample --start 1 --end N --no-depth [--overwrite]
```

- 輸出根資料夾預設名為 `resample`；相對路徑會接在 `<root>` 底下（即 `<root>/resample`）。

### Step 3 — `convert_cls4png_to_npy.py`：語意 PNG → class-id `.npy`

把 4 類彩色語意 PNG 依調色盤轉成 `uint8` 的 class-id `.npy`，**就地**輸出在與 PNG 相同的資料夾（[convert_cls4png_to_npy.py:72](convert_cls4png_to_npy.py#L72)）。此步是 Step 4 推論的**必要前置**（載入器只讀 `seg/{ts}.npy`）。

```bash
# 注意：--root 必須指向 seg 資料夾。一次處理所有 video 的 seg：
for d in <root>/resample/video*/seg; do
    python3 convert_cls4png_to_npy.py --root "$d" --force
done
```

- convert 以 `rglob("*.png")` 遞迴尋找 PNG 且**無檔名過濾**（[convert_cls4png_to_npy.py:65](convert_cls4png_to_npy.py#L65)），因此 `--root` 必須指到**只含語意圖的 `seg/`**，**不可**指向同時含 `img/` 相片的 `resample/videoN`（strict 模式會因相片顏色不在 `PALETTE4` 而報錯）。
- 未知顏色預設會**報錯（strict）**；若確定顏色略有偏差，可加 `--non_strict` 改用最近鄰對應。
- 已存在的 `.npy` 預設會跳過，加 `--force` 可覆蓋。

### Step 4 — `park_L2_ASAP.py`：路徑推論

- **環境**：`conda activate stp3_env`。
- **輸入** `--dataroot` 指向某個 `resample/videoN`；載入器會讀取（[NuscenesData_0624_ASAP.py:52-53](stp3/data_0512_graduate/NuscenesData_0624_ASAP.py#L52-L53)、[:501-505](stp3/data_0512_graduate/NuscenesData_0624_ASAP.py#L501-L505)）：
  - `resample_index.csv`、`odom.csv`
  - `img/{img_timestep}.png`、`seg/{seg_timestep}.npy`
  - 深度目前**停用**（`cfg.USE_DEPTH = False`，[park_L2_ASAP.py:627](park_L2_ASAP.py#L627)），故不需要 `depth_infer/`。
- **checkpoint**：`--checkpoint`（預設 `last.ckpt`）；本機請用 `model/best-box-col-*.ckpt`。
  > ⚠️ **`checkpoint/last.ckpt` 不是完整模型**：只有 10 個 tensor（純 AD-MLP planner head，3.4MB），
  > **不含 `model.vlm.*` 視覺權重**，無法推論。完整模型是 `model/best-box-col-*.ckpt`（646 個 tensor）。

```bash
conda activate stp3_env
python park_L2_ASAP.py \
    --checkpoint "model/best-box-col-epoch=24-epoch_val_plan_obj_box_col=0.0054.ckpt" \
    --dataroot dataset/0624bkgd/resample/video1
```

- **輸出**：每次執行都集中在 `inference/imgs/<MMDDHHMMSS>/`（每次執行一個時間戳資料夾，不覆蓋前次，[park_L2_ASAP.py:40](park_L2_ASAP.py#L40)、[:564](park_L2_ASAP.py#L564)），內含：
  - `trajectories.csv`、`l2_errors.csv`、`l2_error_summary.csv`（[park_L2_ASAP.py:581-587](park_L2_ASAP.py#L581-L587)）
  - `inference_plots/` 內的推論視覺化 PNG（[park_L2_ASAP.py:565-566](park_L2_ASAP.py#L565-L566)）

---

## 4. 檔案路徑總覽

```
senpai/
├── bag_to_data.py             # Step 1：原始 bag → img/ seg/ + odom.csv（推薦）
├── seg_real_time.py           # 備選（即時）：語意分割 ROS 節點，被 bag_to_data 重用其函式
├── extract_bag_data.py        # 備選（即時）：bag → PNG + odom.csv，被 bag_to_data 重用其函式
├── resample.py                # （舊版，勿用；請改用 dataset/0624bkgd/resample.py）
├── convert_cls4png_to_npy.py  # Step 3：seg PNG → class-id .npy
├── park_L2_ASAP.py            # Step 4：路徑推論主程式
├── environment.yml            # conda 環境定義（name: stp3_env）
├── checkpoint/
│   └── last.ckpt              # ⚠️ 非主權重：純 AD-MLP baseline（10 tensor）
│                              #    被完整模型當凍結 coarse baseline 載入，勿刪
├── model/
│   └── best-box-col-*.ckpt    # ★ 主推論權重（646 tensor，含 model.vlm.*）
├── stp3/                      # ST-P3 模型與資料集程式
│   ├── config.py              # 設定入口 get_cfg
│   └── data_0512_graduate/
│       └── NuscenesData_0624_ASAP.py   # 推論資料集載入器
├── dataset/
│   └── 0624bkgd/
│       ├── resample.py        # Step 2：0.5 秒重採樣（★新版，請用這支）
│       └── resample/video1/   # 範例：已重採樣好的一段資料
└── inference/                 # Step 4 輸出根目錄
    └── imgs/<時間戳>/         # 每次執行一個資料夾，內含：
        ├── trajectories.csv   #   輸入/GT/預測軌跡
        ├── l2_errors.csv      #   逐點 L2 誤差
        ├── l2_error_summary.csv #  L2 誤差統計摘要
        └── inference_plots/   #   推論視覺化 PNG
```

---

## 5. 端到端範例

以 `dataset/0624bkgd` 為資料根目錄，處理 `video1`（每個 `videoN/` 內先放好一個原始 `.bag`）：

```bash
# Step 1（noetic python3）：原始 bag → img/ seg/ + odom.csv
source /opt/ros/noetic/setup.bash
python3 bag_to_data.py --root dataset/0624bkgd --start 1 --end 1

# Step 2（新版 resample）：0.5 秒重採樣
python3 dataset/0624bkgd/resample.py \
    --root dataset/0624bkgd --output resample --start 1 --end 1 --no-depth

# Step 3：seg PNG → .npy（--root 指向 seg 資料夾，勿指整個 resample）
for d in dataset/0624bkgd/resample/video*/seg; do
    python3 convert_cls4png_to_npy.py --root "$d" --force
done

# Step 4（stp3_env）：推論路徑
conda activate stp3_env
python park_L2_ASAP.py \
    --checkpoint "model/best-box-col-epoch=24-epoch_val_plan_obj_box_col=0.0054.ckpt" \
    --dataroot dataset/0624bkgd/resample/video1
```

---

## 附錄：即時推論（realtime/）

若要**即時**跑完整條鏈路（相機 → 分割 → 路徑），不需要上面的四個步驟：
[realtime/realtime_planner_node.py](realtime/realtime_planner_node.py) 是單一 ROS 節點，
訂閱相機與 `/odom`，內部跑 SegFormer + ST-P3，直接把預測路徑以 `nav_msgs/Path` 發到 `/senpai/path`。

```bash
source /opt/ros/noetic/setup.bash   # 注意：即時節點用 noetic python3，不是 stp3_env
python3 realtime/realtime_planner_node.py
```

詳見 [realtime/README.md](realtime/README.md)。

---

## 6. 注意事項 / 常見陷阱

- **外部預設路徑必須覆寫**：`convert_cls4png_to_npy.py` 的 `--root` 預設（[:53](convert_cls4png_to_npy.py#L53)）與 `park_L2_ASAP.py` 的 dataroot 預設（[:629](park_L2_ASAP.py#L629)）、範例 checkpoint（[:34](park_L2_ASAP.py#L34)）都是外部路徑 `/home/cyc/...`，在本機不存在，請一律用參數指定自己的路徑。
- **`checkpoint/last.ckpt` 不可當主權重**：它是純 AD-MLP baseline（只吃 21 維狀態、不看影像）的存檔，只有 10 個 tensor，缺全部 633 個 `model.vlm.*` 視覺權重 → strict 載入會直接 `RuntimeError: Missing key(s)`。請一律用 `model/best-box-col-*.ckpt`。**但不要刪除它** —— 完整模型會把它當凍結的 AD-MLP coarse baseline 載入（`codex_pure_ASAP.py:16` 寫死路徑），缺檔會無法建構模型。
- **convert 的 `--root` 只能指向 `seg/`**：convert 遞迴抓所有 `*.png` 且無檔名過濾，指向整個 `resample/videoN`（含 `img/` 相片）會在 strict 模式報未知色錯誤。
- **resample 用新版**：請用 `dataset/0624bkgd/resample.py`（無 poseimu、支援 `--no-depth`）。頂層 `resample.py` 為舊版，會強制要求每個 video 內有 `poseimu_zero.csv` 與 `depth/`，缺檔會直接中斷。
- **深度全程停用**：原始 ZED bag 通常無深度 topic；`bag_to_data.py` 不輸出深度、Step 2 加 `--no-depth`、Step 4 推論設定 `USE_DEPTH = False`，一路一致。
- **環境分離**：Step 1–3 用 ROS noetic 的 `python3`；Step 4 用 `stp3_env`。兩者是不同環境（py3.8 vs py3.9），勿混用。
- **調色盤一致性**：分割輸出顏色必須與 Step 3 的 `PALETTE4` 完全一致；`bag_to_data.py` 存檔時做 RGB→BGR 轉換即是為此，否則 Step 3 在 strict 模式下會因未知顏色報錯。

---

## 附錄：備選（即時）資料轉出流程

若需要以即時方式產生資料（例如邊跑相機邊處理），可用舊的四步流程取代 Step 1。
`seg_real_time.py` 是 ROS 節點，訂閱即時相機 topic，發佈 `/seg_cls4_224`、`/image_224`、`/depth_224`，本身不寫檔；再用 `extract_bag_data.py` 從錄下的 bag 抽出 PNG。

```bash
# 終端 1：啟動 ROS master
roscore

# 終端 2：啟動語意分割節點
python3 seg_real_time.py

# 終端 3：錄製節點輸出（含里程計）成新 bag
rosbag record -O <root>/video1/processed.bag /image_224 /seg_cls4_224 /depth_224 /odom

# 終端 4：播放原始 bag（提供 /zed2i/... 與 /odom）
rosbag play 2026-06-23-18-23-14.bag

# 錄完後：從新 bag 抽出 PNG 與 odom.csv
python3 extract_bag_data.py --root <root> --start 1 --end N [--overwrite]
```

- `seg_real_time.py` 的輸入／輸出 topic 可用 ROS private param 覆寫：`~in_topic`、`~depth_topic`、`~out_topic`、`~depth_out_topic`（[seg_real_time.py:216-220](seg_real_time.py#L216-L220)）。
- `extract_bag_data.py` 讀取的 topic 預設為 `/image_224`、`/seg_cls4_224`、`/depth_224`、`/odom`（[extract_bag_data.py:12-17](extract_bag_data.py#L12-L17)），可用 `--img-topic` 等覆寫。
- 此流程之後接 Step 2 → Step 3 → Step 4，與主流程相同。
# stp3
