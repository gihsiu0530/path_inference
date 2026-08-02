# 即時路徑推論（Realtime Trajectory Inference）

把原本的離線四步流程（`bag_to_data.py` → `resample.py` → `convert_cls4png_to_npy.py` → `park_L2_ASAP.py`）
壓成**單一 ROS 節點**，訂閱相機與里程計，直接把 ST-P3 預測的未來路徑以 topic 發出。

```
                              鍵盤 ─→ /senpai/command ┐
                                                      ▼
相機 + /odom ─→ SegFormer-B2（4 類分割）──┬─→ ST-P3 ─→ nav_msgs/Path ─→ 即時視窗
             └→ Depth-Anything-V2（相對深度）┘
```

三支節點：
| 檔案 | 角色 |
|---|---|
| [realtime_planner_node.py](realtime_planner_node.py) | 主推論節點：相機 + /odom → 分割 → ST-P3 → `/senpai/path` |
| [keyboard_command.py](keyboard_command.py) | 鍵盤即時操控，發 `/senpai/command` |
| [visualize.py](visualize.py) | matplotlib 即時視窗：機器人座標、已走軌跡、預測軌跡 |

---

## 1. 執行環境

**先 source noetic，再用 conda `stp3_ros`**：

```bash
source /opt/ros/noetic/setup.bash
conda activate stp3_ros
```

`stp3_ros` 已同時具備 `rospy`、`cv_bridge`、`torch 2.2.2+cu121`、`transformers`、`pytorch_lightning`、
`pandas`、`pyquaternion`、`fvcore`、`yacs`，CUDA 可用；實測可完整跑起本節點。

> ⚠️ **不要用系統的 noetic `python3`**：它缺 `pyquaternion`、`pytorch_lightning`、`fvcore`，
> 節點在 import 階段就會 `ModuleNotFoundError`。
> `source /opt/ros/noetic/setup.bash` 仍然必要（提供 ROS 環境變數與 `rosbag`/`roscore`），
> 但實際執行請用 `stp3_ros` 的直譯器。

> ⚠️ **不要用 `stp3_env`**：該環境沒有 `rospy`，無法執行本節點；它是另一台機器的離線環境定義。

---

## 2. Checkpoint（重要）

本節點預設載入 `model/best-box-col-epoch=24-epoch_val_plan_obj_box_col=0.0054.ckpt`。

> ⚠️ **不可拿 `checkpoint/last.ckpt` 當 `~checkpoint`**：它是**另一個模型**（純 AD-MLP baseline）的存檔，
> 只有 10 個 tensor（3.4MB）：一個 3 層 MLP 的 `plan_head`（6 個）+ 2 個 loss 權重 + 2 個 metric buffer。
> 那個 baseline 只吃 21 維狀態向量、**完全不看影像**。
>
> 即時節點要的 `VLM_STP3_Gen` 有 9,083 萬參數，光 `model.vlm.*`（視覺編碼器 + 軌跡解碼器）就 **633 個 tensor**，
> 而 `last.ckpt` 提供 **0 個**。實際會在 strict 載入時炸掉：
>
> ```
> RuntimeError: Missing key(s) in state_dict: "model.dx", "model.bx",
>   "model.fake_cam_front", "model.vlm.time_queries", ... （600+ 個）
> ```
>
> 若硬改 `strict=False`，`vlm` 會維持隨機初始化 → 輸出的是雜訊路徑。
>
> **但此檔不要刪**：完整模型建構時會把它當作**凍結的 AD-MLP coarse baseline** 載入
> （`codex_pure_ASAP.py:15-18` 的 `DEFAULT_ADMLP_BASELINE_CKPT`，以 repo 根目錄組出 `checkpoint/last.ckpt`），
> 它的 6 個 `plan_head` 張量正好對應完整模型的 `model.admlp_baseline.*`。缺檔會導致完整模型無法建構。

---

## 3. Topic 介面

| 方向 | 參數 | 預設 Topic | 型別 |
|---|---|---|---|
| 訂閱 | `~in_topic` | `/zed2i/zed_node/rgb_raw/image_raw_color` | `sensor_msgs/Image` |
| 訂閱 | `~odom_topic` | `/odom` | `nav_msgs/Odometry` |
| 訂閱 | `~command_topic` | `/senpai/command` | `std_msgs/String` |
| 發布 | `~path_topic` | `/senpai/path` | `nav_msgs/Path`（base_link 局部座標） |
| 發布 | `~path_global_topic` | `/senpai/path_global` | `nav_msgs/Path`（odom 全域座標） |
| 發布 | `~array_topic` | `array_topic` | `std_msgs/Float64MultiArray`（給 MPC 控制器） |
| 發布 | `~seg_topic` | `/senpai/seg_cls4_224` | `sensor_msgs/Image`（除錯用） |

其他參數：`~checkpoint`、`~frame_id`（預設 `base_link`）、`~sample_interval`（預設 `0.5`）、`~device`、`~use_fp16`、`~save_plots`（預設 `true`）、`~plot_seg`（預設 `true`，推論圖是否附上語意分割面板）、`~plot_depth`（預設 `true`，推論圖是否附上深度面板）、`~fixed_speed`（預設 `0.0` = 關閉，見下）、`~use_depth`（預設 **`true`**，見下）、`~da_v2_repo`、`~da_v2_ckpt`。

### 深度輸入（`~use_depth`，預設**開啟**）

checkpoint **是用真實深度訓練的**，所以這個預設是 `true`，不要隨手關掉。

深度來源是節點內即時跑的 **Depth-Anything-V2 (vitl)**，**不是 ZED 的 `depth_registered`**：

| | 訓練/推論用的 | ZED 相機的 |
|---|---|---|
| 內容 | 相對**逆**深度（disparity），**大值 = 近** | 公制深度，uint16，單位 mm |
| 尺度 | 每幀自己一個尺度（實測連續幀間漂移約 2%） | 絕對公制 |

兩者連大小方向都相反，把 ZED 深度餵進去會比餵零深度更糟。

**管線**（每個 0.5 s 週期，接在既有的 `rgb_224` 之後）：

```
rgb_224 (224,224,3) uint8 ─→ /255（無 ImageNet 正規化）─→ DA-V2 vitl
  ─→ relu 輸出 (224,224) ─→ fp16 往返 ─→ ring buffer ─→ batch['depth_224_seq'] (1,3,224,224)
```

這**逐位元重現**離線 `infer_depth_da_v2.py` 的 **batch 分支**（`model(x)`）。該腳本有兩條數值完全不同的路徑，走哪條取決於輸入尺寸：

| 分支 | 觸發條件 | 前處理 |
|---|---|---|
| `model(x)`（**本節點用這條**） | 224×224（= 16×14，DINOv2 patch 整除） | 只有 `/255`，**沒有** ImageNet mean/std |
| `model.infer_image(im)` | 尺寸不被 14 整除（如 720×1280）→ 例外後 fallback | resize 到 518 + ImageNet 正規化 |

離線資料集是拿 224×224 的 crop 去產生的，所以是第一條。**不要**改用 `infer_image`。

> ⚠️ **不要對深度做任何 normalize / rescale**：DA-V2 輸出實測落在 14~370，而模型端
> `codex_pure_ASAP._depth()` 是 `clamp(0, 80) / 80` —— 約 **73% 的像素直接飽和成 1.0**。
> 這看起來很浪費，但 checkpoint 就是在這個分布上訓練的，「把範圍用好」等於把輸入推離
> 訓練分布，只會更糟。

> ⚠️ **不要換成 vits/vitb**：整個輸出分布會位移，checkpoint 沒看過。

**效能**（RTX 4070 SUPER 實測，224 輸入）：fp32 **17.8 ms**/幀、峰值顯存 1.4 GB。
節點刻意用 fp32：autocast fp16 雖然快到 13.4 ms，但會多佔 700 MB（cast buffer），
而 17.8 ms 只佔 0.5 s 週期的 3.6%，換不到值得的東西。

**檔案位置**：套件在 `third_party/Depth-Anything-V2/`、權重在 `model/depth_anything_v2_vitl.pth`
（1.3 GB）。兩者都可用 `~da_v2_repo` / `~da_v2_ckpt` 覆寫。

> 這兩份是從 `/home/cyc/dataset/1222_obstacle/output_dataset/tools/Depth-Anything-V2/`
> **複製**進來的，不是引用 —— `/home/cyc/dataset/` 底下的資料夾會消失（開發期間
> `0504_what_up`、`0507_vlm`、`1208_school_stp3` 都不見過），預設路徑指過去會讓節點某天突然起不來。
> `path_inference/` 整個在 `.gitignore` 裡，所以這 1.3 GB 不會進版控。

**關掉會怎樣**：`_use_depth:=false` 時 batch 不放 `depth_224_seq`，
`codex_pure_ASAP.forward` 會補一組**零深度**（fusion 層的深度通道是固定的、拿不掉）。
節點啟動時會 `logwarn` 提醒這是訓練/推論不一致。實測同一幀開/關的預測差異可達 **0.62 m**。

**載入失敗就報錯**：`~use_depth` 開著但套件或權重找不到時，節點直接 raise 起不來，
不會靜默退回零深度 —— 「看起來正常在跑但其實餵零深度」比起不來難發現得多。

### 固定速度輸入（`~fixed_speed`，預設關閉）

模型在「車子有速度」時推論出的路徑比較符合預期，但靜止或慢速起步時，
由真實 odom 擬合出的速度接近 0，軌跡會縮在原點附近。

設 `_fixed_speed:=1.0` 後，餵給模型的 `admlp_input` 改用**合成的等速直線歷史**：
過去 4 點固定為 `(0,-2,0) (0,-1.5,0) (0,-1,0) (0,-0.5,0)`（模型座標 `x_left, y_front, yaw`，
0.5 s 節拍），速度固定為 `(0, 1.0, 0)`，加速度全 0。等於告訴模型「我正以 1 m/s 直行」。

```bash
python3 realtime/realtime_planner_node.py _fixed_speed:=1.0
```

> 只影響 `admlp_input` 這 15 維。`future_egomotion`（視覺時序分支）、發布的三個路徑
> topic、以及推論可視化圖上的 input 軌跡**全部仍使用真實 odom**，
> 所以圖上可以直接對照「模型以為的速度」與「實際走的路」。
> 值 `<= 0` 視同關閉（負值會 warn 後當成 0）。

### 給 MPC 控制器的 `array_topic`

`~array_topic`（預設 `array_topic`）發的是**與 `/senpai/path_global` 同一組全域點**，
只是攤平成 `std_msgs/Float64MultiArray`，內容為 `[x0,y0,x1,y1,...]`（起點 + 未來 6 點，共 7 點 = 14 個值）。
這正是 `mpc_4state/local_path.cpp::callbackorgwp` 期望的格式，因此 MPC 控制鏈
（`local_path` → `mpc`）可**零修改**直接吃即時推論路徑，取代原本 `global_path` 讀 CSV 的全域路線。
每次推論（約 2 Hz）整包覆蓋前一次（rolling）。

> ⚠️ **不要同時啟動 `global_path`**：它也發 `array_topic`，兩者會互相蓋掉。用即時路徑時只跑本節點。
>
> ⚠️ **座標系一致性**：`array_topic` 的座標沿用 `~odom_topic` 訊息的 frame（此 bag 為 `map`）。
> 必須確保 **本節點與控制器 `local_path` 吃的是同一個 `/odom` 定位來源**，最近點搜尋才會對齊。

### 推論可視化圖（`~save_plots`，預設開啟）

**預設就會**每次推論存一張**離線同風格的 combo 圖**到
`realtime/inference/<MM_DD_HH_MM_SS>/inference_plots/`，檔名 `{序號:06d}_{時間戳ns}.png`。
預設版面共五格，由左至右：

| # | 面板 | 尺寸 | 內容 |
|---|------|------|------|
| 1 | 相機影像 | 224×224 | center-crop 後、真正餵進模型的那一格 RGB |
| 2 | `SEG PALETTE4` | 224×224 | 語意分割上色，與 `/senpai/seg_cls4_224` 及 `bag_to_data.py` 產的離線資料集**同一組顏色**（路面紫、人紅、可動藍、靜物灰），可直接對照 |
| 3 | `SEG model-input` | 224×224 | 同一張分割圖，但用 `SEG_PALETTE`（§8 那組刻意差一格的調色盤）上色，也就是**真正進到 checkpoint 的像素值**。路面在這格是**黑色**、靜物是綠色，這是正常的，不要去「修」 |
| 4 | `DEPTH DA-V2` | 224×224 | Depth-Anything-V2 的相對逆深度，**亮 = 近**。每幀 min-max 拉伸（與離線 `--save_vis` 相同），這只是**顯示用**，餵給模型的仍是未正規化的原始值 |
| 5 | 軌跡面板 | 512×512 | 🟢 過去 input + 🔵 GT + 🔴 預測 + L2 數字，與 `inference/imgs/…/inference_plots` 的圖一致 |

第 2~4 格都對應第 1 格那張影像，所以「軌跡預測很怪」時可以一眼分辨是**分割壞了**、
**深度壞了**還是**規劃壞了**。要減面板 / 不想存圖時：

```bash
python3 realtime/realtime_planner_node.py _save_plots:=false   # 完全不存圖
python3 realtime/realtime_planner_node.py _plot_seg:=false     # 拿掉兩格分割面板
python3 realtime/realtime_planner_node.py _plot_depth:=false   # 拿掉深度面板
```

（`~use_depth:=false` 時第 4 格自動消失，不需要另外設 `~plot_depth`。）

**GT 從何而來（圖會晚 3 秒落地）**：即時推論當下拿不到未來 GT，因此節點會把每次推論
**排入佇列**，等後續 6 個 0.5s 取樣（= 3 s）到齊後，用機器人**實際走過的路徑**當 GT
補畫並算 L2。所以圖比即時畫面**延遲約 3 秒**才出現，這是拿到真實 GT 的必要代價。
GT 的座標算法與離線 loader 的 `get_gt_trajectory` 完全相同（`(x_left, y_front, yaw)`，
開頭補 `(0,0,0)`），因此紅藍兩線可直接比對。

> ⚠️ **GT 的時間跨度會略大於 3 s**：GT 取樣點沿用 `cb_image` 的節拍閘門（`>= ~sample_interval`），
> 而相機幀率是離散的 —— 15 Hz 相機實測中位數落在 **0.533 s**（8 幀）而非 0.5 s，
> 6 個點就是 3.2 s。GT 因此比模型的 3 s 預測視野**多走約 6.7%**，會系統性地略微放大
> **縱向** L2。要精確評估請用離線 `park_L2_ASAP.py`（資料已重取樣到準確的 0.5 s）。

> ⚠️ **L2 的意義要看有沒有閉環**：若 MPC 正在**追蹤本節點發出的路徑**（`array_topic` 那條鏈），
> 機器人實際走的就會逼近預測本身，L2 會變成**自我實現的小數字**，不能當成離線那種
> 「預測 vs. 人類駕駛」的預測誤差來看。要評估真正的預測品質，請在**開環**下看
> （只跑本節點、由人操控或 bag 重播），或直接用離線 `park_L2_ASAP.py` 的 L2。

**尾端影像**：Ctrl-C 停止節點（或偵測到時鐘重啟）時，最後未滿 3 s 的 ≤6 次推論會用
**已收集到的部分 GT** 補畫出來（藍線較短），不會被丟掉；這幾張的 `L2 final=` 顯示的是
**現有最後一步**而非第 6 步。

**資料夾切換時機**：節點在**第一次寫圖**時建立第一個時間戳資料夾；之後**每次重播 bag**
（偵測到時間倒退＝時鐘重啟）會自動**開一個新的時間戳資料夾**，不需重啟節點。同一段連續
播放的所有圖都存在同一個資料夾（舊資料夾的尾端圖會在切換前補齊）。

### command（必要，且無法自動取得）

模型需要 `LEFT` / `FORWARD` / `RIGHT` 指令。
離線版是從**未來 GT 軌跡終點**反推的 —— 即時推論拿不到未來，因此必須由外部提供。
**建議用鍵盤節點即時操控**（見 [§5](#5-鍵盤即時操控-keyboard_commandpy)）：

```bash
python3 realtime/keyboard_command.py    # 方向鍵 ←/↑/→ = LEFT/FORWARD/RIGHT
```

或用單次指令手動測試：

```bash
rostopic pub /senpai/command std_msgs/String "data: 'LEFT'" -r 1
```

未收到任何指令時預設為 `FORWARD`。無法辨識的字串會被忽略並保留前一個指令。

> ℹ️ **橫向符號**：模型輸出軌跡的第 0 通道是橫向、且**右為正**，與 loader `gt_trajectory`
> 的 `x_left`（左為正）相反。節點在 `plan()` 統一翻一次符號（與離線的
> `park_L2_ASAP.py:742` 相同），之後三個發布 topic 與存圖都直接沿用，不再各自處理。
> 指令**原樣**送進模型，`/senpai/command` 的語意就是物理語意（送 `LEFT` 真的向左）。
>
> 舊版曾有 `~flip_command` 參數，在送進模型前把 `LEFT`↔`RIGHT` 對調，理由是
> 「checkpoint 的 command 通道是反的」。那是誤判 —— 當時的 A/B 實驗看的是缺少上述符號
> 翻轉的發布端，兩種解釋無法區分。用離線 `l2_errors.csv` 可判定模型沒有反：翻轉後
> 177 個轉彎樣本（`|gt_x| >= 1`）符號**全部**一致、`corr(gt_x, pred_x) = 0.88`，
> 不翻轉的話橫向誤差是 5.53 m。`dir_loss`（LEFT → `pred_last_x` 為負）在「右為正」
> 慣例下與 loader 標籤一致。該參數已移除。

### 輸出格式

同時發**兩個** `nav_msgs/Path`，各 **7 個點**（起點 + 未來 6 點），間隔 0.5 秒：

- **`/senpai/path`（`frame_id=base_link`，局部座標）**：起點固定為 `(0,0,0)`，
  座標遵循 ROS REP-103（x 前、y 左）—— 模型內部用的是 `(x_left, y_front)`，節點已轉換回來。
- **`/senpai/path_global`（全域座標）**：把同一組軌跡用**當前 `/odom` 姿態**
  （位置 + 朝向）轉到全域 —— 起點即 `/odom` 當前 `(x, y)`、朝向為機器人當前 heading，
  未來 6 點沿此姿態延伸，每點的 quaternion 為「機器人 yaw + 模型相對 yaw」（全域 heading）。
  `frame_id` **直接沿用 `/odom` 訊息的 `header.frame_id`**（取不到才 fallback `odom`）；
  例如 `dataset/0624bkgd` 的 bag 其 `/odom` frame 是 `map`，故此 topic 也會是 `map`。

兩者並存、內容一一對應；base_link 版行為不變，`visualize.py` 也仍讀 base_link 版自行轉全域。

---

## 4. 啟動（每個終端都要先 `source /opt/ros/noetic/setup.bash && conda activate stp3_ros`）

```bash
# 終端 1
roscore

# 終端 2：主推論節點
python3 realtime/realtime_planner_node.py

# 終端 3：鍵盤即時操控（見 §5）
python3 realtime/keyboard_command.py

# 終端 4：即時視窗（座標 + 已走軌跡 + 預測軌跡，見 §6）
python3 realtime/visualize.py

# 終端 5：實機相機，或用 bag 回放測試（注意 topic 名稱，見下方 ⚠️）
rosbag play /home/cyc/campus_ws/mpcdata/bkgd_20260623/2026-06-23-18-23-14.bag \
  /zed2i/zed_node/right_raw/image_raw_color:=/zed2i/zed_node/rgb_raw/image_raw_color
```

> ⚠️ **bag 的影像 topic 可能不是 `rgb_raw`**：`bkgd_20260623` 這支 bag 錄的是
> `/zed2i/zed_node/**right_raw**/image_raw_color`，而節點預設訂閱 `rgb_raw`。
> **不做 remap 的話節點收不到任何影像、完全不會推論也不出圖，且不會報錯**（只會一直
> `waiting for /odom` 之後就靜默）。先用 `rosbag info <bag>` 確認 topic 名稱，再用上面的
> remap（或改用 `_in_topic:=...` 啟動節點）。

首次啟動較慢：需下載 SegFormer 權重、載入 Depth-Anything-V2（1.3 GB）並建立 ST-P3 模型（約 1–3 分鐘）。

啟動時 log 會印出深度狀態，可用來確認沒被誤關：

```
[planner] loading Depth-Anything-V2 /home/cyc/campus_ws/path_inference/model/depth_anything_v2_vitl.pth
[planner] use_depth=true: Depth-Anything-V2 relative depth (vitl) -> depth_224_seq, as the checkpoint was trained
```

推論中的每行計時也會多一段 `depth=`：

```
[planner] seg=12.3 ms | depth=17.8 ms | plan=45.6 ms | total=75.7 ms | command=FORWARD
```

### 檢查

```bash
rostopic hz /senpai/path             # 約 2 Hz（0.5 秒節拍）
rostopic echo -n1 /senpai/path       # base_link 局部，起點 (0,0,0)
rostopic echo -n1 /senpai/path_global  # odom 全域，起點 = 當前 /odom (x,y)
```

RViz（可選）：
- 看**局部路徑**：Fixed Frame 設 `base_link`，Path display 指向 `/senpai/path`。
- 看**全域路徑**：Fixed Frame 設成 `/odom` 的 frame（`dataset/0624bkgd` 的 bag 是 `map`；
  用 `rostopic echo -n1 /senpai/path_global` 看 `frame_id` 確認），Path display 指向
  `/senpai/path_global`（起點會貼著機器人當前位置、沿朝向延伸）。

---

## 5. 鍵盤即時操控 [keyboard_command.py](keyboard_command.py)

在**自己的終端**執行（需要真正的 TTY），把按鍵即時發到 `/senpai/command`：

```bash
source /opt/ros/noetic/setup.bash && conda activate stp3_ros
python3 realtime/keyboard_command.py
```

| 按鍵 | 指令 |
|---|---|
| `←` / `a` | LEFT |
| `↑` / `w` / 空白 | FORWARD |
| `→` / `d` | RIGHT |
| `q` / `Ctrl-C` | 離開 |

**鎖存式**：按一次某方向就維持該指令，直到你按下另一個方向。發布用 latch，所以較晚啟動的推論節點也會收到最後一個指令。
（方向即物理方向，送 `←` 就是往左，指令原樣進模型，見 §3。）

---

## 6. 即時視窗 [visualize.py](visualize.py)

開一個 matplotlib 視窗（需要 `$DISPLAY`），在 **odom 全域座標**上即時畫出：

- 🔴 機器人當前位置與朝向（來自 `/odom`）
- 🔵 **已走軌跡**（累積 `/odom`）
- 🟠 **預測軌跡**（`/senpai/path`，已用當前 pose 從 base_link 轉到全域）
- 左上角文字：`x` / `y` / `yaw` / `cmd`

```bash
source /opt/ros/noetic/setup.bash && conda activate stp3_ros
python3 realtime/visualize.py
```

參數：`~view_span`（視窗半徑，預設 15 m）、`~min_step`（軌跡取點的最小位移，預設 0.05 m）、
`~history_len`（已走軌跡最多保留點數，預設 4000）。關掉視窗或 `Ctrl-C` 即結束。

> 若在遠端／無 `$DISPLAY` 環境，改用 RViz（§4 檢查）看 `/senpai/path`。

---

## 7. 在 RViz 看全域路徑 [senpai_rviz.launch](senpai_rviz.launch)

一鍵起 RViz 並預載設定，即時看 `/senpai/path_global`（全域 odom 座標的預測路徑）：

```bash
source /opt/ros/noetic/setup.bash
python3 realtime/realtime_planner_node.py       # 終端 1（會發 /senpai/path_global）
roslaunch realtime/senpai_rviz.launch           # 終端 2（起 static TF + RViz）
rosbag play dataset/0624bkgd/video1/2026-06-23-18-23-14.bag   # 終端 3
```

橘色路徑會隨每次推論更新（約 2 Hz），起點貼著機器人當前 map 位置、沿朝向延伸。

> ⚠️ **為何要 launch 而不是直接開 RViz**：此 bag **沒有任何 `/tf`**，而 RViz 的
> Fixed Frame 必須存在於 TF 樹中。直接把 Fixed Frame 設成 `map` 會報
> 「Fixed Frame [map] does not exist」而畫不出東西。launch 裡用一個
> `static_transform_publisher`（`map → map_root`）讓 `map` frame「存在」，
> 路徑仍以自身絕對 map 座標渲染，不受此靜態變換數值影響。

**換資料集**：`/senpai/path_global` 的 frame 是沿用 `/odom` 的 `header.frame_id`（此 bag 為
`map`）。若別的資料集 `/odom` frame 是 `odom`，用 `roslaunch realtime/senpai_rviz.launch frame:=odom`
並把 [senpai_path.rviz](senpai_path.rviz) 的 `Fixed Frame` 一併改成 `odom`。

**看不到路徑？** 多半是視角沒對到路徑座標（此 bag 路徑約在 `(37,-8) → (57,-17)`）。
用滑鼠拖曳平移、滾輪縮放即可；或調整 Views 面板的 `X`/`Y`（Focal Point）到該處。

---

## 8. 設計說明（維護時必讀）

### 0.5 秒節拍是硬性假設

模型訓練時的 `SAMPLE_INTERVAL = 0.5`。節點在相機回呼中以「距上次取樣 ≥0.5s」為條件取樣，
其餘影像**在跑分割前就丟棄**，確保送進模型的序列間隔與訓練一致。

### 暖機需要 5 個 pose，不是 3 個

- 影像緩衝 **3 筆**（`TIME_RECEPTIVE_FIELD=3`）
- 深度緩衝 **3 筆**（與影像同一節拍，`~use_depth` 開啟時才存在）
- Pose 緩衝 **5 筆**（`ADMLP_PAST_FRAMES=4` + t0 = 2.5 秒）

**pose 的歷史視野比影像長**，所以暖機以 pose 為準（約 2.5 秒）。全部湊滿前不推論。

### ⚠️ 調色盤錯位是刻意保留的

`SEG_PALETTE`（複製自 loader `NuscenesData_0624_ASAP.py:30-35`）是**直接用 seg id 索引**，
與 `convert_cls4png_to_npy.py` 的 `PALETTE4` **錯開一格**：

| seg id | PALETTE4 語意 | SEG_PALETTE 實際上色 |
|---|---|---|
| 0 | road | `(0,0,0)` 黑 |
| 1 | person | `(128,64,128)` |
| 2 | movable | `(220,20,60)` |
| 3 | static | `(0,142,0)` 綠 |

訓練走的就是這條路徑，**權重學到的就是這個錯位配色**。
節點必須產生 `PALETTE4` 語意的 seg id，再套 `SEG_PALETTE` 上色 —— **這不是 bug，不要「修正」**，
否則輸入分佈會偏離訓練分佈。

### 模型不需要的東西

- **相機內外參**：模型是純影像空間（`codex_pure_ASAP.py:622` 直接忽略）→ 一律 `torch.empty(0)`。
- **`gt_trajectory`**：`final_traj` 只源自 `self.vlm(...)`，GT 僅用於取 `device` 與算 loss
  （`codex_pure_ASAP.py:756-770`）→ 即時推論傳零張量，預測不受影響。
- **`future_egomotion[2]`**：離線版取自未來影格，但模型只讀 index 0、1
  （`codex_pure_ASAP.py:653-666`）→ 填零。
（**深度不在此列**：`~use_depth` 預設開啟，餵的是 Depth-Anything-V2 的相對深度，
見 §3。只有把它關掉時，模型 forward 才會自動補零深度。）

### 呼叫順序

必須**先 `forward` 再 `planning`** —— `planning` 依賴 forward 快取的 `_last_rgb_seq` 等，
未 forward 會 assert 失敗（`codex_pure_ASAP.py:757-759`）。節點直接重用
`park_L2_ASAP.py` 的 `_call_model_forward` / `_call_model_planning`，確保與離線語意一致。
