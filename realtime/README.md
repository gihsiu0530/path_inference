# 即時路徑推論（Realtime Trajectory Inference）

把原本的離線四步流程（`bag_to_data.py` → `resample.py` → `convert_cls4png_to_npy.py` → `park_L2_ASAP.py`）
壓成**單一 ROS 節點**，訂閱相機與里程計，直接把 ST-P3 預測的未來路徑以 topic 發出。

```
                              鍵盤 ─→ /senpai/command ┐
                                                      ▼
相機 + /odom ─→ SegFormer-B2（4 類分割）─→ ST-P3 ─→ nav_msgs/Path ─→ 即時視窗
```

三支節點：
| 檔案 | 角色 |
|---|---|
| [realtime_planner_node.py](realtime_planner_node.py) | 主推論節點：相機 + /odom → 分割 → ST-P3 → `/senpai/path` |
| [keyboard_command.py](keyboard_command.py) | 鍵盤即時操控，發 `/senpai/command` |
| [visualize.py](visualize.py) | matplotlib 即時視窗：機器人座標、已走軌跡、預測軌跡 |

---

## 1. 執行環境

**只用 ROS noetic 的 `python3`（3.8）**：

```bash
source /opt/ros/noetic/setup.bash
```

此環境已同時具備 `rospy`、`torch 2.4.1+cu121`、`transformers`、`pytorch_lightning`、`pandas`、`pyquaternion`，CUDA 可用。

> ⚠️ **不要用 `stp3_env`**：該環境沒有 `rospy`，也沒有 `transformers`，無法執行本節點。
> 離線流程的 Step 4 才需要 `stp3_env`；即時流程完全不需要。

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
> （`codex_pure_ASAP.py:16` 寫死絕對路徑 `/home/systemlab/senpai/checkpoint/last.ckpt`），
> 它的 6 個 `plan_head` 張量正好對應完整模型的 `model.admlp_baseline.*`。缺檔會導致完整模型無法建構。

---

## 3. Topic 介面

| 方向 | 參數 | 預設 Topic | 型別 |
|---|---|---|---|
| 訂閱 | `~in_topic` | `/zed2i/zed_node/right_raw/image_raw_color` | `sensor_msgs/Image` |
| 訂閱 | `~odom_topic` | `/odom` | `nav_msgs/Odometry` |
| 訂閱 | `~command_topic` | `/senpai/command` | `std_msgs/String` |
| 發布 | `~path_topic` | `/senpai/path` | `nav_msgs/Path`（base_link 局部座標） |
| 發布 | `~path_global_topic` | `/senpai/path_global` | `nav_msgs/Path`（odom 全域座標） |
| 發布 | `~seg_topic` | `/senpai/seg_cls4_224` | `sensor_msgs/Image`（除錯用） |

其他參數：`~checkpoint`、`~frame_id`（預設 `base_link`）、`~sample_interval`（預設 `0.5`）、`~device`、`~use_fp16`、`~save_plots`（預設 `false`）。

### 推論可視化圖（`~save_plots`）

設 `_save_plots:=true` 時，每次推論會存一張**離線同風格的 combo 圖**到
`realtime/inference/<MM_DD_HH_MM_SS>/inference_plots/`（每次跑一個新時間戳資料夾）：
左邊 224×224 相機影像、右邊軌跡面板（🟢 過去 input + 🔴 預測）。即時推論**沒有未來 GT**，
所以不會有藍色 GT 軌跡、也沒有 L2 數字，其餘與 `inference/imgs/…/inference_plots` 一致。

```bash
python3 realtime/realtime_planner_node.py _save_plots:=true
```

預設（不帶此參數）不產生任何檔案。

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

> ⚠️ **command 反向補償（`~flip_command`，預設 `true`）**：
> 此 checkpoint 的 command 通道是**反的** —— 直接餵 `LEFT` 會讓路徑往**右**偏 2–3m，反之亦然。
> （模型 `codex_pure_ASAP.py` 的 `dir_loss` 方向與 loader `:630` 的 `LEFT`/`RIGHT` 標籤相反；
> 已用「同一場景只改 command」的受控 A/B 實驗證實。e）
> 節點預設在送進模型前把 `LEFT`↔`RIGHT` 對調，讓 `/senpai/command` 的語意符合物理直覺（送 `LEFT` 真的向左）。
> 若你想餵原始標籤（例如日後修好模型），設 `_flip_command:=false`。
> `FORWARD` 不受影響。這是**模型層的既有 bug**，不是即時節點造成的。

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

## 4. 啟動（每個終端都要先 `source /opt/ros/noetic/setup.bash`）

```bash
# 終端 1
roscore

# 終端 2：主推論節點
python3 realtime/realtime_planner_node.py

# 終端 3：鍵盤即時操控（見 §5）
python3 realtime/keyboard_command.py

# 終端 4：即時視窗（座標 + 已走軌跡 + 預測軌跡，見 §6）
python3 realtime/visualize.py

# 終端 5：實機相機，或用 bag 回放測試
rosbag play dataset/0624bkgd/video1/2026-06-23-18-23-14.bag
```

首次啟動較慢：需下載 SegFormer 權重並建立 ST-P3 模型（約 1–3 分鐘）。

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
source /opt/ros/noetic/setup.bash
python3 realtime/keyboard_command.py
```

| 按鍵 | 指令 |
|---|---|
| `←` / `a` | LEFT |
| `↑` / `w` / 空白 | FORWARD |
| `→` / `d` | RIGHT |
| `q` / `Ctrl-C` | 離開 |

**鎖存式**：按一次某方向就維持該指令，直到你按下另一個方向。發布用 latch，所以較晚啟動的推論節點也會收到最後一個指令。
（提醒：方向的物理意義已由 planner 的 `~flip_command` 補償，送 `←` 真的往左，見 §3。）

---

## 6. 即時視窗 [visualize.py](visualize.py)

開一個 matplotlib 視窗（需要 `$DISPLAY`），在 **odom 全域座標**上即時畫出：

- 🔴 機器人當前位置與朝向（來自 `/odom`）
- 🔵 **已走軌跡**（累積 `/odom`）
- 🟠 **預測軌跡**（`/senpai/path`，已用當前 pose 從 base_link 轉到全域）
- 左上角文字：`x` / `y` / `yaw` / `cmd`

```bash
source /opt/ros/noetic/setup.bash
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
- Pose 緩衝 **5 筆**（`ADMLP_PAST_FRAMES=4` + t0 = 2.5 秒）

**pose 的歷史視野比影像長**，所以暖機以 pose 為準（約 2.5 秒）。兩者湊滿前不推論。

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
- **深度**：全程停用，模型 forward 會自動補零深度。

### 呼叫順序

必須**先 `forward` 再 `planning`** —— `planning` 依賴 forward 快取的 `_last_rgb_seq` 等，
未 forward 會 assert 失敗（`codex_pure_ASAP.py:757-759`）。節點直接重用
`park_L2_ASAP.py` 的 `_call_model_forward` / `_call_model_planning`，確保與離線語意一致。
