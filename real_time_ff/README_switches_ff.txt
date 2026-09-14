Realtime ASAP FF 功能開關與啟動範例
====================================

執行入口
--------

請執行：

  /home/cyc/ST-P3_please/real_time_ff/realtime_planner_node_ff.py

不要執行舊的 realtime_planner_node.py。預設模型位置是：

  /home/cyc/ST-P3_please/real_time_ff/model/last.ckpt

模型種類會依 checkpoint 內的 TAG 自動辨識：

  Planning_ASAP_ff         -> all_admlp
  Planning_ASAP_hybrid_ff  -> hybrid


一、模型與 ego 輸入開關
-----------------------

1. _checkpoint:=PATH

   指定 FF checkpoint。未設定時使用：

     real_time_ff/model/last.ckpt

2. _expected_model_variant:=auto|all_admlp|hybrid

   auto（預設）
     依 checkpoint TAG 自動辨識模型。

   all_admlp
     要求載入全 AD-MLP 模型。如果 checkpoint 實際不是此模型，程式會停止，
     可避免拿錯參數檔。

   hybrid
     要求載入 hybrid 模型。FORWARD 使用定速/odom 外推 coarse，
     LEFT、RIGHT 使用 AD-MLP coarse。

   這個參數只是「檢查開關」，不會把一個 checkpoint 強制轉成另一種模型。

3. _ego_input_mode:=auto|real_odom|fixed_speed

   auto（預設）
     若舊參數 _fixed_speed 大於 0，使用 fixed_speed；否則使用 real_odom。
     為了實驗結果明確，建議直接指定 real_odom 或 fixed_speed。

   real_odom
     模型的 future_egomotion、ego_history_egomotion、admlp_input 都由實際
     odom 建立。

   fixed_speed
     上述三種 ego 輸入全部改成定速直線假資料，速度由
     _fixed_speed_mps 指定。

     注意：即使使用 fixed_speed，程式仍需要接收真實 /odom，因為真實 odom
     仍用於發布 global path、MPC array，以及推論圖中的實際行駛 GT。

4. _fixed_speed_mps:=FLOAT

   fixed_speed 模式下的假定速度，單位 m/s，必須大於 0。
   例如：_fixed_speed_mps:=1.0

5. _fixed_speed:=FLOAT

   舊版相容參數。只在 _ego_input_mode:=auto 時用來決定是否進入固定速度模式。
   新實驗建議不要使用，改用明確的 _ego_input_mode 與 _fixed_speed_mps。


二、影像、深度與運算開關
------------------------

1. _use_depth:=true|false

   true（預設）
     執行 Depth-Anything-V2，將相對 inverse depth 輸入 FF 模型。

   false
     不執行 depth model，但 FF 模型仍會收到全零 depth map。
     FF checkpoint 訓練時使用真實相對深度，因此不建議關閉。

2. _da_v2_repo:=PATH

   Depth-Anything-V2 程式目錄。預設：

     real_time_ff/third_party/Depth-Anything-V2

3. _da_v2_ckpt:=PATH

   Depth-Anything-V2 權重。預設：

     real_time_ff/model/depth_anything_v2_vitl.pth

4. _device:=cuda|cpu

   模型執行裝置。CUDA 可用時預設為 cuda，否則為 cpu。

5. _use_fp16:=true|false

   true（預設）
     FF planner 在 CUDA 上使用 FP16 autocast，以降低推論時間與顯存。

   false
     FF planner 使用 FP32。

   Depth-Anything-V2 仍依目前程式設計使用 FP32，不受此開關影響。

6. _sample_interval:=0.5

   每次取樣間隔，單位秒。模型目前依 0.5 秒訓練，部署時應維持 0.5；
   若與 checkpoint 設定不同，FF 節點會停止並提示錯誤。


三、推論圖開關
--------------

1. _save_plots:=true|false

   true（預設）
     每次有效推論都排入存圖佇列；約 3 秒後取得真實行駛 GT 才寫入 PNG。

   false
     完全不建立推論圖，可降低 CPU、I/O 與畫圖時間。

   圖片預設輸出至：

     /home/cyc/ST-P3_please/realtime/inference/
       MM_DD_HH_MM_SS/inference_plots/

2. _plot_seg:=true|false

   控制存圖中是否加入 semantic panel。只在 _save_plots:=true 時有效。
   FF 版本顯示的是與訓練資料一致的正確 PALETTE4。

3. _plot_depth:=true|false

   控制存圖中是否加入 Depth-Anything-V2 可視化 panel。
   只在 _save_plots:=true 且 _use_depth:=true 時有效。

   這只影響圖片內容，不影響餵給模型的 depth。

推論圖顏色：

  綠色 = ego 歷史輸入
  藍色 = 推論後實際由 odom 量到的行駛軌跡 GT
  紅色 = 模型最後輸出的預測軌跡


四、ROS topic 與輸出設定
-----------------------

  _in_topic:=/zed2i/zed_node/rgb_raw/image_raw_color
      RGB 相機輸入。

  _odom_topic:=/odom
      真實 odometry 輸入。

  _command_topic:=/senpai/command
      指令輸入，內容使用 FORWARD、LEFT 或 RIGHT。

  _path_topic:=/senpai/path
      local frame Path 輸出。

  _path_global_topic:=/senpai/path_global
      odom/global frame Path 輸出。

  _array_topic:=/senpai/array_topic
      給既有 MPC 的全域軌跡，格式為 [x0,y0,x1,y1,...]。

  _seg_topic:=/senpai/seg_cls4_224
      四類 semantic 可視化輸出。

  _frame_id:=base_link
      local Path 的 frame_id。

  _plot_mode:=realtime  # 立即存圖、不畫 GT、側向固定 ±1 m
  _plot_mode:=with_gt   # 約 3 秒後存圖，補畫藍色 GT 與 L2
  _plot_mode:=off       # 完全不建立、不儲存圖片


五、完整範例
------------

範例 1：全 AD-MLP 模型 + 正常 odom 輸入

  source /opt/ros/noetic/setup.bash
  /home/cyc/miniconda3/envs/stp3_env/bin/python \
    /home/cyc/ST-P3_please/real_time_ff/realtime_planner_node_ff.py \
    _checkpoint:=/path/to/all_admlp_ff.ckpt \
    _expected_model_variant:=all_admlp \
    _ego_input_mode:=real_odom \
    _use_depth:=true \
    _use_fp16:=true \
    _save_plots:=true \
    _plot_seg:=true \
    _plot_depth:=true

此設定下，FORWARD、LEFT、RIGHT 的 coarse trajectory 全部由 AD-MLP 產生；
模型使用實際 odom 建立所有 ego motion/state 輸入。


範例 2：Hybrid 模型 + 固定 1 m/s 往前輸入

  source /opt/ros/noetic/setup.bash
  /home/cyc/miniconda3/envs/stp3_env/bin/python \
    /home/cyc/ST-P3_please/real_time_ff/realtime_planner_node_ff.py \
    _checkpoint:=/path/to/hybrid_ff.ckpt \
    _expected_model_variant:=hybrid \
    _ego_input_mode:=fixed_speed \
    _fixed_speed_mps:=1.0 \
    _use_depth:=true \
    _use_fp16:=true \
    _save_plots:=true \
    _plot_seg:=true \
    _plot_depth:=true

此設定下：

  FORWARD     -> 使用固定 1 m/s 的 ego motion 往前外推 coarse
  LEFT/RIGHT  -> 使用 AD-MLP coarse，但 AD-MLP 收到的是固定 1 m/s 直線 ego 輸入

真實 odom 仍會被訂閱並用於 global path、MPC array 與推論圖 GT，不會因為
fixed_speed 模式而停用。


六、常用精簡設定
----------------

若只想即時控制、不需要 PNG，可在任一範例加入：

  _save_plots:=false

若 checkpoint 固定放在預設位置：

  /home/cyc/ST-P3_please/real_time_ff/model/last.ckpt

則可以省略 _checkpoint。若信任 checkpoint TAG 自動判斷，也可以省略
_expected_model_variant；啟動 log 會顯示實際辨識到的 all_admlp 或 hybrid。


0805
cd ~/campus_ws/path_inference/real_time_ff
python realtime_planner_node_ff.py     _checkpoint:=/home/cyc/campus_ws/path_inference/real_time_ff/model/0804_admlp/last.ckpt     _expected_model_variant:=all_admlp     _ego_input_mode:=real_odom     _use_depth:=true     _use_fp16:=true     _save_plots:=true     _plot_seg:=true     _plot_depth:=true

python realtime_planner_node_ff.py     _checkpoint:=/home/cyc/campus_ws/path_inference/real_time_ff/model/0805_hybrid/last.ckpt     _expected_model_variant:=hybrid     _ego_input_mode:=fixed_speed     _fixed_speed_mps:=1.0     _use_depth:=true     _use_fp16:=true     _save_plots:=true     _plot_seg:=true     _plot_depth:=true _plot_mode:=realtime




七、語義分割後端（僅 realtime_planner_node_ff_VIO_VLM.py）
----------------------------------------------------------

_segmentation_backend:=NAME  可選 segformer（預設）、yolo26_sem、twinlitenet。
三者都輸出相同格式的 (224,224) PALETTE4 class id，可直接互相對照比較。

  segformer     SegFormer-B2 Cityscapes 19 類合併成 4 類（預設，行為未改動）
  yolo26_sem    YOLO26 semantic，同樣 19 類轉 4 類，需要 ultralytics
  twinlitenet   TwinLiteNet 可行駛區域 + 木棧道紋理濾除

twinlitenet 是二元的：可行駛 -> class 0 (road)，其餘 -> class 3 (static)，
class 1/2 (person/movable) 恆為空。FF 模型的 off-road cost 只讀 BEV 的 road
通道，所以這是有意的取捨。

木棧道濾除沿用 TwinLiteNet/test_video_filtered.py 的函式（直接 import，不是
複製）。差別只有時序投票：離線腳本輸出 5 幀視窗的中心幀，會慢 2 幀；這裡改成
只看已收到的幀做因果投票，所以沒有延遲。為了讓投票視窗維持在合理的時間跨度，
TwinLiteNet 是在 cb_image 以相機幀率執行，而不是每 0.5 s 規劃時才跑一次。

  _twinlite_weights:=PATH          預設 TwinLiteNet/pretrained/best.pth
  _twinlite_wood_filter:=BOOL      預設 true；false 就是純 DA，不做木紋濾除
  _twinlite_vote_window:=N         預設 5，對應離線的 WIN
  _twinlite_vote_min:=N            預設 3，對應離線的 VOTE
  _twinlite_largest_component:=BOOL 預設 true，只保留連到畫面底部的最大區塊
  _twinlite_min_keep_ratio:=R      預設 0.25，木紋濾除的誤判保護
  _twinlite_prepass_max_hz:=HZ     預設 15.0，相機幀率 pre-pass 的節流上限

_twinlite_min_keep_ratio 是保護機制：wood_mask 用水平梯度比值判斷木紋，遠處的
草皮有時也會通過。若濾除後的面積掉到「沒做木紋濾除的結果」的這個比例以下，就
判定濾除誤判，改用未濾除的遮罩並印出 warning。在 bkgd_right_raw.mp4 全片以
15 Hz 實測，真正的木棧道濾除最低只掉到 0.69，所以 0.25 不會誤觸。

範例：

python realtime_planner_node_ff_VIO_VLM.py _checkpoint:=/home/cyc/campus_ws/path_inference/real_time_ff/model/0805_hybrid/best-l2-epoch=28-epoch_val_plan_L2=1.2814.ckpt     _expected_model_variant:=hybrid     _ego_input_mode:=fixed_speed     _fixed_speed_mps:=1.0     _segmentation_backend:=twinlitenet     _use_depth:=true     _use_fp16:=true     _save_plots:=true     _plot_seg:=true     _plot_depth:=true _plot_mode:=realtime


八、離線 bag 測試（offline_bag_runner_ff.py）
---------------------------------------------

用 rosbag 的 Python API 直接把 bag 灌進 realtime_planner_node_ff.py，不需要
roscore，也不需要 rosbag play。

  /home/cyc/anaconda3/envs/stp3_ros/bin/python offline_bag_runner_ff.py \
      --bag PATH.bag  _name:=value ...

_name:=value 參數與實車指令完全相同，額外的旗標：

  --bag PATH        要重播的 bag（必填）
  --start SEC       跳過 bag 開頭幾秒
  --max-samples N   規劃 N 個取樣點後停止（0 = 整個 bag）
  --quiet           關掉 node 的 INFO log

與實車的兩個刻意差異：

1. 不丟幀。實車上 cb_image 在前一次規劃還沒跑完時會直接 return；離線每個
   0.5 s 取樣點都會完整推論，所以同一顆 bag 兩次跑的取樣幀完全一致，可以拿
   不同 segmentation backend 直接對照（輸出的 PNG 檔名相同）。
2. 不發布 topic。Publisher 是 no-op，但 build_path() 之類仍然照跑，所以那條
   路徑上的 bug 一樣會炸出來。

注意 bag 裡的 topic 名稱通常要指定，例如 bkgd0623 那顆是 right_raw：

  _in_topic:=/zed2i/zed_node/right_raw/image_raw_color

realtime_planner_node_ff.py 只需要 /odom，不需要 VIO；
realtime_planner_node_ff_VIO_VLM.py 需要 /ov_msckf/poseimu，沒有那個 topic 的
bag 不能直接餵給它。
