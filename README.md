# MARS - 多重目標自動辨識系統
**Multiple-target Automated Recognition System**

MARS 是一套具備高度可擴展性與工業級強健防呆機制的「端到端 (End-to-End) 全流程自動化多目標辨識系」。專為複雜、多物件場景設計，原生支援**單階層全景目標定位**與**雙階層級聯微觀特徵提取 (Two-Stage Cascade)** 架構。

本系統涵蓋從訓練時期「影片時序抽幀與 YUV-CLAHE 光學調整」、「主物件安全限幅裁切與次物件坐標投影」、「YOLO 模型擬合訓練」、「TensorRT FP16 靜態加速引擎編譯」，到辨識時期「影片時序串流與影格時間標記 (recg_ts)」、「動態領域管線調度」、「影像處理工具鏈 (幾何透視校正拉平 / HSV 色彩提取 / OCR 字元辨識)」以及「PostgreSQL 高效批次寫入 (Upsert) 與實體檔案分層歸檔」之全流程自動化。

---

## 🌟 核心特色 (Core Features)

1. **雙階層級聯協調架構 (Two-Stage Cascade Architecture)**
   - **Stage-1 (宏觀主階層)**：全景影像主物件定位與鎖定（e.g., 車輛 VEH），原生支援常規水平框 (HBB) 與旋轉邊界框 (OBB)。
   - **Stage-2 (微觀次階層)**：驅動 `StageCoordinator`，針對全景標註實施「安全限幅外擴裁切 (Safe Cropping)」，自動將次物件坐標投影換算至局部切片並將類別編號平移歸零，同步產出次階 `data.yaml` 與次領域物件生成溯源標記 (`sub_obj_lineage.json`)。

2. **影像處理與特徵提取工具鏈 (Image Processing & Feature Extraction Tools)**
   - **幾何透視校正 (`GeomWarper`)**：自動解析 OBB 頂點幾何順序，透過 OpenCV 透視變換矩陣將傾斜次目標（e.g., 車牌）拉平校正為標準矩形。
   - **物件邊界裁切 (`BodyCropper`)**：依主物件位置實施自適應邊界檢測與安全限幅裁切，防止切片尺寸過小或無效幾何區域導致程式例外中斷。
   - **色彩特徵提取 (`ColorExtractor`)**：核心區域取樣與 HSV 色彩空間轉換，自動排除背景雜訊並統計車體主要外觀顏色。
   - **光學字元辨識 (`CharReader`)**：首選預載之 `PaddleOCR` 引擎，並內建 `EasyOCR` 內部備援辨識機制；對接 `CHAR_RULES` 進行字串清洗、特徵屬性識別（e.g., 一般車、電動車、軍車）與白名單正則過濾。

3. **模型辨識效率最佳化與記憶體資源管理 (Inference Optimization & Memory Management)**
   - 全自動導出最佳原生權重並實施 **TensorRT FP16 半精度靜態量化**，大幅縮短辨識處理耗時。
   - 具備視覺模型加載機制 (`VisionModelPreloader`)，於辨識啟動時將主/次階層 YOLO 模型與重型影像處理工具 (e.g., OCR) 預先常駐於主記憶體映射表 (`yolo_pool` / `tool_pool`)，辨識期間無需重複讀取硬碟，大幅降低系統 I/O 負擔；任務結束後主動清空預載模型快取並觸發記憶體資源回收機制，徹底防範記憶體資源溢出。

4. **工業級資料庫部署與衝突覆寫 (Robust Database Orchestration)**
   - 整合 `psycopg2`，動態解析 Schema 與 Table 定義並自動建置資料表。
   - 採用 `INSERT ... ON CONFLICT DO UPDATE` (Upsert) 高效批次寫入機制，搭配變異偵測 (`IS DISTINCT FROM`)，確保高併發時序資料零遺漏且具備嚴格冪等性。

5. **PyQt6 圖形化操作介面 (GUI Management Platform)**
   - 構建整合式主視窗核心框架 (`MainWindow`)，收納系統總覽 (`DashboardView`)、模型訓練 (`TrainView`)、模型辨識 (`InferView`) 與資料審計 (`DatabaseView`) 多分頁視圖。
   - 透過後端管線橋接引擎 (`PipelineWorker`) 與日誌轉發工具 (`attach_gui_log_handler`)，以非同步獨立執行緒調度耗時任務，搭配操作按鈕狀態互鎖機制，保障主介面流暢不凍結並提供終端日誌即時滾動渲染。

---

## 🗂️ 專案目錄結構 (Project Structure)

```text
MARS_Project/
├── config/                         # 全域組態管理
│   ├── __init__.py                 # 全域組態聚合入口
│   ├── db_cfg.py                   # 資料庫全域組態配置
│   ├── algo_cfg.py                 # 演算法全域組態配置
│   ├── path_cfg.py                 # 實體路徑全域組態配置
│   └── maps/                       # 子組態映射管理中心
│       ├── schm_maps/              # 資料表屬性定義集
│       └── stage_maps/             # 階層式管線策略集
│
├── data/                           # 實體資料管理
│   ├── 01_train/                   # 模型訓練資料
│   │   ├── task_queue/             # 批次訓練任務佇列
│   │   ├── raw_vids/               # 待抽幀之模型訓練影片
│   │   ├── frames/                 # 待標註之時序抽幀全景影像
│   │   ├── roboflow/               # 物件標註資料集
│   │   └── archive/                # 模型訓練資料歸檔集
│   └── 02_infer/                   # 模型辨識資料
│       ├── vids_todo/              # 待處理之模型辨識影片
│       └── vids_done/              # 模型辨識資料歸檔集
│
├── src/                            # 核心功能管理
│   ├── __init__.py                 # 核心功能聚合套件入口
│   ├── data_prep/                  # 訓練-資料整備套件
│   ├── modl_opts/                  # 訓練-模型優化套件
│   ├── vis_recg/                   # 辨識-視覺辨識套件
│   ├── post_proc/                  # 辨識-後處理過濾套件
│   ├── database/                   # 資料庫管理套件
│   └── database/                   # 圖形化介面套件
│
├── utils/                          # 共用工具管理中心
│   ├── __init__.py                 # 共用工具聚合套件入口
│   ├── env/                        # 系統環境管理套件
│   └── log/                        # 系統日誌管理套件
│
├── logs/                           # 系統日誌紀錄
├── weights/                        # 模型版本管理
├── main.py                         # 專案管線調度工具
├── train_pipe.py                   # 訓練管線調度工具
├── infer_pipe.py                   # 辨識管線調度工具
├── README.md                       # 專案操作使用說明
├── requirements.txt                # 專案工具安裝清單
└── .env                            # 全域環境變數組態
```

---

## 🔄 管線生命週期 (Pipeline Lifecycle)

```
[訓練生命週期 : T1 ~ T8]
T1. 影像抽幀 (Extra)    : 依領域與拍攝視角動態匹配間距，多核心並行抽取高解析時序影格
T2. 光線調整 (Clahe)    : 實施 YUV-CLAHE 自適應直方圖均衡化，局部抑制強光反光並提亮陰影細節
T3. 人工畫框 (Roboflow) : 全景影像實施單趟混合標註，以分段標籤集界定主物件與次物件幾何範圍
T4. 級聯協調 (Coord)    : 實施主物件安全限幅裁切、次目標坐標投影與標籤平移歸零，並產出血統溯源檔
T5. 模型訓練 (Yolo)     : 依階層策略規範自動掛載骨幹權重，串聯驅動 Stage-1 宏觀與 Stage-2 微觀模型擬合收斂
T6. 靜態編譯 (TensorRT) : 導出最佳原生權重並實施 FP16 半精度量化，編譯生成高吞吐靜態加速引擎 (.engine)
T7. 資料寫入 (Write)    : 高效批次寫入模型版本資訊 (modl_ver_info)，並同步各輪次損失與收斂等成效指標 (modl_fit_eval)
T8. 檔案歸檔 (Archive)  : 建立 <YYYY-MM-DD>.txt 日期標記，將影片、抽幀影像與標註資料封存至 archive/

[辨識生命週期 : I1 ~ I7]
I1. 模型預載 (Preload)  : 依循階層策略規範，將各階層 YOLO 模型與重型影像處理工具 (e.g., OCR) 預載至主記憶體中
I2. 串流抽幀 (Stream)   : 解析動態視訊來源並依領域間距採樣影格，計算各幀精確之實際時間標記 (recg_ts)
I3. 物件偵測 (Detect)   : 驅動 Stage-1 宏觀檢測模型定位並鎖定全景主物件位置 (HBB/OBB)
I4. 特徵辨識 (Pipeline) : 動態調度領域專屬管線，串聯執行主物件裁切、次物件定位、透視校正拉平與字元辨識
I5. 資料寫入 (Write)    : 展平多階層複合特徵資訊，高效批次寫入資料庫中 (e.g., live_veh_recg)
I6. 檔案歸檔 (Archive)  : 建立 <VER_ID>.txt 版本標記，並將處理完成之待辨識影片搬遷至封存目錄 (vids_done)
I7. 資源釋放 (Clear)    : 清空預載模型快取並觸發記憶體資源回收 (防止資源殘留與溢出)
```

---

## 🏷️ 命名規範 (Naming Conventions)

### 1. 影片命名規範 (6 段式命名法)
所有放入 `raw_vids/` 或 `vids_todo/` 的影片檔案（支援 `.mp4`、`.avi`、`.mkv` 等），必須嚴格符合 6 段底線切分命名格式：
> **格式**：`<PHASE>_<CAM_BRAND>_<DOMAIN>_<ANGLE>_<YYYYMMDD>_<HHMMSS>.<ext>`

- `<PHASE>`：運行階段代碼，僅限 `TRAIN`（訓練）或 `INFER`（辨識）。
- `<CAM_BRAND>`：攝影設備或機型代碼（e.g., `ACEPRO`、`S25U`、`CAM01`）。
- `<DOMAIN>`：目標領域識別碼（e.g., 車輛 `VEH`、馬拉松 `MARA`）。
- `<ANGLE>`：拍攝視角代碼（e.g., 俯視 `ROOF`、側視 `SIDE`）。
- `<YYYYMMDD>`：拍攝日期（8 碼西曆數字，e.g., `20260705`）。
- `<HHMMSS>`：拍攝時間（6 碼 24 小時制數字，e.g., `120000`）。

*命名範例*：
- 訓練影片：`TRAIN_ACEPRO_VEH_ROOF_20260705_120000.mp4`
- 辨識影片：`INFER_S25U_VEH_SIDE_20260707_120000.mp4`

### 2. 模型版本標準 (Semantic Versioning)
模型版本代碼（`ver_id`）採嚴格語意化版本編號 `v<Major>.<Minor>.<Patch>`（e.g., `v1.0.0`）：
- **主要版本 (Major)**：物件類別數量異動（e.g., `nc` 增減）、輸入尺寸變更、神經網絡骨幹架構更替。
- **次要版本 (Minor)**：物件類別不變下，納入新採樣之訓練影像資料進行大幅重新訓練，或模型規格提升（e.g., `Nano 升至 Small`）。
- **修補版本 (Patch)**：訓練輪次 (`Epochs`) 與學習率 (`LR`) 調整，或針對少量樣本進行模型微調。

---

## ⚙️ 環境安裝與配置 (Setup & Installation)

### 1. 系統需求
- **作業系統**：Windows 10/11 / Ubuntu 20.04+
- **Python 版本**：3.10 ~ 3.12 (建議 3.12)
- **資料庫**：PostgreSQL 14+
- **加速運算**：NVIDIA GPU（建議具備 8GB 以上顯示卡記憶體 VRAM，需安裝 CUDA 12.1+ 及對應之 TensorRT 環境）

### 2. 安裝相依套件
```bash
pip install -r requirements.txt
```

### 3. 環境變數設定 (.env)
在專案根目錄建立 `.env` 檔案，配置 PostgreSQL 連線參數與專案搜尋路徑：
```bash
# 資料庫連線配置
DB_HOST=localhost
DB_PORT=5432
DB_NAME=MARS_DB
DB_USERNAME=postgres
DB_PASSWORD=your_secure_password

# 專案搜尋路徑
PYTHONPATH=.
```

---

## ⌨️ 命令列介面 (CLI)

所有任務皆透過全域程序啟動中心 `main.py` 進行統一調度。

### 1. 圖形化操作介面 (GUI)
需要使用圖形化介面操作時執行：
```bash
# 開啟圖形化操作介面 (設計版本 PyQt6)
python main.py --mode gui
```

### 2. 資料庫初始化 (Database Initialization)
首次部署系統或需要重置資料庫環境時執行：
```bash
# 檢查並自動建立所需資料表（初次執行全面建立，後續執行自動補齊新增表格）
python main.py --mode init

# 強制刪除現存既有資料表(含架構)，徹底重新建置 ⚠️
python main.py --mode init --wipe
```

### 3. 啟動訓練管線 (Run Training Pipeline)
需要處理原始訓練影片、訓練新版模型或更新權重時執行：
```bash
# 完整流程 (影片抽幀 -> 光線調整 -> 級聯協調 -> 模型訓練 -> 靜態編譯 -> 資料寫入 -> 檔案歸檔)
python main.py --mode train --stage all --ver_id VEH:v1.0.0

# 多領域同時執行
python main.py --mode train --stage all --ver_id VEH:v1.0.0 MARA:v1.0.0

# 資料整備 (影片抽幀 -> 光線調整)
python main.py --mode train --stage prepare

# 模型優化 (級聯協調 -> 模型訓練 -> 靜態編譯 -> 資料寫入 -> 檔案歸檔)
python main.py --mode train --stage train --ver_id VEH:v1.0.0 --epochs VEH:150 --lr VEH:0.01
```

### 4. 啟動辨識管線 (Run Inference Pipeline)
需要對待處理影片進行自動辨識、追蹤並產出結果記錄時執行：
```bash
# 自動探測最新權重版本，並優先使用 TensorRT 靜態編譯
python main.py --mode infer --stage infer

# 明確指定模型權重版本
python main.py --mode infer --stage infer --ver_id VEH:v1.0.0

# 停用 TensorRT 靜態編譯，降級使用 PyTorch (.pt) 原生權重進行辨識
python main.py --mode infer --stage infer --no_trt
```

---

## 📦 管線運作調度圖 (Pipeline Execution Flow)
系統支援 **CLI 命令列** 與 **PyQt6 圖形化介面 (GUI)** 雙軌啟動，底層統一由全域啟動中心 `main.py` 驅動，生命週期終止時保證執行快取清理。
```text
[訓練管線調度流程 : train_pipe]
啟動 train_pipe (exec_stage='all')
  │
  ├──► 階段一：資料整備 (DATA-PREP)
  │      └─ ConcExtractor: 掃描 raw_vids/，多進程執行 T1 影像抽幀與 T2 光線調整 (YUV-CLAHE)
  │           └─ 輸出高畫質時序影格 (Frame) 至 frames/<DOMAIN>/<VID_STEM>/
  │
  └──► 階段二：模型優化 (MODL-OPTS)
         └─ 逐領域循環處理 (e.g., VEH):
              ├─ [T4 級聯協調] StageCoordinator 實施主物件裁切、次物件座標投影與類別標籤歸零
              ├─ [T5 模型訓練] YoloModelTrainer 串聯訓練 Stage-1 與 Stage-2 模型
              ├─ [T6 靜態編譯] TensorRTExporter 將 best.pt 編譯為 FP16 best.engine
              ├─ [T7 資料寫入] write_repo.train 寫入 modl_ver_info 與 modl_fit_eval，記錄模型資訊與訓練成效
              └─ [T8 檔案歸檔] archive_train_file 建立 <YYYY-MM-DD>.txt 日期標記，並將影片、抽幀影像與標註資料分層封存至 archive/

[辨識管線調度流程 : infer_pipe]
啟動 infer_pipe (exec_stage='infer')
  │
  ├──► 掃描 vids_todo/ 取得待處理之有效影片
  └──► 逐影片迭代處理 (e.g., VEH):
         ├─ [I1 模型預載] VisionModelPreloader 預載主/次 YOLO 模型與 PaddleOCR 至主記憶體中
         ├─ [I2 串流抽幀] VideoStreamer 依視角間距採樣，計算各幀精確之實際時間標記 (recg_ts)
         ├─ [I3 物件偵測] ObjectDetector 驅動 Stage-1 宏觀模型定位並鎖定全景主物件位置 (HBB/OBB)
         ├─ [I4 特徵辨識] PipelineDispatcher 派發 veh_pipe:
         │    ├─ BodyCropper: 依主物件位置實施邊界限幅裁切
         │    ├─ ColorExtractor: HSV 空間萃取車體主色
         │    ├─ 次物件偵測: 定位次目標車牌 (OBB) 與廠牌 (HBB)
         │    ├─ GeomWarper: 車牌四點透視投影拉平校正
         │    ├─ CharReader: OCR 字元識別與規則過濾清洗
         │    └─ 坐標空間還原: 次目標局部點位 (l_bbox) 換算為全景坐標 (g_bbox)
         ├─ [I5 資料寫入] write_repo.infer 展平多階特徵，批次 Upsert 寫入 live_veh_recg
         ├─ [I6 檔案歸檔] archive_infer_file 建立 <VER_ID>.txt 版本標記，並將影片封存至 vids_done/
         └─ [I7 資源釋放] preloader.clear_cache() 清空預載模型快取並觸發記憶體資源回收 (防止資源殘留與溢出)
```