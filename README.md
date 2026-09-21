# MARS - 多重目標自動辨識系統
**Multiple-target Automated Recognition System**

MARS 是一套具備高度可擴展性與工業級強健防呆機制的「端到端 (End-to-End) 智慧電腦視覺系統」。專為複雜、多物件場景設計，原生支援**單階層全景目標定位**與**雙階層級聯微觀特徵提取 (Two-Stage Cascade)** 架構。

本系統涵蓋了從前置視訊「時序抽幀與光學自適應調整」、「全景幾何裁切與空間坐標投影」、「YOLO 雙階深度擬合」、「TensorRT FP16 靜態加速引擎編譯」，到推論期「無漂移時間戳視訊串流解碼」、「動態領域管線調度」、「多模態特徵解析 (幾何透視拉平 / HSV 色彩 / OCR 字元)」以及「PostgreSQL 高效批次防撞覆寫 (Upsert) 落地與生命週期同構歸檔」之全流程自動化。

---

## 🌟 核心特色 (Core Features)

1. **雙階層級聯幾何協調架構 (Two-Stage Cascade Architecture)**
   - **Stage-1 (宏觀主階層)**：全景大圖目標定位（如：整輛車 VEH），支援常規水平框 (HBB) 與旋轉外接框 (OBB)。
   - **Stage-2 (微觀次階層)**：驅動 `StageCoordinator`，針對全景標註實施「安全限幅外擴裁切 (Safe Cropping)」，自動將次目標坐標投影換算至局部切片並將類別編號平移歸零 (Class Remapping)，同步產出次階 `data.yaml` 與完整資料血統溯源檔 (`sub_obj_lineage.json`)。
2. **多模態特徵提取工具鏈 (Multi-modal Feature Extraction Tools)**
   - **空間幾何校正 (`GeomWarper`)**：自動解析 OBB 頂點幾何順序，透過 OpenCV 透視變換矩陣將傾斜目標（如車牌）拉平校正為標準矩形。
   - **邊界安全限幅 (`BodyCropper`)**：提供自適應邊界檢測與限幅保護，杜絕影像切片維度退化與記憶體拷貝異常。
   - **色彩特徵提取 (`ColorExtractor`)**：核心區域取樣與 HSV 色彩空間轉換，自動消除背景噪訊並統計物件主體色。
   - **強健字元辨識 (`CharReader`)**：首選預載之 `PaddleOCR` 引擎，並內建 `EasyOCR` 自動降級備援機制；對接 `CHAR_RULES` 進行字串清洗、屬性分離（如一般車、電動車、軍車）與白名單正則過濾。
3. **推論效能極致最佳化與顯存防禦 (Extreme Inference & VRAM Defense)**
   - 全自動導出最佳權重並實施 **TensorRT FP16 半精度靜態量化**，大幅縮短推論延遲。
   - 具備預載管理機制 (`VisionModelPreloader`)，於辨識啟動時將主/次 YOLO 模型與重量級深度學習引擎預先駐留於顯存 (`yolo_pool` / `tool_pool`)，推論期零硬碟 I/O 開銷；任務結束後顯式觸發快取清理與垃圾回收，徹底杜絕顯存溢出 (OOM)。
4. **工業級資料庫部署與衝突覆寫 (Robust Database Orchestration)**
   - 整合 `psycopg2`，動態解析 Schema/Table 定義並部署 DDL。
   - 採用 `INSERT ... ON CONFLICT DO UPDATE` (Upsert) 機制，搭配變異偵測 (`IS DISTINCT FROM`)，確保高併發時序資料零遺漏且具備嚴格冪等性。

---

## 📂 專案目錄結構 (Project Structure)

```text
MARS_Project/
├── config/                         # 全域組態管理中心
│   ├── __init__.py                 # 全域組態聚合入口
│   ├── db_cfg.py                   # 資料庫連線參數與資料表架構映射
│   ├── algo_cfg.py                 # 演算法基準 (抽幀間距、CLAHE、YOLO 訓練/推論參數)
│   ├── path_cfg.py                 # 專案實體絕對/相對路徑管理
│   └── maps/                       # 動態策略與資料庫定義映射
│       ├── schm_maps/              # 資料庫表綱要定義 (train_schm, infer_schm, infer_tables)
│       └── stage_maps/             # 領域階層策略 (train_stage, infer_stage, rules)
│
├── data/                           # 專案實體資料掛載區
│   ├── 01_train/                   # 模型訓練資料區
│   │   ├── task_queue/             # 批次訓練任務佇列 (task_001, task_002...)
│   │   ├── raw_vids/               # 當前待抽幀之原始訓練影片
│   │   ├── frames/                 # 待標註之時序抽幀全景影像 (<DOMAIN>/<VID_STEM>/)
│   │   ├── roboflow/               # 標註資料集 (含主階與 sub/ 次階資料集)
│   │   └── archive/                # 訓練同構鏡像歷史歸檔庫 (<DOMAIN>/<VER_ID>/)
│   └── 02_infer/                   # 模型辨識資料區
│       ├── vids_todo/              # 待處理之辨識影片
│       └── vids_done/              # 辨識完成之歷史歸檔區 (<DOMAIN>/<YYYY-MM-DD>/)
│
├── src/                            # 核心功能套件
│   ├── __init__.py                 # 核心功能聚合套件入口
│   ├── data_prep/                  # 資料整備工具 (並行抽幀、光學 CLAHE 調整、狀態探測)
│   ├── modl_opts/                  # 模型優化工具 (階層級聯協調、YOLO 自動擬合、TensorRT 編譯)
│   ├── vis_recg/                   # 視覺辨識工具 (模型預載、串流解碼、目標檢測、領域管線、影像處理)
│   ├── post_proc/                  # 後處理過濾工具 (時序抗噪、跨影格平滑追蹤 - 擴充模組)
│   └── database/                   # 資料庫調度工具 (架構初始化、DDL 部署、批次 Upsert 寫入)
│
├── utils/                          # 系統共用工具
│   ├── __init__.py                 # 共用工具聚合入口
│   ├── env/                        # 環境管理 (快取清理、外部輸入解析、檔案搬遷與最新權重探測)
│   └── log/                        # 結構化日誌輸出 (分級列印、月/日自動輪轉檔案)
│
├── logs/                           # 實體系統日誌儲存目錄 (logs/<YYYY-MM>/<YYYY-MM-DD>.log)
├── weights/                        # 模型權重儲存庫 (weights/<DOMAIN>/[main|sub]/<VER_ID>/)
├── main.py                         # 系統核心啟動與 CLI 入口
├── train_pipe.py                   # 自動化訓練管線調度器
├── infer_pipe.py                   # 自動化辨識管線調度器
├── requirements.txt                # 系統環境依賴套件表
└── .env                            # 環境變數設定檔 (資料庫存取憑證)
```

---

## 🔄 系統管線生命週期 (Pipeline Lifecycle)

```
[訓練生命週期 : T1 ~ T8]
T1. 影像抽幀 (Extra)    : 依領域與拍攝視角動態匹配間距，多核心並行抽取高解析時序影格
T2. 光線調整 (Clahe)    : 實施 YUV-CLAHE 自適應直方圖均衡化，局部抑制強光反光並提亮陰影細節
T3. 人工畫框 (Roboflow) : 全景影像實施單趟混合標註，以分段標籤集同步界定主物件與次目標幾何範圍
T4. 級聯協調 (Coord)    : 實施主物件安全限幅裁切、次目標坐標投影與標籤平移歸零，並建立血統溯源檔
T5. 模型訓練 (Yolo)     : 依階層策略自動掛載骨幹權重，串聯驅動 Stage-1 宏觀與 Stage-2 微觀模型擬合收斂
T6. 靜態編譯 (TensorRT) : 導出最佳原生權重並實施 FP16 半精度量化，編譯生成高吞吐靜態加速引擎 (.engine)
T7. 資料寫入 (Write)    : 結構化寫入模型版本詮釋資料 (modl_ver_info)，並同步各輪次損失與收斂指標 (modl_fit_eval)
T8. 檔案歸檔 (Archive)  : 將原始影片、抽幀集與標註資料庫同構鏡像封存至歸檔區，維護版本實體履歷

[辨識生命週期 : I1 ~ I7]
I1. 模型預載 (Preload)  : 依循階層策略查閱註冊表，將各階 YOLO 模型與重型影像引擎 (如 OCR) 預先駐留於顯存
I2. 串流抽幀 (Stream)   : 解析動態視訊來源並依領域間距採樣影格，同步綁定精確無漂移之實體地理時間戳記
I3. 物件偵測 (Detect)   : 驅動 Stage-1 宏觀檢測模型定位全景目標，自適應解析輸出結構化邊界框 (HBB/OBB)
I4. 特徵辨識 (Pipeline) : 動態調度領域專屬管線，串聯執行主體裁切、次目標定位、透視校正拉平與字元辨識
I5. 資料寫入 (Write)    : 展平多階層複合特徵資訊，高效批次持久化寫入即時推論資料表 (如 live_veh_recg)
I6. 檔案歸檔 (Archive)  : 將處理完畢之待辨識影片建立版本戳記，自動分流搬遷至結構化歷史封存目錄 (vids_done)
I7. 資源釋放 (Clear)    : 清空模型快取池並顯式觸發垃圾回收，徹底釋放 GPU 顯存與記憶體以防禦溢出 (OOM)
```

---

## 🏷️ 命名規範與資料標準 (Conventions & Standards)

### 1. 影片標準命名規範 (6 段式命名法)
所有放入 `raw_vids/` 或 `vids_todo/` 的影片，必須符合以下格式：
> **格式**：`<PHASE>_<CAM_BRAND>_<DOMAIN>_<ANGLE>_<YYYYMMDD>_<HHMMSS>.mp4`

- `<PHASE>`：運行階段代碼，僅限 `TRAIN`（訓練）或 `INFER`（辨識）。
- `<CAM_BRAND>`：攝影機或設備代碼（如：`ACEPRO`, `SONY`, `CAM01`）。
- `<DOMAIN>`：目標領域識別碼（如：車輛 `VEH`、馬拉松 `MARA`）。
- `<ANGLE>`：拍攝角度視角（如：俯視 `ROOF`、側視 `SIDE`）。
- `<YYYYMMDD>`：拍攝日期（8 碼西曆數字，如：`20260705`）。
- `<HHMMSS>`：拍攝時間（6 碼 24 小時制數字，如：`120000`）。

*範例*：
- 訓練影片：`TRAIN_ACEPRO_VEH_ROOF_20260705_120000.mp4`
- 辨識影片：`INFER_ACEPRO_VEH_ROOF_20260707_120000.mp4`

### 2. 模型版本命名標準 (Semantic Versioning)
模型版本代碼（`ver_id`）採嚴格語意化命名 `v<Major>.<Minor>.<Patch>`（如 `v1.0.0`）：
- **主要版本 (Major)**：類別數量異動（`nc` 增減）、輸入尺寸變更、骨幹架構換代。
- **次要版本 (Minor)**：類別結構不變下的大規模資料擴增，或模型等級提升（如 Nano 升 Medium）。
- **修補版本 (Patch)**：超參數微調（Batch、學習率調整）或特定極端環境小規模資料調整 (Fine-tuning)。

---

## ⚙️ 環境安裝與配置 (Setup & Installation)

### 1. 系統需求
- **作業系統**：Ubuntu 20.04+ / Windows 10/11
- **Python 版本**：3.9+
- **資料庫**：PostgreSQL 14+
- **加速運算**：NVIDIA GPU（建議具備 8GB 以上顯存，需安裝 CUDA 11.8+ / 12.X 及對應之 TensorRT 環境）

### 2. 安裝相依套件
```bash
pip install -r requirements.txt
```

### 3. 環境變數設定 (`.env`)
在專案根目錄建立 `.env` 檔案，配置 PostgreSQL 連線授權資訊與執行路徑：
```env
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

## 💻 命令列操作手冊 (CLI Usage)

所有任務皆透過全域程序啟動中心 `main.py` 進行統一調度。

### 1. 資料庫初始化 (Database Initialization)
首次部署系統或需要重置資料庫環境時執行：
```bash
# 建立缺失的架構 (Schema) 與所有領域資料表
python main.py --mode init

# 【危險操作】強制刪除既有所有 Schema 與資料表，徹底重新建置
python main.py --mode init --wipe
```

### 2. 啟動訓練管線 (Run Training Pipeline)
自動依據當前實體檔案狀態執行對應階段：
```bash
# 【全流程執行】抽幀 -> 級聯切片協調 -> YOLO訓練 -> TRT靜態編譯 -> 入庫與歸檔
python main.py --mode train --stage all --ver_id VEH:v1.0.0

# 多領域同時擬合訓練
python main.py --mode train --stage all --ver_id VEH:v1.0.0 MARA:v1.0.0

# 【僅執行資料整備】僅對 raw_vids/ 內影片進行抽幀與 CLAHE 光學優化
python main.py --mode train --stage prepare

# 【僅執行模型優化】標註完成後，指定訓練輪次與初始學習率
python main.py --mode train --stage train --ver_id VEH:v1.0.0 --epochs VEH:150 --lr VEH:0.01
```

### 3. 啟動辨識管線 (Run Inference Pipeline)
處理 `data/02_infer/vids_todo/` 內所有待辨識影片：
```bash
# 自動探測最新權重版本，優先使用 TensorRT (.engine) 引擎加速推論
python main.py --mode infer --stage infer

# 明確指定模型權重版本
python main.py --mode infer --stage infer --ver_id VEH:v1.0.0

# 停用 TensorRT 加速，降級使用 PyTorch (.pt) 原生權重進行推論
python main.py --mode infer --stage infer --no_trt
```

---

## 🗺️ 端到端執行調度圖 (Execution Flow)

```text
[訓練管線調度流程 : train_pipe]
啟動 train_pipe (exec_stage='all')
  │
  ├──► 【階段一：DATA-PREP 資料整備】
  │      └─ conc_vid_extr: 掃描 raw_vids/，多進程執行 T1 影像抽幀與 T2 光線調整 (YUV-CLAHE)
  │         └─ 輸出高畫質時序影格至 frames/<DOMAIN>/<VID_STEM>/
  │
  └──► 【階段二：MODL-OPTS 模型優化】
         └─ 逐領域循環 (例: VEH):
              ├─ [T4 級聯協調] StageCoordinator 實施主體安全裁切、次目標投影與標籤歸零
              ├─ [T5 模型訓練] YoloModelTrainer 串聯訓練 Stage-1 與 Stage-2 模型
              ├─ [T7 資料寫入] 寫入 modl_ver_info 與 modl_fit_eval
              ├─ [T6 靜態編譯] TensorRTExporter 將 best.pt 編譯為 FP16 best.engine
              └─ [T8 檔案歸檔] archive_train_file 搬遷影片、抽幀與標註集至 archive/ 目錄

[辨識管線調度流程 : infer_pipe]
啟動 infer_pipe (exec_stage='infer')
  │
  ├──► 掃描 vids_todo/ 取得有效待處理視訊任務
  └──► 逐影片迭代處理 (例: VEH 影片):
         ├─ [I1 模型預載] VisionModelPreloader 預載主/次 YOLO 模型與 PaddleOCR 至顯存快取池
         ├─ [I2 串流抽幀] VideoStreamer 依視角間距採樣，綁定精確無漂移之實體時間戳 (recg_ts)
         ├─ [I3 物件偵測] ObjectDetector 驅動 Stage-1 模型輸出全景邊界框 (HBB/OBB)
         ├─ [I4 特徵辨識] PipelineDispatcher 派發 veh_pipe:
         │    ├─ BodyCropper: 裁切主車體
         │    ├─ ColorExtractor: HSV 萃取車色
         │    ├─ 次物件偵測: 抓取車牌 (OBB) 與廠牌 (HBB)
         │    ├─ GeomWarper: 車牌透視拉平校正
         │    ├─ CharReader: OCR 號碼讀取與正規化清洗
         │    └─ 坐標空間還原: 次目標局部點位換算回全景坐標 (g_bbox)
         ├─ [I5 資料寫入] 展平多階特徵，批次 Upsert 寫入 live_veh_recg
         ├─ [I6 檔案歸檔] archive_infer_file 標記 <VER_ID>.txt 並搬遷至 vids_done/
         └─ [I7 資源釋放] preloader.clear_cache() 顯式清空快取並觸發垃圾回收 (防禦 OOM)
```