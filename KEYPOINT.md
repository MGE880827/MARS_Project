# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 核心架構與管線生命週期規範 (Project Core Architecture & Pipeline Specifications)
# 維護日期: 2026-09-21
# 檔案路徑: MARS_Project/KEYPOINT.md
# ##########################################################################################

# [訓練流程]
    T1. 影像抽幀 (Extra)    : 依領域與拍攝視角動態匹配間距，多核心並行抽取高解析時序影格
    T2. 光線調整 (Clahe)    : 實施 YUV-CLAHE 自適應直方圖均衡化，局部抑制強光反光並提亮陰影細節
    T3. 人工畫框 (Roboflow) : 全景影像實施單趟混合標註，以分段標籤集同步界定主物件與次目標幾何範圍
    T4. 級聯協調 (Coord)    : 實施主物件安全限幅裁切、次目標坐標投影與標籤平移歸零，並建立血統溯源檔
    T5. 模型訓練 (Yolo)     : 依階層策略自動掛載骨幹權重，串聯驅動 Stage-1 宏觀與 Stage-2 微觀模型擬合收斂
    T6. 靜態編譯 (TensorRT) : 導出最佳原生權重並實施 FP16 半精度量化，編譯生成高吞吐靜態加速引擎 (.engine)
    T7. 資料寫入 (Write)    : 結構化寫入模型版本詮釋資料 (modl_ver_info)，並同步各輪次損失與收斂指標 (modl_fit_eval)
    T8. 檔案歸檔 (Archive)  : 將原始影片、抽幀集與標註資料庫同構鏡像封存至歸檔區，維護版本實體履歷

# [辨識流程]
    I1. 模型預載 (Preload)  : 依循階層策略查閱註冊表，將各階 YOLO 模型與重型影像引擎 (如 OCR) 預先駐留於顯存
    I2. 串流抽幀 (Stream)   : 解析動態視訊來源並依領域間距採樣影格，同步綁定精確無漂移之實體地理時間戳記
    I3. 物件偵測 (Detect)   : 驅動 Stage-1 宏觀檢測模型定位全景目標，自適應解析輸出結構化邊界框 (HBB/OBB)
    I4. 特徵辨識 (Pipeline) : 動態調度領域專屬管線，串聯執行主體裁切、次目標定位、透視校正拉平與字元辨識
    I5. 資料寫入 (Write)    : 展平多階層複合特徵資訊，高效批次持久化寫入即時推論資料表 (如 live_veh_recg)
    I6. 檔案歸檔 (Archive)  : 將處理完畢之待辨識影片建立版本戳記，自動分流搬遷至結構化歷史封存目錄 (vids_done)
    I7. 資源釋放 (Clear)    : 清空模型快取池並顯式觸發垃圾回收，徹底釋放 GPU 顯存與記憶體以防禦溢出 (OOM)

# [專題架構]
MARS_Project /
│
├── config/           
│   ├── __init__.py     # 全域組態聚合入口
│   ├── db_cfg.py       # 資料庫全域組態配置
│   ├── algo_cfg.py     # 演算法全域組態配置
│   ├── path_cfg.py     # 實體路徑全域組態配置
│   │
│   └── maps/
│       ├── __init__.py         # 子組態映射聚合入口
│       ├── schm_maps/  
│       │   ├── __init__.py          # 資料表屬性定義集入口 
│       │   ├── train_schm.py        # 訓練層級之資料表屬性定義
│       │   ├── infer_schm.py        # 辨識層級之資料表屬性定義
│       │   └── infer_tables/
│       │       ├── __init__.py      # 辨識資料表套件標記
│       │       ├── veh_table        # 車輛領域之資料表屬性定義
│       │       └── ...後續擴充
│       │
│       └── stage_maps/
│           ├── __init__.py          # 階層式管線策略配置入口
│           ├── train_stage.py       # 模型訓練階層之策略配置
│           ├── infer_stage.py       # 模型辨識階層之策略配置
│           └── rules/
│               ├── __init__.py      # 領域策略規則套件標記
│               ├── veh_rule         # 車輛領域階層策略配置 (雙層)
│               ├── ...後續擴充
│               └── sgl_demo.py      # DENO領域階層策略配置 (單層)
│
│
├── data/               # 影像、圖片及標註資料集
│   │
│   ├── 01_train/                    # 模型訓練資料類型
│   │   │
│   │   ├── task_queue/                    # 批次訓練任務佇列
│   │   │   ├── task_001/
│   │   │   │   └── <VID_STEM>.mp4
│   │   │   └── task_002/
│   │   │       └── <VID_STEM>.mp4
│   │   │
│   │   ├── raw_vids/                      # 待抽幀之模型訓練影片
│   │   │   └── <VID_STEM>.mp4                      # 模型訓練影片
│   │   │ 
│   │   ├── frames/                        # 待標註之時序抽幀全景影像
│   │   │   └── <DOMAIN>/                           # 核心領域類別
│   │   │       └── <VID_STEM>/                     # 同影片之全景抽幀影像
│   │   │           └── <VID_STEM>_<FRM_INDX>.jpg   # 全景抽幀影像
│   │   │
│   │   ├── roboflow/                      # 物件標註資料集
│   │   │   └── <DOMAIN>/                           # 主領域物件標註資料集 (e.g., "VEH")
│   │   │       ├── data.yaml                       # 全景類別配置檔 (含主領域與次領域類別)
│   │   │       ├── train/                          # 全景訓練集 (images/ & labels/)
│   │   │       ├── valid/                          # 全景驗證集 (images/ & labels/)
│   │   │       ├── test/                           # 全景測試集 (images/ & labels/)
│   │   │       └── sub/                            # 次領域物件標註資料集 (單階層領域無此目錄)
│   │   │           ├── sub_obj_lineage.json        # 次領域之物件生成溯源標記
│   │   │           └── <SUB_DOMAIN>/               # 次領域類別 (e.g., "LIC", "BRAND")
│   │   │               ├── data.yaml               # 主物件裁切訓練配置檔
│   │   │               ├── train/                  # 主物件裁切訓練集 (images/ & labels/)
│   │   │               └── valid/                  # 主物件裁切訓練集 (images/ & labels/)
│   │   │
│   │   └── archive/                       # 模型訓練資料歸檔集
│   │       └── <DOMAIN>/
│   │           └── <VER_ID>/
│   │               ├── <YYYY-MM-DD>.txt            # 訓練日期標註檔
│   │               ├── vids/                       # 待抽幀之模型訓練影片
│   │               │   └── <VID_STEM>.mp4
│   │               ├── imgs/                       # 待標註之時序抽幀全景影像
│   │               │   └── <VID_STEM>/                        # 同影片之全景抽幀影像
│   │               │       └── <VID_STEM>_<FRM_INDX>.jpg      # 全景抽幀影像
│   │               └── roboflow/                   # 物件標註資料集
│   │                   ├── data.yaml                          # 全景類別配置檔 (含主領域與次領域類別)
│   │                   ├── train/                             # 全景訓練集 (images/ & labels/)
│   │                   ├── valid/                             # 全景驗證集 (images/ & labels/)
│   │                   ├── test/                              # 全景測試集 (images/ & labels/)
│   │                   └── sub/                    # 次領域物件標註資料集 (單階層領域無此目錄)
│   │                       ├── sub_obj_lineage.json           # 次領域之物件生成溯源標記
│   │                       └── <SUB_DOMAIN>/                  # 次領域類別 (e.g., "LIC", "BRAND")
│   │                           ├── data.yaml                  # 主物件裁切訓練配置檔
│   │                           ├── train/                     # 主物件裁切訓練集 (images/ & labels/)
│   │                           └── valid/                     # 主物件裁切訓練集 (images/ & labels/)
│   │
│   └── 02_infer/                    # 模型辨識資料類型
│       │
│       ├── vids_todo/               # 待處理之模型辨識影片
│       │   └── <VID_STEM>.mp4             # 模型辨識影片
│       │
│       └── vids_done/               # 模型辨識資料歸檔集
│           └── <DOMAIN>/
│               └── <YYYY-MM-DD>/
│                   ├── <VER_ID>.txt       # 模型版本標註檔
│                   └── <VID_STEM>.mp4     # 模型辨識影片
│
│
├── src/
│   ├── __init__.py                 # 核心功能聚合套件入口
│   │ 
│   ├── data_prep/                  # 訓練-資料整備 (負責 T1-影像抽偵, T2-光線調整)
│   │   ├── __init__.py                      # 資料整備工具套件入口
│   │   ├── data_prep_insp.py                # 資料整備檢查與統計工具
│   │   │   ├── check_raw_video_exist              # 訓練影片狀態檢驗函式
│   │   │   ├── get_raw_video_summary              # 訓練影片數據統計函式
│   │   │   ├── get_unarchived_frames_detail       # 未歸檔抽幀細節探測函式
│   │   │   └── check_unarchived_frames_exist      # 未歸檔抽幀狀態檢驗函式
│   │   │
│   │   ├── sgl_vid_extr.py                  # 單一影片抽幀工具 [※後續擴充-影像品質過濾函式]
│   │   └── conc_vid_extr.py                 # ️️影片並行抽幀工具
│   │
│   │
│   ├── modl_opts/                  # 訓練-模型優化 (負責 T4-級聯協調, T5-模型訓練, T6-靜態編譯)
│   │   ├── __init__.py                      # 模型優化工具套件入口
│   │   ├── modl_opts_insp.py                # 模型優化檢查與統計工具
│   │   │   ├── get_available_datasets             # 標註資料集細節探測函式
│   │   │   ├── check_unprocessed_datasets_exist   # 標註資料集狀態檢驗函式
│   │   │   └── get_dataset_summary                # 標註資料集數據統計函式
│   │   │
│   │   ├── stage_coord.py                   # 階層級聯協調工具
│   │   ├── auto_train.py                    # 自動化模型訓練工具
│   │   └── trt_export.py                    # 最佳權重靜態編譯工具
│   │
│   │
│   ├── vis_recg/                   # 辨識-視覺辨識
│   │   ├── __init__.py                      # 物件辨識工具套件入口
│   │   ├── vis_recg_insp.py                 # 視覺辨識檢查與統計工具
│   │   │   ├── check_pending_videos_exist         # 辨識影片狀態檢驗函式
│   │   │   ├── get_pending_video_summary          # 辨識影片數據統計函式
│   │   │   └── get_available_pending_domains      # 待辦影片領域細節探測函式
│   │   │
│   │   ├── modl_load.py                     # 視覺辨識模型加載工具
│   │   ├── vid_stream.py                    # 影片時序串流工具
│   │   ├── obj_detect.py                    # 影像物件偵測工具
│   │   ├── dispatcher.py                    # 領域管線配發工具
│   │   │
│   │   ├── pipelines/                       # 領域管線流程圖
│   │   │   ├── __init__.py                  # 領域特徵提取管線入口
│   │   │   ├── veh_pipe.py                  # 車輛領域特徵提取工具
│   │   │   └── ...後續擴充
│   │   │
│   │   └── img_proc/                        # 影像處理工具
│   │       ├── __init__.py                  # 影像處理工具套件入口
│   │       ├── color_extr.py                # 色彩特徵提取工具
│   │       ├── body_crop.py                 # 物件邊界裁減工具
│   │       ├── geom_warp.py                 # 幾何透視校正工具
│   │       ├── char_ocrs.py                 # 光學字元辨識工具
│   │       └── char_rules/         
│   │           ├── __init__.py              # 字元辨識規則套件聚合入口
│   │           ├── veh_rule.py              # 車輛領域之字元辨識配置規則
│   │           └── ...後續擴充
│   │
│   ├── post_proc/                  # 辨識-後處理過濾
│   │   ├── __init__.py                      # 物件過濾工具套件入口 [※後續擴充]         
│   │   ├── plat_fltr.py                     # 車牌時序過濾工具 [※後續擴充]
│   │   └── trck_fltr.py                     # 人物追蹤過濾工具 [※後續擴充]
│   │
│   │
│   └── database/                   # 資料庫管理
│       ├── __init__.py             # 資料庫核心聚合套件入口 
│       ├── db_init.py              # 資料庫架構初始化工具
│       ├── db_mngr.py              # 資料庫調度管理工具
│       │
│       ├── deploy_repo/            # 資料庫部署指令
│       │   │
│       │   ├── train/                       # TRAIN-SQL 部署指令集
│       │   │   ├── ver_info.sql             # 模型版本資訊 (modl_ver_info)
│       │   │   └── fit_eval.sql             # 模型成效指標 (modl_fit_eval)
│       │   │
│       │   └── infer/                       # INFER-SQL 部署指令集
│       │       ├── veh_recg.sql             # 車輛辨識日誌 (live_veh_recg)
│       │       ├── veh_moni.sql             # 車輛監控日誌 (live_veh_moni)
│       │       └── mara_moni.sql            # [※後續擴充]
│       │
│       ├── write_repo              # 資料庫寫入指令 (負責 T7-資料寫入, I5-資料寫入)
│       │   ├── __init__.py                  # 資料庫寫入聚合套件入口
│       │   │
│       │   ├── train/                       # TRAIN-SQL 寫入指令集
│       │   │   ├── __init__.py              # 訓練資料寫入工具套件入口
│       │   │   ├── ver_info.py              # 模型版本資訊同步寫入工具
│       │   │   └── fit_eval.py              # 模型成效指標同步寫入工具
│       │   │
│       │   └── infer/                       # INFER-SQL 寫入指令集
│       │       ├── __init__.py              # 辨識資料寫入工具套件入口
│       │       ├── veh_domain.py            # 車輛領域資料庫寫入工具
│       │       └── mara_domain.py           # [※後續擴充]
│       │
│       └── query_repo/                      # 資料庫查詢指令
│           ├── __init__.py                  # 資料庫查詢聚合套件入口 [※後續擴充]
│           │
│           ├── train/                       # TRAIN-SQL 查詢指令集
│           │   ├── __init__.py              # 訓練資料查詢工具套件入口 [※後續擴充]
│           │   └── .py                      # [※後續擴充]
│           │
│           └── infer/                       # INFER-SQL 查詢指令集
│               ├── __init__.py              # 辨識資料查詢工具套件入口 [※後續擴充]
│               └── .py                      # [※後續擴充]
│ 
│ 
├── utils/
│   ├── __init__.py                 # 共用工具聚合套件入口
│   │
│   ├── env/                        # 系統環境管理 (負責 T8-檔案歸檔, I6-檔案歸檔)
│   │   ├── __init__.py                       # 環境管理工具套件入口
│   │   │
│   │   ├── cache_remover.py                  # 環境快取移除工具
│   │   │   └── remove_pycache                # Python 編譯快取移除函式
│   │   │
│   │   ├── input_resolver.py                 # 外部輸入解析工具
│   │   │   ├── analyze_video_filename        # 單一影片檔名分析函式
│   │   │   ├── to_root_relative              # 專案相對路徑解析函式
│   │   │   └── parse_kv_pair                 # CLI 鍵值對標準解析函式
│   │   │
│   │   └── file_manager.py                   # 實體檔案管理工具
│   │       ├── _resolve_task_queue_route     # 待抽幀任務佇列路由解析函式
│   │       ├── upload_train_videos           # 訓練階段影片檢驗上傳函式
│   │       ├── upload_infer_videos           # 辨識階段影片檢驗上傳函式
│   │       ├── dispatch_next_task            # 佇列任務遞補與重命名調度函式
│   │       ├── archive_train_file            # 訓練資料自動歸檔函式
│   │       ├── archive_infer_file            # 辨識資料自動歸檔函式
│   │       └── find_latest_weight            # 最新權重版本探測函式
│   │   
│   └── log/                        # 系統日誌管理
│       ├── __init__.py                       # 日誌管理工具套件入口
│       └── sys_log.py                        # 系統日誌紀錄工具
│
│
├── logs/                           # 系統日誌紀錄
│   └── <YYYY-MM>/                            # 每月歸檔區域 (e.g., 2026-09/)
│       └── <YYYY-MM-DD>.log                  # 每日日誌記錄 (e.g., 2026-09-10.log)
│
├── weights/                        # 模型版本管理
│   └── <DOMAIN>/                             # 主領域類別 (e.g., VEH-車輛, MARA-馬拉松)
│       │
│       ├── main/                             # 一階段宏觀辨識 (主領域) 
│       │   └── <VER_ID>/                     # 核心版本封裝目錄 (e.g., v1.0.0/)
│       │       ├── best.pt                   # 最佳辨識原生權重 (PyTorch)
│       │       ├── best.engine               # 靜態編譯加速推論引擎 (TRTensor)
│       │       └── runs/                     # 該版本之核心權重訓練相關資訊
│       │
│       └── sub/                              # 二階段微觀特徵 (次領域，單階層領域無此目錄)
│           └── <SUB_DOMAIN>/                 # 次領域類別 (e.g., LIC-車牌, BRAND-廠牌)
│               └── <VER_ID>/                 # 核心版本封裝目錄 (e.g., v1.0.0/)
│                   ├── best.pt               # 最佳辨識原生權重 (PyTorch)
│                   ├── best.engine           # 靜態編譯加速推論引擎 (TRTensor)
│                   └── runs/                 # 該版本之核心權重訓練相關資訊
│    
├── .env                # 全域環境變數組態
├── main.py             # 專案管線調度工具
├── train_pipe.py       # 訓練管線調度工具
├── infer_pipe.py       # 辨識管線調度工具
├── README.md           # 專案說明文件
├── KEYPOINT.md         # 專案說明文件
└── requirements.txt    # 專案工具套件


# [未來擴充]
第三階段
├── db_maint/           # 數據庫維護 (DB Maintenance)
│   ├── auto_clean.py       # 磁碟空間清理任務 (自動刪除 30 天前的車牌截圖)
│   └── cloud_sync.py       # 冷熱資料分離任務 (將舊的車牌紀錄打包備份至雲端)

第四階段
├── dsys_rept/          # 報表與告警 (System Reporting)
│   ├── daily_stat.py       # 每日車流量與辨識率統計報表生成任務
│   └── blkl_alert.py       # 贓車/黑名單車輛背景比對與 Email 發送任務3


# [待訓練或辨識影片]
A. 命名規範 <PHASE>_<CAM_BRAND>_<DOMAIN>_<ANGLE>_<YYYYMMDD>_<HHMMSS>.mp4
   1. <PHASE>     : 運行階段 >>> TRAIN(訓練)、INFER(辨識)
   2. <CAM_BRAND> : 相機廠牌 >>> ACEPRO(運動相機)
   3. <DOMAIN>    : 目標領域 >>> VEH(車輛)、MARA(馬拉松)
   4. <ANGLE>     : 拍攝角度 >>> ROOF(俯視)、SIDE(側視) 
   5. <YYYYMMDD>  : 拍攝日期 >>> 西元年制(八碼數字) e.g., 20260705
   6. <HHMMSS>    : 拍攝時間 >>> 24小時制(六碼數字) e.g., 120000

B. 命名範例
   1. 訓練階段 (TRAIN) : TRAIN_ACEPRO_VEH_ROOF_20260705_120000.mp4
   2. 辨識階段 (INFER) : INFER_ACEPRO_VEH_ROOF_20260707_120000.mp4

C. 存放位置
   1. 訓練階段 (TRAIN) : data/01_train/raw_vids/
   2. 辨識階段 (INFER) : data/02_infer/vids_todo/


# [待歸檔資料]
A. 訓練階段
   1. 主要路徑: data/01_train/archive/<DOMAIN>/<VER_ID>/
      - <DOMAIN>     : 主領域識別代碼 (e.g., "VEH")
      - <SUB_DOMAIN> : 次領域識別代碼 (e.g., "LIC", "BRAND")
      - <YYYY-MM-DD> : 訓練執行日期，西元年制，由四碼-兩碼-兩碼組成 (e.g., 2026-07-10)
      - <VER_ID>     : 模型核心版本代碼 (e.g., "v1.0.0")
      - <VID_STEM>   : 抽幀圖片資料夾名稱，由六項元素組成 <PHASE>_<CAM_BRAND>_<DOMAIN>_<ANGLE>_<YYYYMMDD>_<HHMMSS>
      - <FRM_INDX>   : 抽幀圖片序號，五位數字 (e.g., 000001)
      - <VID_STEM>_<FRM_INDX> : 抽幀圖片名稱 (e.g., TRAIN_ACEPRO_VEH_ROOF_20260705_12000_00001)
   2. 存放位置
      - 空白標註 : <YYYY-MM-DD>.txt
      - 影片格式 : vids/<VID_STEM>.mp4
      - 圖片格式 : imgs/<VID_STEM>/<VID_STEM>_<FRM_INDX>.jpg
      - 標註資料 : roboflow/

B. 辨識階段
   1. 主要路徑: data/02_infer/vids_done/<DOMAIN>/<YYYY-MM-DD>/
      - <DOMAIN>     : 主領域識別目錄 (e.g., "VEH")
      - <YYYY-MM-DD> : 辨識執行日期，西元年制，由四碼-兩碼-兩碼組成 (e.g., 2026-07-20)
      - <VER_ID>     : 模型核心版本代碼 (e.g., "v1.0.0")
      - <VID_STEM>   : 辨識影片名稱，由六項元素組成 <PHASE>_<CAM_BRAND>_<DOMAIN>_<ANGLE>_<YYYYMMDD>_<HHMMSS>
   2. 存放位置
      - 空白標註 : <VER_ID>.txt
      - 影片格式 : <VID_STEM>.mp4


# [模型核心權重]
A. 命名規範 <ver_id>
   1. 由英文字母 v + 三組數字組成 >>> v<N1>.<N2>.<N3> (e.g., v1.0.0)
   2. 主要版本 (N1) : 辨識類別數量變更（nc增減）、輸入尺寸結構變動，或底層演算法架構換代 (YOLOv8 升 YOLOv10)。
   3. 次要版本 (N2) : 原類別結構不變下，執行大規模場景訓練資料擴增，或模型規格變更（如 Nano 升 Medium）。
   4. 修補版本 (N3) : 訓練參數調整（學習率、Batch），或特定死角與反光等小規模資料調整（Fine-tuning）。

B. 存放位置
   1. 主要路徑: weights/<DOMAIN>/
      - 主領域路徑: main/<VER_ID>/
      - 次領域路徑: sub/<SUB_DOMAIN>/<VER_ID>/
      - 原生權重 (PyTorch)  : best.pt
      - 靜態權重 (TRTensor) : best.engine
      - 訓練資料 (runs)     : runs/

   2. 核心權重訓練資料架構
      ├── weights/           # 核心權重
      │   ├── best.pt             # 最佳權重: 所有訓練輪次中，mAP50 最佳那論次
      │   └── last.pt             # 最後權重: 最後一次訓練輪次
      │
      ├── args.yaml          # 訓練階段參數: batch, lr0, imgsz, accumulate...等
      ├── results.csv        # 時序收斂日誌: 紀錄每一輪次的 loss, mAP50, mAP95, Precision, Recall
      ├── results.png        # results.csv 數據折線圖
      │
      ├── train_batch0.jpg   # 預覽圖 0: 訓練前處理馬賽克增強 (Mosaic Augmentation)
      ├── train_batch1.jpg   # 預覽圖 1
      ├── train_batch2.jpg   # 預覽圖 2
      │
      ├── val_batch0_labels.jpg   # 驗證集真實標註預覽圖 (Ground Truth) 
      ├── val_batch0_pred.jpg     # 驗證集模型預測預覽圖 (Predictions)
      │
      ├── confusion_matrix.png              # 混淆矩陣圖: 評估類別之間有沒有發生誤判，如 bus 被誤判為 truck
      ├── confusion_matrix_normalized.png   # 正規化後的混淆矩陣圖
      │
      ├── F1_curve.png       # F1-Score 曲線圖
      ├── PR_curve.png       # 召回率曲線圖-運算精準度 (Precision-Recall Curve)
      ├── P_curve.png        # 精準度曲線圖 (Precision Curve)
      └── R_curve.png        # 召回率曲線圖 (Recall Curve)


# [各階段流程]
A. [啟動 train_pipe(exec_stage='all')]
  │
  ├─► 【階段一：DATA-PREP 資料整備】
  │     │
  │     └─ 1. conc_vid_extr (多核心並行抽幀)
  │           ├─ 掃描 raw_vids/，解析檔名分配視角間距 (stream_cfg)
  │           ├─ sgl_vid_extr (單影片處理進程)
  │           │    ├─ T1 影像抽幀: 依間隔 (step_frms) 解碼
  │           │    └─ T2 光線調整: 執行 YUV-CLAHE 自適應光學均衡化
  │           └─ 輸出高畫質影格至 frames/<DOMAIN>/<VID_STEM>/
  │
  ├─► 【階段二：MODL-OPTS 模型優化】
  │     │
  │     ├─► 【循環主領域 1: VEH】
  │     │     │
  │     │     ├─ 1. StageCoordinator (T4 級聯協調)
  │     │     │     ├─ 讀取 roboflow/VEH/ 全景標註資料 (train/valid/test)
  │     │     │     ├─ body_crop: 依 crop_margin 執行主物件安全限幅裁切
  │     │     │     ├─ project_coords: 次目標 (LIC, BRAND) 坐標投影與標籤平移歸零
  │     │     │     └─ 產出次階資料集 sub/ 與血統溯源檔 (sub_obj_lineage.json)
  │     │     │
  │     │     ├─ 2. YoloModelTrainer (T5 模型訓練 & T7 資料寫入)
  │     │     │     ├─ 讀取 TRAIN_STAGE 骨幹權重 (bone_name) 與尺寸，並自 algo_cfg 掛載運算超參數 (batch, lr)
  │     │     │     ├─ 訓練 Stage-1 全景主模型 (VEH)
  │     │     │     ├─ 訓練 Stage-2 局部次模型 (LIC, BRAND)
  │     │     │     ├─ 解析 run/results.csv 提取各輪次 mAP50 等收斂指標
  │     │     │     └─ sync_train_database: 批次寫入 modl_ver_info 與 modl_fit_eval
  │     │     │
  │     │     ├─ 3. TensorRTExporter (T6 靜態編譯)
  │     │     │     └─ 讀取各階 best.pt，執行 FP16 半精度量化，編譯導出 best.engine
  │     │     │
  │     │     └─ 4. file_manager.archive_train_file (T8 檔案歸檔)
  │     │           └─ 將 raw_vids, frames, roboflow 鏡像封存至 archive/VEH/<VER_ID>/
  │     │
  │     └─► 【循環主領域 2: MARA】
  │           └─ (同上流程，若為單階層則略過 T4 級聯協調，直接執行 T5~T8)


B. [啟動 infer_pipe(exec_stage='infer')]
  │
  ├─► 【掃描 vids_todo 取得任務駐列】
  │
  ├─► 【循環處理影片: INFER_ACEPRO_VEH_ROOF...】
  │     │
  │     ├─ 1. VisionModelPreloader (I1 模型預載)
  │     │     ├─ 查閱 stage_cfg['VEH']['INFER'] 階層策略
  │     │     ├─ YOLO 主模型: 載入 VEH (best.engine)
  │     │     ├─ YOLO 次模型: 載入 LIC, BRAND (best.engine)
  │     │     └─ 引擎工具鏈: 偵測到 char_ocr ──> 預載 PaddleOCR 實例
  │     │     (※ 快取於 yolo_pool 與 tool_pool 顯存中，防止推論期重複 IO)
  │     │
  │     ├─ 2. VideoStreamer (I2 串流抽幀)
  │     │     ├─ 建立 OpenCV 影片迭代器
  │     │     ├─ 動態綁定無漂移之精確時間戳 (recg_ts)
  │     │     └─ yield 釋出當前幀影像矩陣 (frame) 與時空詮釋資料
  │     │
  │     ├─ 3. ObjectDetector (I3 物件偵測)
  │     │     └─ 餵入 Stage-1 模型，提取全景宏觀結構化邊界框清單 (HBB/OBB)
  │     │
  │     ├─ 4. PipelineDispatcher & veh_pipe (I4 特徵辨識)
  │     │     ├─ body_crop: 依 crop_margin 執行主車體裁切
  │     │     ├─ color_extr: HSV 色彩空間提取主視覺車色
  │     │     ├─ sub_models: 偵測次物件 (LIC 旋轉框, BRAND 水平框)
  │     │     ├─ geom_warp: 針對 LIC 執行透視矩陣變換與拉平校正
  │     │     ├─ char_read: 調用 OCR 提取號碼，並經 char_rules 白名單正規化過濾
  │     │     └─ 座標空間還原: 將次物件局部點位映射回全域座標 (g_bbox)
  │     │
  │     ├─ 5. _sync_infer_database (I5 資料寫入)
  │     │     ├─ 將多階層特徵字典 (proc_results) 展平對齊 SQL 欄位
  │     │     └─ 呼叫 db_mngr.SQL_UPSERT 批次持久化至 live_veh_recg 資料表
  │     │
  │     ├─ 6. file_manager.archive_infer_file (I6 檔案歸檔)
  │     │     └─ 辨識完畢之影片加上 <VER_ID>.txt 版本戳記，搬遷至 vids_done/VEH/
  │     │
  │     └─ 7. preloader.clear_cache (I7 資源釋放)
  │           └─ 清空 pool 字典並顯式呼叫 gc.collect()，防禦 OOM 顯存溢出
  │
  └─► 【循環處理影片: INFER_ACEPRO_MARA_SIDE...】
        └─ (同上流程，依 stage_cfg 動態配發 mara_pipe 執行專屬邏輯)