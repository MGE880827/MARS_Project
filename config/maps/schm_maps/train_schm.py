# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 訓練層級之資料表屬性定義 (Train Level Table Attributes Definition)
# 維護日期: 2026-09-17
# 檔案路徑: MARS_Project/config/maps/schm_maps/train_schm.py
# ##########################################################################################

"""
[名稱] 訓練層級之資料表屬性字典 (Train Level Table Attributes Dictionary)
[作用] 負責集中定義訓練層級之資料表實體名稱、寫入模式與核心欄位結構，確保模型成效履歷落地之絕對一致性。
"""
# ==========================================================================================
# < 字典配置說明表 >
# 1. schm : [str]  資料表所屬綱要架構 (train, infer)
# 2. tabl : [str]  資料表實體名稱
# 3. mode : [dict] 欄位衝突更新模式 (如 ON CONFLICT DO UPDATE 之覆寫策略)
# 4. cols : [list] 資料表欄位順序 (嚴格對應 SQL 寫入與 DDL 欄位定義)
# 5. ukey : [list] 資料表唯一主鍵或複合唯一鍵清單
# ==========================================================================================

TRAIN_SCHEMA_INFO = {
    # =============================================
    # ｜TRAIN｜TABL-001｜模型版本資訊
    # =============================================
    "ver_info" : {
        "schm" : "train",
        "tabl" : "modl_ver_info",
        "mode" : {},
        "cols" : [
            "main_domain",  # 主領域識別代碼 (e.g., "VEH")          ※ 次階層時恆保留所屬主領域
            "main_ver_id",  # 主領域版本代碼 (e.g., "v1.0.0")       ※ 次階層時恆保留所屬主版本
            "sub_domain",   # 次領域識別代碼 (e.g., "LIC", "BRAND") ※ Stage-1 主階層恆寫入 "NONE"
            "sub_ver_id",   # 次領域版本代碼 (e.g., "v1.1.0")       ※ Stage-1 主階層恆寫入 "NONE"
            "stage_level",  # 當前模型階層 (1: Stage-1 宏觀檢測, 2: Stage-2 微觀檢測)
            "bbox_type",    # 邊界框類型 ("HBB": 水平框, "OBB": 旋轉框)
            "train_date",   # 訓練完成日期戳記 (YYYY-MM-DD)
            "algo_name",    # 演算法架構 (e.g., "MARS-YOLO11")
            "bone_name",    # 基礎預訓練骨幹權重 (e.g., "yolo11n.pt", "yolo11n-obb.pt")
            "wght_path",    # 最佳權重實體儲存路徑 (weights/<DOMAIN>/.../best.pt)
            "class_qty",    # 該領域物件類別數量
            "img_size",     # 輸入影像之解析度 (e.g., 640, 480)
            "batch_size",   # 訓練批次大小
            "accumulate",   # 梯度累積步數
            "epoch_qty",    # 總訓練輪次數
            "learn_rate"    # 最佳化初始學習率
        ],
        "ukey" : ["main_domain", "main_ver_id", "sub_domain", "sub_ver_id"]
    },
    # =============================================
    # ｜TRAIN｜TABL-002｜模型成效指標
    # =============================================
    "fit_eval" : {
        "schm" : "train",
        "tabl" : "modl_fit_eval",
        "mode" : {},
        "cols" : [
            "main_domain",  # 主領域識別代碼 (e.g., "VEH")          ※ 次階層時恆保留所屬主領域
            "main_ver_id",  # 主領域版本代碼 (e.g., "v1.0.0")       ※ 次階層時恆保留所屬主版本
            "sub_domain",   # 次領域識別代碼 (e.g., "LIC", "BRAND") ※ Stage-1 主階層恆寫入 "NONE"
            "sub_ver_id",   # 次領域版本代碼 (e.g., "v1.1.0")       ※ Stage-1 主階層恆寫入 "NONE"
            "epoch_idx",    # 當前訓練輪數序號 (1 ~ epoch_qty)
            "train_loss",   # 訓練損失收斂值 (Box Loss)
            "val_loss",     # 驗證損失收斂值 (Box Loss)
            "map50",        # 邊界框平均精準度 (mAP@0.5)
            "map95",        # 邊界框平均精準度 (mAP@0.5:0.95)
            "prec_val",     # 模型精準率 (Precision)
            "recall_val"    # 模型召回率 (Recall)
        ],
        "ukey" : ["main_domain", "main_ver_id", "sub_domain", "sub_ver_id", "epoch_idx"]
    }
}