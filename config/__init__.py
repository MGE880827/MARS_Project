# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 全域組態聚合入口 (Global Configuration Aggregation Entry)
# 維護日期: 2026-09-11
# 檔案路徑: MARS_Project/config/__init__.py
# ##########################################################################################

# 掛載內部依賴
from .db_cfg import DB_PARAMS, TABLES
from .algo_cfg import algo_config
from .path_cfg import path_config
from .maps.stage_maps import TRAIN_STAGE_RULES, INFER_STAGE_RULES

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. DB_PARAMS         : [封裝] 資料庫連線組態字典 >>> 儲存底層驅動程序所需之核心定址與授權參數。
# 2. TABLES            : [封裝] 資料表屬性檢索字典 >>> 儲存全域資料表之架構歸屬、實體名稱與鍵值等核心配置。
# 3. algo_config       : [實例] 演算法組態管理引擎 >>> 負責全域演算法核心參數之集中管理與靜態調用。
# 4. path_config       : [實例] 實體路徑組態管理引擎 >>> 負責全域資料夾與檔案實體路徑之集中管理與靜態調用。
# 5. TRAIN_STAGE_RULES : [定義] 模型訓練階層策略字典 >>> 定義多領域訓練任務規格、骨幹配置與空間裁切映射。
# 6. INFER_STAGE_RULES : [定義] 模型辨識階層策略字典 >>> 定義多領域即時辨識管線、信心度門檻與幾何拉平策略。

__all__ = [
    "DB_PARAMS",
    "TABLES",
    "algo_config",
    "path_config",
    "TRAIN_STAGE_RULES",
    "INFER_STAGE_RULES"
]