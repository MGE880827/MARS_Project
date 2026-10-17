# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 訓練資料寫入工具套件入口 (Train Data Write Toolkit Entry)
# 維護日期: 2026-09-16
# 檔案路徑: MARS_Project/src/database/write_repo/train/__init__.py
# ##########################################################################################

# 掛載內部依賴
from .ver_info import upsert_version_info
from .fit_eval import upsert_fit_evaluation

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. upsert_version_info   : [函式] 模型版本資訊寫入函式 >>> 寫入模型核心規格、訓練參數與權重檔路徑至 train.modl_ver_info。
# 2. upsert_fit_evaluation : [函式] 模型成效指標寫入函式 >>> 寫入模型各輪次 (Epoch) 之訓練損失值與評估指標至 train.modl_fit_eval。

__all__ = ["upsert_version_info", "upsert_fit_evaluation"]