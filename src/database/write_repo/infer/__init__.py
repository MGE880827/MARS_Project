# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 辨識資料寫入工具套件入口 (Infer Data Write Toolkit Entry)
# 維護日期: 2026-09-20
# 檔案路徑: MARS_Project/src/database/write_repo/infer/__init__.py
# ##########################################################################################

# 掛載內部依賴
from .veh_domain import upsert_veh_recg_logs, upsert_veh_monitor_records

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. upsert_veh_recg_logs      : [函式] 車輛辨識日誌寫入函式 >>> 寫入即時車輛偵測與特徵辨識結果至 infer.live_veh_recg。
# 2. upsert_veh_monitor_records: [函式] 車輛監控清單寫入函式 >>> 寫入或更新管制車輛黑名單設定至 infer.live_veh_moni。

__all__ = [
    # [VEH] 車輛辨識領域
    "upsert_veh_recg_logs", "upsert_veh_monitor_records"
]