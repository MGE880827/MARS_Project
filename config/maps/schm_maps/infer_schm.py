# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 辨識層級之資料表屬性定義 (Infer Level Table Attributes Definition)
# 維護日期: 2026-09-17
# 檔案路徑: MARS_Project/config/maps/schm_maps/infer_schm.py
# ##########################################################################################

# 掛載外部依賴
import pkgutil
import importlib
from pathlib import Path

# ==========================================================================================
"""
[名稱] 辨識層級之資料表屬性動態聚合引擎 (Infer Level Table Attributes Dynamic Aggregation Engine)
[作用] 自動掃描 infer_tables/ 目錄下所有以 _table 結尾之領域資料表設定檔，動態聚合出完整的 INFER_SCHEMA_INFO 總表。
"""
INFER_SCHEMA_INFO = {}

_table_dir = Path(__file__).resolve().parent / "infer_tables"

# 掃描 infer_tables/ 目錄下所有以 _table 結尾之領域資料表定義
if _table_dir.exists():
    for _, _mod_name, _ in pkgutil.iter_modules([str(_table_dir)]):
        # 僅鎖定以 _table 結尾之領域設定檔，跳過內部私有或暫存模組
        if _mod_name.endswith("_table"):
            _module = importlib.import_module(f".infer_tables.{_mod_name}", package=__package__)
            # 聚合各領域模組導出之 INFER_SCHEMA_INFO
            if hasattr(_module, "INFER_SCHEMA_INFO"):
                INFER_SCHEMA_INFO.update(getattr(_module, "INFER_SCHEMA_INFO"))

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. INFER_SCHEMA_INFO : [定義] 辨識層級之資料表屬性字典 >>> 動態聚合各領域辨識流程資料表之實體名稱與核心欄位結構。

__all__ = ["INFER_SCHEMA_INFO"]