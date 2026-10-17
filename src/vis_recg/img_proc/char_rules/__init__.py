# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 字元辨識規則套件聚合入口 (Character OCR Rules Package Entry)
# 維護日期: 2026-09-18
# 檔案路徑: MARS_Project/src/vis_recg/img_proc/char_rules/__init__.py
# ##########################################################################################

# 掛載外部依賴
import pkgutil
import importlib
from pathlib import Path

# ==========================================================================================
"""
[名稱] 字元辨識規則動態聚合引擎 (Character OCR Rules Dynamic Aggregation Engine)
[作用] 自動掃描當前 char_rules/ 目錄下所有以 _rule 結尾之領域配置檔，動態探測模組中定義之 CHAR_RULES 工具字典並聚合至總表。
"""
CHAR_RULES = {}

_rule_dir = Path(__file__).resolve().parent

# 掃描 char_rules/ 目錄下所有以 _rule 結尾之領域資料表定義
if _rule_dir.exists():
    for _, _mod_name, _ in pkgutil.iter_modules([str(_rule_dir)]):
        # 僅鎖定以 _rule 結尾之領域設定檔，跳過內部私有或暫存模組
        if _mod_name.endswith("_rule"):
            _module = importlib.import_module(f".{_mod_name}", package=__package__)
            # 聚合各領域模組導出之 CHAR_RULES
            if hasattr(_module, "CHAR_RULES"):
                CHAR_RULES.update(getattr(_module, "CHAR_RULES"))

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. CHAR_RULES : [定義] 全域字元辨識規則總表 >>> 動態聚合各領域之字元辨識清洗規格與特徵標籤。

__all__ = ["CHAR_RULES"]