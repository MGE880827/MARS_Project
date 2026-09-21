# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 模型辨識階層之策略配置 (Model Inference Stage Configuration)
# 維護日期: 2026-09-18
# 檔案路徑: MARS_Project/config/maps/stage_maps/infer_stage.py
# ##########################################################################################

# 掛載內部依賴
import pkgutil
import importlib
from pathlib import Path

# ==========================================================================================
"""
[名稱] 模型辨識階層之策略動態聚合引擎 (Model Inference Stage Dynamic Aggregation Engine)
[作用] 聚合多領域階層化辨識架構，由 rules/ 目錄動態掃描各領域靜態規則檔，依領域規範主階段目標檢測信心度、次階段局部微觀特徵擷取門檻與幾何校正策略。
"""
INFER_STAGE_RULES = {}

_rule_dir = Path(__file__).resolve().parent / "rules"

# 掃描 rules/ 目錄下所有以 _rule 結尾之領域配置檔
if _rule_dir.exists():
    for _, _mod_name, _ in pkgutil.iter_modules([str(_rule_dir)]):
        # 僅鎖定以 _rule 結尾之領域設定檔，跳過內部私有或暫存模組
        if _mod_name.endswith("_rule"):
            _module = importlib.import_module(f".rules.{_mod_name}", package=__package__)
            _domain = _mod_name.split("_")[0].upper()
            # 僅抽取該領域之 INFER 策略字典
            if hasattr(_module, "INFER"):
                INFER_STAGE_RULES[_domain] = getattr(_module, "INFER")

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. INFER_STAGE_RULES : [定義] 各領域辨識策略動態聚合總表 >>> 動態搜集 rules/ 目錄下所有以 _rule 結尾模組之 INFER 策略字典。

__all__ = ["INFER_STAGE_RULES"]