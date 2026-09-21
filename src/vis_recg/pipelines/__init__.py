# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 領域特徵提取管線聚合入口 (Domain Feature Extraction Pipeline Package Entry)
# 維護日期: 2026-09-19
# 檔案路徑: MARS_Project/src/vis_recg/pipelines/__init__.py
# ##########################################################################################

# 掛載外部依賴
import pkgutil
import importlib
from pathlib import Path

# ==========================================================================================
"""
[名稱] 領域特徵提取管線動態聚合引擎 (Domain Feature Extraction Pipelines Dynamic Aggregation Engine)
[作用] 自動掃描當前 pipelines/ 目錄下所有以 _pipe 結尾之領域管線檔案，動態探測模組中定義之 PIPELINE_TOOLS 工具字典並聚合至總表。
"""
PIPELINE_TOOLS = {}

_pipe_dir = Path(__file__).resolve().parent

# 掃描 pipelines/ 目錄下所有以 _pipe 結尾之領域管線定義
if _pipe_dir.exists():
    for _, _mod_name, _ in pkgutil.iter_modules([str(_pipe_dir)]):
        # 僅鎖定以 _pipe 結尾之領域管線檔，跳過內部私有或暫存模組
        if _mod_name.endswith("_pipe"):
            _module = importlib.import_module(f".{_mod_name}", package=__package__)
            # 動態聚合各領域模組導出之 PIPELINE_TOOLS
            if hasattr(_module, "PIPELINE_TOOLS"):
                PIPELINE_TOOLS.update(getattr(_module, "PIPELINE_TOOLS"))

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. PIPELINE_TOOLS : [定義] 全域領域特徵提取管線總表 >>> 動態聚合各領域管線類別映射 (e.g., {"VEH": VehiclePipeline})

__all__ = ["PIPELINE_TOOLS"]