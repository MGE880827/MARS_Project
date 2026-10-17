# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 模型優化工具套件入口 (Model Optimization Toolkit Entry)
# 維護日期: 2026-09-16
# 檔案路徑: MARS_Project/src/modl_opts/__init__.py
# ##########################################################################################

# 掛載內部依賴
from . import modl_opts_insp
from .stage_coord import StageCoordinator
from .auto_train import YoloModelTrainer
from .trt_export import TensorRTExporter

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. modl_opts_insp   : [檔案] 模型優化檢查與統計工具 >>> 負責探測待訓練標註資料集，並統計影像與類別規模。
# 2. StageCoordinator : [類別] 階層級聯協調引擎 >>> 負責全景安全外擴裁切、次目標坐標動態投影與次階獨立訓練集建構。
# 3. YoloModelTrainer : [類別] 自動化模型訓練引擎 >>> 統籌單/雙階層架構擬合訓練，並將版本資訊與收斂指標批次同步至資料庫。
# 4. TensorRTExporter : [類別] 最佳權重靜態編譯引擎 >>> 承接訓練成果執行 FP16 靜態半精度量化，導出高吞吐 TensorRT (.engine) 靜態引擎。


__all__ = ["modl_opts_insp", "StageCoordinator", "YoloModelTrainer", "TensorRTExporter"]