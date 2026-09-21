# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 物件辨識工具套件入口 (Visual Recognition Toolkit Entry)
# 維護日期: 2026-09-19
# 檔案路徑: MARS_Project/src/vis_recg/__init__.py
# ##########################################################################################

# 掛載內部依賴
from . import vis_recg_insp
from .modl_load import VisionModelPreloader
from .vid_stream import VideoStreamer
from .obj_detect import ObjectDetector
from .dispatcher import PipelineDispatcher
from . import img_proc
from . import pipelines

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. vis_recg_insp        : [檔案] 視覺辨識檢查與統計工具 >>> 負責待辦辨識影片狀態探測、領域細節檢查與數據統計分析。
# 2. VisionModelPreloader : [類別] 視覺辨識模型加載引擎 >>> 負責多階段辨識模型之組態檢索、權重預載與辨識資源生命週期管控。
# 3. VideoStreamer        : [類別] 影片時序串流引擎 >>> 負責各類型動態視訊來源之連接管理、時序影格解碼緩衝與流量穩定控制。
# 4. ObjectDetector       : [類別] 物件影像偵測引擎 >>> 負責驅動多尺度空間檢測模型，執行各階段目標定位、姿態估算與空間座標映射。
# 5. PipelineDispatcher   : [類別] 領域管線配發工具 >>> 負責依目標領域策略動態分派影像處理工作鏈，實現階層特徵提取與語意解析。
# 6. img_proc             : [套件] 影像處理工具套件 >>> 集中提供邊界限幅裁切、空間幾何校正與光學符號識別等底層處理運算子。
# 7. pipelines            : [套件] 領域特徵提取管線套件 >>> 動態聚合之領域管線集合空間。

__all__ = [
    # 一級視覺管線核心引擎
    "vis_recg_insp", "VisionModelPreloader", "VideoStreamer", "ObjectDetector", "PipelineDispatcher",

    # 子工具與管線命名空間 (供離線呼叫、單元測試或特定領域組態調用)
    "img_proc", "pipelines"
]
