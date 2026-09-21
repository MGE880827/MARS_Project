# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 資料整備工具套件入口 (Data Preparation Toolkit Entry)
# 維護日期: 2026-09-13
# 檔案路徑: MARS_Project/src/data_prep/__init__.py
# ##########################################################################################

# 掛載內部依賴
from . import data_prep_insp
from .sgl_vid_extr import SingleVideoExtractor
from .conc_vid_extr import ConcurrentVideoExtractor

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. data_prep_insp           : [檔案] 資料整備檢查與統計工具 >>> 專職負責訓練影片探測、領域分佈數量統計與未歸檔抽幀狀態之細節監控。
# 2. SingleVideoExtractor     : [類別] 單一影片抽幀引擎 >>> 負責單一影片之 OpenCV 影片串流解析、依指定間隔實施抽幀，以及高規格訓練影像預處理與整備。
# 3. ConcurrentVideoExtractor : [類別] 影片並行抽幀引擎 >>> 負責全域目錄掃描、任務佇列建構、多核心進程分發與日誌彙整。

__all__ = ["data_prep_insp", "SingleVideoExtractor", "ConcurrentVideoExtractor"]