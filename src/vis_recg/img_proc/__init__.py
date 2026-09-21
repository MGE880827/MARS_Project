# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 影像處理工具套件入口 (Image Processing Toolkit Entry)
# 維護日期: 2026-09-18
# 檔案路徑: MARS_Project/src/vis_recg/img_proc/__init__.py
# ##########################################################################################

# 掛載內部依賴
from .body_crop import BodyCropper
from .color_extr import ColorExtractor
from .geom_warp import GeomWarper
from .char_ocrs import CharReader

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. BodyCropper    : [類別] 物件邊界裁切引擎 >>> 接收 HBB 座標 [x1,y1,x2,y2] 並執行邊界限幅與安全切片裁切。
# 2. ColorExtractor : [類別] 色彩特徵提取引擎 >>> 接收裁切後 ROI 影像，透過 HSV 色彩空間統計並提取主流色彩特徵。
# 3. GeomWarper     : [類別] 幾何透視校正引擎 >>> 接收 OBB 4 頂點座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 並計算透視矩陣，完成傾斜目標平整拉直校正。
# 4. CharReader     : [類別] 光學字元辨識引擎 >>> 接收校正後 ROI 影像，調用底層 OCR 提取字元並依領域規則執行正規化。

__all__ = ["BodyCropper", "ColorExtractor", "GeomWarper", "CharReader"]