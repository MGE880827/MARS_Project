# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 色彩特徵提取工具 (Color Feature Extraction Tool)
# 維護日期: 2026-09-18
# 檔案路徑: MARS_Project/src/vis_recg/img_proc/color_extr.py
# ##########################################################################################

# 掛載外部依賴
import cv2
import traceback
import numpy as np

# 掛載內部依賴
from utils import log

# ==========================================================================================
class ColorExtractor:
    """
    [名稱] 色彩特徵提取引擎 (Color Feature Extraction Engine)
    [作用] 接收裁切後之物件影像 (ROI)，透過中央區域擷取與 HSV 色彩空間轉換，統計並輸出物件之主視覺色彩。
    """
    # =============================================
    def __init__(self, sys_cfg=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「色彩特徵提取引擎」，並預載入 HSV 色彩空間之標準閾值定義字典。
        [參數] sys_cfg: [dict] 全域組態檢索 (選填，保留未來擴充彈性)
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg  = sys_cfg

        # 定義 HSV 色彩空間之上下限範圍 (Hue: 0~179, Saturation: 0~255, Value: 0~255)
        self.color_ranges = {
            "black"  : {"lower": np.array([0, 0, 0]),       "upper": np.array([180, 255, 46])},
            "white"  : {"lower": np.array([0, 0, 221]),     "upper": np.array([180, 30, 255])},
            "gray"   : {"lower": np.array([0, 0, 46]),      "upper": np.array([180, 43, 220])},
            "red"    : {"lower": np.array([0, 70, 50]),     "upper": np.array([10, 255, 255])},
            "red2"   : {"lower": np.array([170, 70, 50]),   "upper": np.array([180, 255, 255])},   # 紅色在 Hue 數值首尾相接
            "orange" : {"lower": np.array([11, 43, 46]),    "upper": np.array([25, 255, 255])},
            "yellow" : {"lower": np.array([26, 43, 46]),    "upper": np.array([34, 255, 255])},
            "green"  : {"lower": np.array([35, 43, 46]),    "upper": np.array([77, 255, 255])},
            "cyan"   : {"lower": np.array([78, 43, 46]),    "upper": np.array([99, 255, 255])},
            "blue"   : {"lower": np.array([100, 43, 46]),   "upper": np.array([124, 255, 255])},
            "purple" : {"lower": np.array([125, 43, 46]),   "upper": np.array([155, 255, 255])}
        }

    # =============================================
    def __call__(self, roi_crop):
        """
        [名稱] Func.B 色彩特徵提取函式
        [功能] 接收物件之 ROI 裁切影像，擷取中央 60% 核心區域以避開背景邊界，計算 HSV 像素占比最大之色彩名稱。
        [參數] roi_crop: [numpy.ndarray] 物件之 BGR 影像矩陣 (H, W, C)
        [輸出] str: 辨識出之主流色彩英文名稱 (e.g., "white", "black")；若失敗則回傳 "UNK"
        """
        fallback_color = "UNK"

        # 驗證輸入參數有效性
        if roi_crop is None or not isinstance(roi_crop, np.ndarray):
            log.CONTENT(
                type = "PROCESS",
                targ = "COLOR-EXTR",
                idnt = "VALIDATE_ARGUMENTS",
                stat = "WARN",
                msge = "[參數: roi_crop] 驗證警示 >>> 輸入影像為 None 或格式無效，跳過色彩特徵提取"
            )
            return fallback_color

        try:
            height, width = roi_crop.shape[:2]

            # 若圖片過小，直接捨棄辨識
            if height < 10 or width < 10:
                return fallback_color

            # [STEP-1] 擷取通用物件之中央核心區域 (取正中央 60% 範圍，避開背景雜訊)
            core_x_min, core_x_max = int(width * 0.20), int(width * 0.80)
            core_y_min, core_y_max = int(height * 0.20), int(height * 0.80)
            core_roi = roi_crop[core_y_min:core_y_max, core_x_min:core_x_max]
            if core_roi.size == 0:
                return fallback_color

            # [STEP-2] 色彩空間轉換與模糊降噪
            blur_roi = cv2.GaussianBlur(core_roi, (5, 5), 0)
            hsv_roi  = cv2.cvtColor(blur_roi, cv2.COLOR_BGR2HSV)

            # [STEP-3] 統計各色彩區間之像素數量
            color_counts = {}
            for color_name, bounds in self.color_ranges.items():
                mask  = cv2.inRange(hsv_roi, bounds["lower"], bounds["upper"])
                count = cv2.countNonZero(mask)
                
                # 合併 red & red2 (因 red 的 Hue 跨越 0 與 180 兩端)
                if color_name == "red2":
                    color_counts["red"] = color_counts.get("red", 0) + count
                else:
                    color_counts[color_name] = count

            # [STEP-4] 判別像素數量最大之主色彩
            if not color_counts:
                return fallback_color

            dominant_color = max(color_counts, key=color_counts.get)
            max_count = color_counts[dominant_color]

            # 若最大色彩的像素總量過低，視為無法判定
            total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
            if (max_count / total_pixels) < 0.1:
                return fallback_color

            return dominant_color

        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "PROCESS",
                targ = "COLOR-EXTR",
                idnt = "HANDLE_COLOR_EXTRACTION_EXCEPTION",
                stat = "FAIL",
                msge = f"[任務: 色彩特徵提取] 運行失敗 >>> 非預期系統異常，回傳 {fallback_color}，堆疊資訊如下: \n{sys_err}"
            )
            return fallback_color