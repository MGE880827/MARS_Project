# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 物件邊界裁切工具 (Object Boundary Cropping Tool)
# 維護日期: 2026-09-18
# 檔案路徑: MARS_Project/src/vis_recg/img_proc/body_crop.py
# ##########################################################################################

# 掛載外部依賴
import traceback
import numpy as np

# 掛載內部依賴
from utils import log

# ==========================================================================================
class BodyCropper:
    """
    [名稱] 物件邊界裁切引擎 (Object Boundary Cropping Engine)
    [作用] 接收影像與 HBB 座標，執行邊界限幅與安全裁切，輸出目標 ROI 影像供二階段模型或分類器使用。
    """
    # =============================================
    def __init__(self, sys_cfg=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「物件邊界裁切引擎」，保留全域組態接收介面以利彈性擴充。
        [參數] sys_cfg: [dict] 全域組態檢索 (選填，保留未來擴充彈性)
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg = sys_cfg
    
    # =============================================
    def __call__(self, frame, hbb_pts):
        """
        [名稱] Func.B 物件邊界裁切函式
        [功能] 接收原始單幀影像與 HBB 座標，執行邊界限幅與安全切片裁切，回傳目標 ROI 影像。
        [參數] 共計 2 組參數，以下說明:
               - frame   : [numpy.ndarray] 輸入之 BGR 影像矩陣 (H, W, C)
               - hbb_pts : [list] 水平邊界框座標 [x1, y1, x2, y2]
        [輸出] numpy.ndarray: 裁切後之 ROI 影像矩陣 (copy)；若參數無效或裁切失敗則回傳 None
        """
        # 驗證輸入參數有效性
        is_valid_pts = isinstance(hbb_pts, (list, tuple, np.ndarray)) and len(hbb_pts) == 4
        if frame is None or not is_valid_pts:
            err_args = "frame" if frame is None else "hbb_pts"
            err_stat = "輸入影像為 None" if err_args == "frame" else "輸入座標為 None 或數量不等於 4"
            log.CONTENT(
                type = "PROCESS",
                targ = "BODY-CROP",
                idnt = "VALIDATE_ARGUMENTS",
                stat = "WARN",
                msge = f"[參數: {err_args}] 驗證警示 >>> {err_stat}，跳過物件邊界裁切程序"
            )
            return None
        
        try:
            # [STEP-1] 提取影像高度與寬度邊界
            img_h, img_w = frame.shape[:2]

            # [STEP-2] 水平邊界框座標數值轉型及邊界限幅
            x1 = max(0, min(int(round(float(hbb_pts[0]))), img_w))
            y1 = max(0, min(int(round(float(hbb_pts[1]))), img_h))
            x2 = max(0, min(int(round(float(hbb_pts[2]))), img_w))
            y2 = max(0, min(int(round(float(hbb_pts[3]))), img_h))

            # [STEP-3] 檢查裁切區域幾何有效性，寬度與高度需大於 0
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w <= 0 or crop_h <= 0 :
                log.CONTENT(
                    type = "PROCESS",
                    targ = "BODY-CROP",
                    idnt = "VERIFY_CROP_DIMENSIONS",
                    stat = "WARN",
                    msge = f"[尺寸: {crop_w}x{crop_h}] 裁切警示 >>> 目標幾何面積無效 (低於 0 像素)，跳過物件邊界裁切程序"
                )
                return None
            
            # [STEP-4] 執行目標區域裁切並回傳 ROI 矩陣
            roi_crop = frame[y1:y2, x1:x2].copy()
            return roi_crop
        
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "PROCESS",
                targ = "BODY-CROP",
                idnt = "HANDLE_BODY_CROP_EXCEPTION",
                stat = "FAIL",
                msge = f"[任務: 物件邊界裁切] 運行失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            return None