# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 幾何透視校正工具 (Geometric Perspective Warping Tool)
# 維護日期: 2026-09-18
# 檔案路徑: MARS_Project/src/vis_recg/img_proc/geom_warp.py
# ##########################################################################################

# 掛載外部依賴
import cv2
import traceback
import numpy as np

# 掛載內部依賴
from utils import log

# ==========================================================================================
class GeomWarper:
    """
    [名稱] 幾何透視校正引擎 (Geometric Perspective Warp Engine)
    [作用] 接收旋轉邊界框 (OBB) 之 4 個傾斜角點，計算空間投影變換矩陣，透過 OpenCV 透視變換 (Perspective Transform) 將傾斜目標校正拉平為標準矩形 ROI。
    """
    # =============================================
    def __init__(self, sys_cfg=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「幾何透視校正引擎」，保留全域組態接收介面以利彈性擴充。
        [參數] sys_cfg: [dict] 全域組態檢索 (選填，保留未來擴充彈性)
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg = sys_cfg
    
    # =============================================
    def _order_points(self, src_obb_pts):
        """
        [名稱] Func.B 頂點順序校正函式
        [功能] 將任意順序輸入之 4 個頂點座標，依序排列為 [左上, 右上, 右下, 左下] 順序。
        [參數] src_obb_pts: [numpy.ndarray] 形狀為 (4, 2) 之頂點座標陣列
        [輸出] numpy.ndarray: 排序完成之 (4, 2) 頂點座標陣列
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # 左上角 (Top Left): x+y 最小；右下角 (Bottom Right): x+y 最大
        pts_sum = np.sum(src_obb_pts, axis=1)
        rect[0] = src_obb_pts[np.argmin(pts_sum)]
        rect[2] = src_obb_pts[np.argmax(pts_sum)]

        # 右上角 (Top Right): y-x 最小；左下角 (Bottom Left): y-x 最大
        pts_diff = np.diff(src_obb_pts, axis=1)
        rect[1] = src_obb_pts[np.argmin(pts_diff)]
        rect[3] = src_obb_pts[np.argmax(pts_diff)]

        return rect

    # =============================================
    def __call__(self, frame, obb_pts):
        """
        [名稱] Func.C 幾何透視校正函式
        [功能] 接收原始單幀影像與 OBB 4 角點，計算目標寬高幾何距離，執行透視矩陣變換並裁切輸出平整之矩形影像。
        [參數] 共計 2 組參數，以下說明:
               - frame   : [numpy.ndarray] 輸入之 BGR 影像矩陣 (H, W, C)
               - obb_pts : [list/np.ndarray] 旋轉矩形 4 頂點座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        [輸出] numpy.ndarray: 校正後之平整 ROI 影像矩陣；若輸入無效或變換失敗則回傳 None
        """
        # 驗證輸入參數有效性
        is_valid_pts = False
        src_obb_pts  = None
        if obb_pts is not None:
            try:
                covert_array = np.array(obb_pts, dtype="float32").reshape(4, 2)
                if covert_array.shape == (4, 2):
                    is_valid_pts = True
                    src_obb_pts  = covert_array
            except Exception:
                is_valid_pts = False

        if frame is None or not is_valid_pts:
            err_args = "frame" if frame is None else "obb_pts"
            err_stat = "輸入影像為 None" if err_args == "frame" else "輸入座標為 None 或無法轉換為 4x2 頂點矩陣"
            log.CONTENT(
                type = "PROCESS",
                targ = "GEOM-WARP",
                idnt = "VALIDATE_ARGUMENTS",
                stat = "WARN",
                msge = f"[參數: {err_args}] 驗證警示 >>> {err_stat}，跳過幾何透視校正程序"
            )
            return None
        
        try:
            # [STEP-1] 頂點順序校正
            rect = self._order_points(src_obb_pts)
            (tl, tr, br, bl) = rect   # tl (top-left), tr (top-right), br (bottom-right), bl (bottom-left)

            # [STEP-2] 計算校正後目標矩形之新寬度，取頂邊與底邊長度之最大值
            t_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            b_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            max_width = max(int(t_width), int(b_width))

            # [STEP-3] 計算校正後目標矩形之新高度，取左邊與右邊長度之最大值
            l_height = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
            r_height = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
            max_height = max(int(r_height), int(l_height))

            # [STEP-4] 攔截微小噪點或異常邊界框，長寬小於 5 像素直接捨棄
            if max_width < 5 or max_height < 5:
                log.CONTENT(
                    type = "PROCESS",
                    targ = "GEOM-WARP",
                    idnt = "VERIFY_WARP_DIMENSIONS",
                    stat = "WARN",
                    msge = f"[尺寸: {max_width}x{max_height}] 校正警示 >>> 目標幾何面積過小 (低於 5x5 像素)，跳過幾何透視校正程序"
                )
                return None

            # [STEP-5] 建立透視校正之目標映射矩陣座標 [左上, 右上, 右下, 左下]
            dst_obb_pts = np.array([
                [0, 0],
                [max_width, 0],
                [max_width, max_height],
                [0, max_height]
            ], dtype="float32")

            # [STEP-6] 計算透視變換空間投影矩陣 M 並執行影像拉平
            M = cv2.getPerspectiveTransform(rect, dst_obb_pts)
            warped_roi = cv2.warpPerspective(frame, M, (max_width, max_height))
            return warped_roi
        
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "PROCESS",
                targ = "GEOM-WARP",
                idnt = "HANDLE_GEOM_WARP_EXCEPTION",
                stat = "FAIL",
                msge = f"[任務: 幾何透視校正] 運行失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            return None