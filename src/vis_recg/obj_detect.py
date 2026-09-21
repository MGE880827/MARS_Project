# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 影像物件偵測工具 (Image Object Detection Tool)
# 維護日期: 2026-09-19
# 檔案路徑: MARS_Project/src/vis_recg/obj_detect.py
# ##########################################################################################

# 掛載外部依賴
import traceback

# 掛載內部依賴
from utils import log

# ==========================================================================================
class ObjectDetector:
    """
    [名稱] 影像物件偵測引擎 (Image Object Detection Engine)
    [作用] 對接全域組態環境，負責接收單幀影像與指定 YOLO 模型，執行物件辨識，並解析出物件區域之標準邊界框 (HBB) 或傾斜邊界框 (OBB)。
    """
    # =============================================
    def __init__(self, sys_cfg):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「影像物件偵測引擎」，並掛載演算法組態 (algo) 與路徑組態 (path)。
        [參數] sys_cfg: [dict] 全域組態檢索，內含 algo_inst 與 path_inst 組態
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg  = sys_cfg
        self.algo_cfg = self.sys_cfg["algo_inst"]
        self.path_cfg = self.sys_cfg["path_inst"]

    # =============================================
    def _parse_hbb(self, results):
        """
        [名稱] Func.B 水平邊界框解析函式
        [功能] 針對常規物件 (e.g., 車體、人體)，解析出 (x1, y1, x2, y2) 座標、類別編號與信心值。
        [參數] results: Ultralytics 辨識結果清單
        [輸出] list: 包含結構化目標字典之清單，各字典欄位詳細說明如下:
               - type   : [str] 邊界框類型，固定為 "HBB"
               - cls_id : [int] 目標類別編號 (e.g., 0: person, 2: car)
               - conf_s : [float] 辨識信心值 (range: 0.0 ~ 1.0, 保留 4 位小數)
               - bbox   : [list] 水平矩形座標 [x1, y1, x2, y2] (浮點數像素點位置)
               (e.g., Case 1: HBB 水平物件) >>> [{"type": "HBB", "cls_id": 2, "conf_s": 0.9245, "bbox": [100.5, 150.2, 300.8, 250.4]}, ...]
        """
        detections = []
        if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for cls_tensor, conf_tensor, xyxy_tensor in zip(boxes.cls, boxes.conf, boxes.xyxy):
            detections.append({
                "type"   : "HBB",
                "cls_id" : int(cls_tensor),
                "conf_s" : round(float(conf_tensor), 4),
                "bbox"   : xyxy_tensor.cpu().numpy().tolist()       # 邊界格式: [x1, y1, x2, y2]
            })
        return detections

    # =============================================
    def _parse_obb(self, results):
        """
        [名稱] Func.C 傾斜邊界框解析函式
        [功能] 針對易發生傾斜之物件 (e.g., 車牌)，解析出旋轉外接矩形之四個頂點座標 (xyxyxyxy)、類別編號與信心值。
        [參數] results: Ultralytics 辨識結果清單
        [輸出] list: 包含結構化目標字典之清單，各字典欄位詳細說明如下:
               - type   : [str] 邊界框類型，固定為 "OBB"
               - cls_id : [int] 目標類別編號 (e.g., 0: license_plate)
               - conf_s : [float] 辨識信心值 (range: 0.0 ~ 1.0, 保留 4 位小數)
               - bbox   : [list] 旋轉矩形 4 頂點座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
               (e.g., Case 2: OBB 傾斜物件) >>> [{"type": "OBB", "cls_id": 0, "conf_s": 0.9518, "bbox": [[210.5, 150.2], [350.8, 165.0], [340.2, 210.4], [200.0, 195.6]]}, ...]
        """
        detections = []
        if not results or not hasattr(results[0], "obb") or results[0].obb is None:
            return detections

        obbs = results[0].obb
        for cls_tensor, conf_tensor, xyxyxyxy_tensor in zip(obbs.cls, obbs.conf, obbs.xyxyxyxy):
            detections.append({
                "type"   : "OBB",
                "cls_id" : int(cls_tensor),
                "conf_s" : round(float(conf_tensor), 4),
                "bbox"   : xyxyxyxy_tensor.cpu().numpy().tolist(),   # 邊界格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            })
        return detections

    # =============================================
    def __call__(self, model, frame, task_rule):
        """
        [名稱] Func.D 影像物件偵測函式
        [功能] 接收已載入之 YOLO 模型與單幀影像，依上層管線傳入該領域任務規則 (內含 domain, conf_thresh, nms_thresh) 進行物件辨識，並自動判別 HBB/OBB 並回傳解析後的結構化邊界框清單。
        [參數] 共計 3 組參數，以下說明:
               - model     : [ultralytics.YOLO] 已預載入之 YOLO 模型實例
               - frame     : [numpy.ndarray] 原始全景影格矩陣 (H, W, C)，供物件特徵偵測辨識使用
               - task_rule : [dict] 該領域下辨識階層策略配置 (e.g., stage_1 或 sub_tasks["LIC"] 字典)
        [輸出] list: 包含結構化目標字典之清單，各字典欄位詳細說明如下:
               - type   : [str] 邊界框類型 ("HBB" 或 "OBB")
               - cls_id : [int] 物件類別編號 (e.g., 0: person, 2: car)
               - conf_s : [float] 辨識信心值 (range: 0.0 ~ 1.0, 保留 4 位小數)
               - bbox   : [list] 水平邊界框座標 (1D4P) [x1, y1, x2, y2] or 旋轉邊界框座標 (2D4P) [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
               (e.g., Case 1: HBB 水平物件) >>> [{"type": "HBB", "cls_id": 2, "conf_s": 0.9245, "bbox": [100.5, 150.2, 300.8, 250.4]}, ...]
               (e.g., Case 2: OBB 傾斜物件) >>> [{"type": "OBB", "cls_id": 0, "conf_s": 0.9518, "bbox": [[210.5, 150.2], [350.8, 165.0], [340.2, 210.4], [200.0, 195.6]]}, ...]
        """
        # [STEP-1] 依據內部任務規則提取參數，若上層漏傳則立即觸發 KeyError 報錯
        domain     = str(task_rule["domain"]).strip().upper()
        confidence = task_rule["conf_thresh"]
        nms_thresh = task_rule["nms_thresh"]
        stage_name = f"YOLO-I-{domain}"

        # [STEP-2] 驗證輸入參數有效性
        if model is None or frame is None:
            err_args = "model" if model is None else "frame"
            err_stat = "載入之模型為 None" if err_args == "model" else "輸入之影像為 None"
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "INVALID_ARGUMENTS",
                stat = "WARN",
                msge = f"[參數: {err_args}] 驗證警示 >>> {err_stat}，跳過物件辨識程序"
            )
            return []

        # [STEP-3] 執行辨識模型與邊界框解析
        try:
            results = model.predict(
                source  = frame,
                conf    = confidence,
                iou     = nms_thresh,
                verbose = False,
                device  = getattr(model, "device", "cpu")
            )
            # 無辨識出任何物件
            if not results or len(results) == 0:
                return []

            # 自動判別使用 OBB 還是 HBB 解析器
            if hasattr(results[0], "obb") and results[0].obb is not None:
                return self._parse_obb(results)
            else:
                return self._parse_hbb(results)

        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "INFERENCE_EXCEPTION",
                stat = "FAIL",
                msge = f"[任務: 物件辨識任務] 運行失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            return []