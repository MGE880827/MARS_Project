# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 車輛領域特徵提取工具 (Vehicle Domain Feature Extraction Tool)
# 維護日期: 2026-09-20
# 檔案路徑: MARS_Project/src/vis_recg/pipelines/veh_pipe.py
# ##########################################################################################

# 掛載外部依賴
import traceback

# 掛載內部依賴
from utils import log
from src.database import write_repo
from ..obj_detect import ObjectDetector
from ..img_proc import BodyCropper, ColorExtractor, GeomWarper, CharReader

# ==========================================================================================
class VehiclePipeline:
    """
    [名稱] 車輛領域特徵提取引擎 (Vehicle Domain Feature Extraction Engine)
    [作用] 實現車輛主領域 (VEH-HBB) 到次領域 (車牌 LIC-OBB, 廠牌 BRAND-HBB) 之影像裁切、幾何校正、字元辨識與座標空間還原完整流程。
    """
    # =============================================
    def __init__(self, sys_cfg, domain, **kwargs):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「車輛領域特徵提取引擎」，經合約校驗確認 stage_inst 策略配置正確後，動態掛載次領域模型與影像處理工具。
        [參數] 共計 3 組參數，以下說明:
               - sys_cfg : [dict] 全域組態檢索，內含 algo_inst, path_inst 與 stage_inst 組態
               - domain  : [str] 目標領域識別代碼 (e.g., "VEH")
               - kwargs  : [dict] 動態注入專屬資源 (支援 preloader、ocr_inst 等擴充實例)
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg   = sys_cfg
        self.algo_cfg  = self.sys_cfg["algo_inst"]
        self.path_cfg  = self.sys_cfg["path_inst"]
        self.stage_cfg = self.sys_cfg["stage_inst"]

        # [STEP-1] 宣告目標領域識別代碼
        self.main_domain = domain
        self.stage_name  = f"PIPE-{domain}"

        # [STEP-2] 掛載模型預載映射表
        self.preloader = kwargs.get("preloader")

        # [STEP-3] 契約前置驗證：校驗 stage_inst 結構是否滿足本領域合約標準
        self.stage_rule = self.stage_cfg.get(self.main_domain, {})
        self._validate_contract()

        # [STEP-4] 索取車牌辨識所需之 OCR 模型實例 (chinese_cht)
        ocr_inst = self.preloader.get_img_tool_model("char_ocr", lang="chinese_cht")

        # [STEP-5] 索取次領域辨識所需之 YOLO 實例 (LIC, BRAND)
        self.sub_models = {}
        sub_tasks = self.stage_rule.get("stage_2", {}).get("sub_tasks", {})
        for sub_name in sub_tasks.keys():
            model_inst = self.preloader.get_yolo_model(sub_name)
            if model_inst is not None:
                self.sub_models[sub_name] = model_inst

        # [STEP-6] 實例化該領域之影像處理工具
        self.detector   = ObjectDetector(sys_cfg=self.sys_cfg)
        self.body_crop  = BodyCropper(sys_cfg=self.sys_cfg)
        self.color_extr = ColorExtractor(sys_cfg=self.sys_cfg)
        self.geom_warp  = GeomWarper(sys_cfg=self.sys_cfg)
        self.char_read  = CharReader(sys_cfg=self.sys_cfg, ocr_inst=ocr_inst)

    # =============================================
    def _validate_contract(self):
        """
        [名稱] Func.B 策略配置合約前置檢驗
        [功能] 依據注入之 domain 代碼與次領域合約規範，靜態校驗 stage_1 與 stage_2 配置規格之欄位完整性。
        [參數] None: 自 self 讀取當前領域策略 (self.stage_rule) 與識別代碼 (self.main_domain)
        [輸出] None: 若規則不符或欄位缺失直接拋出 KeyError (Fail Fast 原則)
        """
        # 領域合約參數定義 (需更動此區塊參數)
        pipe_name   = self.__class__.__name__
        valid_sub_domains  = ["LIC", "BRAND"]
        valid_s1_key = ["domain", "conf_thresh", "nms_thresh", "cls_mapping"]
        valid_s2_key = ["domain", "bbox_type", "conf_thresh", "nms_thresh"]

        # [STEP-1] 檢驗當前領域策略規則是否存在
        if not self.stage_rule:
            raise KeyError(f"[{pipe_name}] 策略字典缺失 [{self.main_domain}] 之配置規則，終止管線初始化")

        # [STEP-2] 檢驗 Stage-1 主階段組態必要欄位
        stage_1_cfg = self.stage_rule.get("stage_1", {})
        missing_s1_keys = [k for k in valid_s1_key if k not in stage_1_cfg]
        if missing_s1_keys:
            raise KeyError(
                f"[{pipe_name}] 領域 ({self.main_domain}) 策略之 stage_1 缺失必要欄位: {missing_s1_keys}"
            )
        
        # [STEP-3] 檢驗 Stage-2 各次階段任務組態與必要欄位
        stage_2_cfg = self.stage_rule.get("stage_2", {})
        sub_tasks   = stage_2_cfg.get("sub_tasks", {})
        for sub_domain in valid_sub_domains:
            if sub_domain not in sub_tasks:
                raise KeyError(
                    f"[{pipe_name}] 領域 ({self.main_domain}) 策略之 stage_2 缺失次任務 '{sub_domain}' 設定"
                )
            sub_info = sub_tasks[sub_domain]
            missing_s2_keys = [k for k in valid_s2_key if k not in sub_info]
            if missing_s2_keys:
                raise KeyError(
                    f"[{pipe_name}] 領域 ({self.main_domain}) 之次任務 ({sub_domain}) 缺失必要欄位: {missing_s2_keys}"
                )

    # =============================================
    def _sync_infer_database(self, proc_results, frame_meta):
        """
        [名稱] Func.D 推論特徵資料庫同步函式
        [功能] 將管線提取之多階層特徵資訊展平為符合 live_veh_recg 結構之紀錄並批次持久化至資料庫。
        [參數] 共計 2 組參數，以下說明:
               - proc_results : [list] 車輛特徵提取結果清單
               - frame_meta   : [dict] 共通影格時空詮釋資料 (cam_id, vid_name, frame_idx, recg_ts)
        [輸出] tuple: (sync_rslt: dict/None, err_ret: dict/None)
               - sync_rslt : [dict/None] 成功或局部同步時回傳狀態字典 (stat_code, stat_msge, rslt_msge)，失敗為 None
               - err_ret   : [dict/None] 失敗時回傳結構化錯誤字典 (包含 500 系統錯誤與 details)，成功為 None
        """
        proj_name = self.algo_cfg.yolo_t.proj_name
        db_name   = f"{proj_name}-DB"

        if not proc_results:
            sync_rslt = {
                "stat_code" : 204,
                "stat_msge" : "NO_CONTENT",
                "rslt_msge" : f"該影格無車輛辨識目標，跳過資料庫同步作業"
            }
            return sync_rslt, None

        try:
            # [STEP-1] 嚴格提取主鍵欄位 (不設預設值，若缺漏直接觸發 KeyError 進入 500 攔截)
            cam_id    = frame_meta["cam_id"]
            vid_name  = frame_meta["vid_name"]
            frame_idx = frame_meta["frame_idx"]
            recg_ts   = frame_meta["recg_ts"]

            cls_mapping = self.stage_rule.get("stage_1", {}).get("cls_mapping", {})
            veh_recg_records = []

            # [STEP-2] 展開特徵並對齊 SQL 欄位
            for idx, item in enumerate(proc_results):
                # 解析主領域特徵 (VEH)
                main_obj_idx  = f"{self.main_domain}_{idx+1:03d}"
                main_cls_id   = item.get("cls_id")
                main_cls_name = cls_mapping.get(main_cls_id, "UNK")
                main_color    = item.get("feature", {}).get("color", "UNK")
                main_conf_s   = item.get("conf_s", 0.0)

                # 解析次領域特徵 (LIC, BRAND)
                sub_plate_num  = "UNK"
                sub_plate_attr = "UNK"
                sub_brand_name = "UNK"
                for sub_info in item.get("sub_detections", []):
                    sub_domain  = sub_info.get("domain")
                    sub_feature = sub_info.get("feature", {})
                    if sub_domain == "LIC":
                        sub_plate_num  = sub_feature.get("char_text")
                        sub_plate_attr = sub_feature.get("char_attr")
                    elif sub_domain == "BRAND":
                        sub_brand_name = sub_feature.get("brand_name")

                veh_recg_records.append({
                    "cam_id"     : cam_id,           # 攝影機定址代碼 (<CAM_BRAND> e.g., "ACEPRO", "CAM_01")
                    "vid_name"   : vid_name,         # 來源影片檔案名稱 (e.g., "INFER_ACEPRO_VEH_ROOF_20260705_120000.mp4")
                    "frame_idx"  : frame_idx,        # 影片當前影格序號 (curr_frms)
                    "recg_ts"    : recg_ts,          # 辨識影格實體時間戳記 (精準對應影格時間)
                    "veh_id"     : main_obj_idx,     # 該影格追蹤識別碼 (e.g., "VEH_00001")
                    "veh_type"   : main_cls_name,    # Stage-1 車輛種類，未辨識出為 "UNK" (e.g., "car", "truck", "bus", "motorcycle")
                    "plate_num"  : sub_plate_num,    # Stage-2 車牌號碼，未辨識出為 "UNK" (e.g., "ABC1234", "軍C1234")
                    "plate_attr" : sub_plate_attr,   # Stage-2 車牌類別，未辨識出為 "UNK" (e.g., "STANDARD", "電動車", "軍車", "試車牌")
                    "veh_brand"  : sub_brand_name,   # Stage-2 車輛廠牌，未辨識出為 "UNK" (e.g., "toyota", "benz")
                    "veh_color"  : main_color,       # 車輛外觀顏色，預設 "UNK"
                    "conf_score" : main_conf_s,      # 演算法綜合辨識置信度 (0.0 ~ 1.0)
                    "crop_path"  : "UNK"             # 局部車牌拉直或車體特徵影像儲存路徑 [NOTE] 後續開發，目前均在記憶體中處理
                })

            # [STEP-3] 批次寫入資料庫與狀態審計
            veh_recg_stat = write_repo.infer.upsert_veh_recg_logs(veh_recg_records)
            if veh_recg_stat:
                log.CONTENT(
                    type = "DATABASE",
                    targ = self.stage_name,
                    idnt = "SYNC_INFER_DATABASE",
                    stat = "SUCC",
                    msge = f"[資料庫: {db_name}] 寫入成功 >>> 共 {len(veh_recg_records)} 筆車輛辨識數據「完全同步」至 live_veh_recg 資料表中"
                )
                sync_rslt = {
                    "stat_code" : 200,
                    "stat_msge" : "COMPLETED",
                    "rslt_msge" : f"車輛辨識與數據寫入同步圓滿完成"
                }
            else:
                log.CONTENT(
                    type = "DATABASE",
                    targ = self.stage_name,
                    idnt = "SYNC_INFER_DATABASE",
                    stat = "WARN",
                    msge = f"[資料庫: {db_name}] 寫入警示 >>> 車輛辨識數據「局部同步」或「底層連線異常」，請調閱資料表紀錄"
                )
                sync_rslt = {
                    "stat_code" : 206,
                    "stat_msge" : "PARTIAL_SUCCESS",
                    "rslt_msge" : f"車輛辨識完成，但數據寫入過程發生局部遺失"
                }

            return sync_rslt, None

        except Exception as e:
            # 資料庫連線或約束異常攔截 (推送業務錯誤 FAIL，精簡印出一行錯誤原因)
            log.CONTENT(
                type = "DATABASE",
                targ = self.stage_name,
                idnt = "SYNC_INFER_DATABASE",
                stat = "FAIL",
                msge = f"[資料庫: {db_name}] 寫入失敗 >>> 車輛辨識數據同步發生異常: {e}"
            )
            err_ret = {
                "stat_code" : 500,
                "stat_msge" : "DATABASE_SYNC_ERROR",
                "rslt_msge" : f"[{self.main_domain}] 車輛辨識數據同步寫入失敗: {e}",
                "details"   : {
                    "domain"     : self.main_domain,
                    "frame_meta" : frame_meta,
                    "records_qty": len(proc_results)
                }
            }
            return None, err_ret

    # =============================================
    def __call__(self, frame, detections, frame_meta):
        """
        [名稱] Func.C 車輛領域特徵提取函式
        [功能] 採雙層階層架構，以車輛為主領域 (VEH-HBB)、車牌 (LIC-OBB) 與廠牌 (BRAND-HBB/OBB) 為次領域，依序進行「次領域物件辨識」及「調度影像處理工具」。
        [流程] 車輛裁切 >> 次領域辨識 >> 幾何透視校正/文字辨識/特徵映射 >> 座標全域還原
        [參數] 共計 3 組參數，以下說明:
               - frame       : [numpy.ndarray] 原始全景影格矩陣 (H, W, C)
               - detections  : [list] 第一階層車輛物件偵測產出之目標清單
               - frame_meta  : [dict] 共通影格時空詮釋資料 (cam_id, vid_name, frame_idx, recg_ts)
        [輸出] tuple: (proc_results: list, sync_rslt: dict/None, err_ret: dict/None)
               - proc_results : [list] 整合車輛與次物件 (車牌、廠牌) 多層級特徵之結構化清單，以下說明:
                 - domain         : [str] 主物件識別代碼 (e.g., "VEH")
                 - type           : [str] 主物件邊界框類型 (e.g., "HBB")
                 - cls_id         : [int] 主物件類別編號 (e.g., 0: "car", 1: "truck", 2: "bus")
                 - conf_s         : [float] 主物件辨識信心值 (range, 0.0 ~ 1.0 保留 4 位小數)
                 - bbox           : [list] 主物件全景水平邊界框座標 (1D4P) [x1, y1, x2, y2]
                 - feature        : [dict] 主物件領域特徵字典 {"color": "black", ....}
                   - color        : [str] 車體主要顏色 (e.g., "black", "white")
                 - sub_detections : [list] 次物件辨識成果結構化清單，內含欄位如下:
                   - domain       : [str] 次物件識別代碼 (e.g., "LIC", "BRAND")
                   - type         : [str] 次物件邊界框類型 (e.g., "OBB")
                   - cls_id       : [int] 次物件類別編號 (e.g., 0: license_plate)
                   - conf_s       : [float] 次物件辨識信心值 (range, 0.0 ~ 1.0 保留 4 位小數)
                   - l_bbox       : [list] 局部裁切圖之旋轉邊界框座標 (2D4P) [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                   - g_bbox       : [list] 還原至全圖之旋轉邊界框座標 (2D4P) [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                   - feature      : [dict] 次物件領域特徵字典，依次領域區分如下:
                       # <LIC 領域>
                       - char_text: [str] 車牌字元辨識結果 (e.g., "ABC-1234", 辨識失敗為 "UNK")
                       - char_attr: [str] 車牌屬性識別 (e.g., "一般", "電動", "租賃", 預設為 "UNK")
                       - char_conf: [float] 車牌字元辨識信心值 (range: 0.0 ~ 1.0, 保留 4 位小數)
                       # <BRAND 領域>
                       - brand_name: [str] 車輛廠牌識別名稱 (e.g., "toyota", "benz", "bmw", 預設為 "UNK")
                       - brand_conf: [float] 廠牌分類識別信心值 (range: 0.0 ~ 1.0, 保留 4 位小數)
                (e.g., Case 1: VEH 階層式輸出) >>> [{
                      "domain": "VEH", "type": "HBB", "cls_id": 2, "conf_s": 0.9245, "bbox": [100.0, 150.0, 400.0, 350.0],
                      "feature": {"color": "black"},
                      "sub_detections": [{
                         "domain": "LIC", "type": "OBB", "cls_id": 0, "conf_s": 0.9518,
                         "l_bbox": [[10.0, 20.0], [90.0, 20.0], [90.0, 50.0], [10.0, 50.0]],
                         "g_bbox": [[110.0, 170.0], [190.0, 170.0], [190.0, 200.0], [110.0, 200.0]],
                         "feature": {"char_text": "ABC-1234", "char_attr": "一般", "char_conf": 0.9852}
                      }]
                    }, ...]
               - sync_rslt    : [dict/None] 成功或局部同步時回傳狀態字典 (stat_code, stat_msge, rslt_msge)，失敗為 None
               - err_ret      : [dict/None] 失敗時回傳結構化錯誤字典 (包含 500 系統錯誤與 details)，成功為 None
        """
        # 驗證輸入參數有效性
        if frame is None:
            log.CONTENT(
                type = "PIPE",
                targ = self.stage_name,
                idnt = "INVALID_FRAME_MATRIX",
                stat = "WARN",
                msge = f"[領域: {self.main_domain}] 驗證警示 >>> 輸入之原始全景影格 (frame) 為 None，跳過特徵提取程序"
            )
            err_ret = {
                "stat_code" : 400,
                "stat_msge" : "INVALID_FRAME_MATRIX",
                "rslt_msge" : f"[{self.main_domain}] 輸入影格資料為 None，無法進行特徵提取流程",
                "details"   : {
                    "domain"     : self.main_domain,
                    "frame_meta" : frame_meta
                }
            }
            return [], None, err_ret

        if not isinstance(frame_meta, dict):
            log.CONTENT(
                type = "PIPE",
                targ = self.stage_name,
                idnt = "VALIDATE_FRAME_META",
                stat = "WARN",
                msge = f"[領域: {self.main_domain}] 驗證警示 >>> 未提供合法影格詮釋資料字典 (frame_meta)，收訖型態: {type(frame_meta)}"
            )
            err_ret = {
                "stat_code" : 400,
                "stat_msge" : "INVALID_FRAME_META",
                "rslt_msge" : f"[{self.main_domain}] 缺少必要之時空詮釋資料字典，請檢查上層調度器傳參",
                "details"   : {
                    "domain"      : self.main_domain,
                    "actual_type" : str(type(frame_meta))
                }
            }
            return [], None, err_ret

        if not detections:
            sync_rslt = {
                "stat_code" : 204,
                "stat_msge" : "NO_CONTENT",
                "rslt_msge" : "該影格第一階層無偵測目標，跳過後續特徵處理"
            }
            return [], sync_rslt, None

        proc_results = []
        main_domain = self.main_domain
        sub_tasks   = self.stage_rule["stage_2"]["sub_tasks"]

        for idx, main_obj in enumerate(detections):
            main_cls_id, main_bbox = None, None
            main_feature_data = {}

            """ [STAGE-1] 主領域執行階段 (VEH) """
            try:
                # [STEP-1] 提取主物件辨識資料 (車輛)
                main_type   = main_obj.get("type")
                main_cls_id = main_obj.get("cls_id")
                main_conf_s = main_obj.get("conf_s")
                main_bbox   = main_obj.get("bbox")
                if main_cls_id is None or main_bbox is None or len(main_bbox) != 4:
                    continue
                w = float(main_bbox[2]) - float(main_bbox[0])
                h = float(main_bbox[3]) - float(main_bbox[1])
                if w < 0 or h < 0:
                    continue
                
                # [STEP-2] 執行主物件切裁，並提取全域座標偏移量
                main_crop = self.body_crop(frame, main_bbox)
                offset_x  = int(round(float(main_bbox[0])))
                offset_y  = int(round(float(main_bbox[1])))

                # [STEP-3] 執行主物件特徵辨識
                main_color = self.color_extr(main_crop)
                main_feature_data["color"] = main_color
                main_item = {
                    "domain"         : main_domain,
                    "type"           : main_type,
                    "cls_id"         : main_cls_id,
                    "conf_s"         : main_conf_s,
                    "bbox"           : main_bbox,
                    "feature"        : main_feature_data,
                    "sub_detections" : []
                }

                """ [STAGE-2] 次領域執行階段 (LIC, BRAND) """
                sub_detections = []
                if main_crop is not None and self.sub_models:
                    for sub_domain in sub_tasks.keys():
                        sub_task_rule = sub_tasks.get(sub_domain, {})
                        if not sub_task_rule:
                            continue

                        # 依領域名稱獲取對應之專屬模型實例
                        sub_model_inst = self.sub_models.get(sub_domain)
                        if sub_model_inst is None:
                            continue

                        try:
                            # [STEP-1] 提取次物件辨識資料 (車牌、廠牌) >>> 物件影像偵測引擎
                            sub_objs = self.detector(
                                model     = sub_model_inst,
                                frame     = main_crop,
                                task_rule = sub_task_rule
                            )
                            for sub_obj in sub_objs:
                                sub_feature_data = {}
                                sub_type   = sub_obj.get("type")
                                sub_cls_id = sub_obj.get("cls_id")
                                sub_conf_s = sub_obj.get("conf_s")
                                sub_l_bbox = sub_obj.get("bbox")

                                if sub_cls_id is None or sub_l_bbox is None:
                                    continue

                                if sub_domain == "LIC":
                                    """ [STAGE-3] 車牌領域影像處理 (LIC) """
                                    # 邊界座標: OBB-旋轉邊界框座標 (2D4P) [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                    # 影像處理: 幾何透視校正、光學字元辨識
                                    if sub_type != "OBB":
                                        continue

                                    # [STEP-1] 幾何透視校正
                                    lic_warp = self.geom_warp(main_crop, sub_l_bbox)

                                    # [STEP-2] 光學字元辨識
                                    char_text, char_attr, char_conf = "UNK", "UNK", 0.0
                                    if lic_warp is not None:
                                        char_rslt = self.char_read(lic_warp, sub_domain)
                                        if isinstance(char_rslt, dict):
                                            char_text = char_rslt["char_text"]
                                            char_attr = char_rslt["char_attr"]
                                            char_conf = char_rslt["char_conf"]

                                    # [STEP-3] 局部座標還原全域座標 (OBB: 2D4P)
                                    sub_g_bbox = [
                                        [round(float(pt[0]) + offset_x, 2), round(float(pt[1]) + offset_y, 2)]
                                        for pt in sub_l_bbox
                                    ]

                                    # [STEP-4] 提取該領域特徵資訊
                                    sub_feature_data = {
                                        "char_text" : char_text,
                                        "char_attr" : char_attr,
                                        "char_conf" : char_conf
                                    }

                                elif sub_domain == "BRAND":
                                    """ [STAGE-4] 廠牌領域影像處理 (BRAND) """
                                    # 邊界座標: HBB-水平邊界框座標 (1D4P) [x1, y1, x2, y2]
                                    # 影像處理: 無
                                    if sub_type != "HBB" or len(sub_l_bbox) != 4:
                                        continue

                                    # [STEP-1] 局部座標還原全域座標 (HBB: 1D4P)
                                    sub_g_bbox = [
                                        round(float(sub_l_bbox[0] + offset_x), 2),
                                        round(float(sub_l_bbox[1] + offset_y), 2),
                                        round(float(sub_l_bbox[2] + offset_x), 2),
                                        round(float(sub_l_bbox[3] + offset_y), 2)
                                    ]

                                    # [STEP-2] 提取該領域特徵資訊
                                    sub_feature_data = {
                                        "brand_name" : sub_task_rule.get("cls_mapping", {}).get(sub_cls_id, "UNK"),
                                        "brand_conf" : sub_conf_s
                                    }

                                else:
                                    continue

                                sub_detections.append({
                                    "domain"  : sub_domain,
                                    "type"    : sub_type,
                                    "cls_id"  : sub_cls_id,
                                    "conf_s"  : sub_conf_s,
                                    "l_bbox"  : sub_l_bbox,
                                    "g_bbox"  : sub_g_bbox,
                                    "feature" : sub_feature_data
                                })

                        except Exception:
                            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
                            sys_err = traceback.format_exc()
                            log.CONTENT(
                                type = "PROCESS",
                                targ = f"PIPE-{main_domain}",
                                idnt = f"SUB_{sub_domain}_EXCEPTION",
                                stat = "WARN",
                                msge = f"[任務: 次物件影像辨識] 運行失敗 >>> 車輛資訊 (#{idx:02d}) 執行次任務 ({sub_domain}) 發生非預期異常，跳過該項，堆疊資訊如下: \n{sys_err}"
                            )
                            continue

                main_item["sub_detections"] = sub_detections
                proc_results.append(main_item)

            except Exception:
                # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
                sys_err = traceback.format_exc()
                log.CONTENT(
                    type = "PROCESS",
                    targ = self.stage_name,
                    idnt = "MAIN_VEHICLE_EXCEPTION",
                    stat = "WARN",
                    msge = f"[任務: 車輛影像處理] 運行失敗 >>> 車輛資訊 (#{idx:02d}, cls_id={main_cls_id}) 發生非預期異常，跳過該車輛處理，堆疊資訊如下: \n{sys_err}"
                )
                continue

        """ [STAGE-3] 特徵資料庫同步寫入階段 """
        sync_rslt, err_ret = self._sync_infer_database(proc_results=proc_results, frame_meta=frame_meta)
        return proc_results, sync_rslt, err_ret
    
# ==========================================================================================
# ⭐｜領域管線工具導出｜對外註冊接口
# ==========================================================================================
PIPELINE_TOOLS = {
    "VEH": VehiclePipeline
}