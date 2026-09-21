# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 階層級聯協調工具 (Hierarchical Cascade Coordination Tool)
# 維護日期: 2026-09-22
# 檔案路徑: MARS_Project/src/modl_opts/stage_coord.py
# ##########################################################################################

# 掛載外部依賴
import cv2
import json
import yaml
import shutil
import traceback
import numpy as np
from pathlib import Path

# 掛載內部依賴
from utils import log
from utils.env import input_resolver

# ==========================================================================================
class StageCoordinator:
    """
    [名稱] 階層級聯協調引擎 (Hierarchical Cascade Coordination Engine)
    [作用] 對接「全域階層策略」，專職統籌雙階層架構之資料級聯轉換：
           1. 針對主階層全景圖片，對主物件實施「邊界安全外擴裁切」 (Safe Cropping)
           2. 實施次目標「坐標動態投影」與邊界「限幅校正」 (HBB / OBB Coordinate Projection)
           3. 將次目標命名空間執行「類別編號平移歸零」 (Class ID Remapping)
           4. 自動生成「次階獨立訓練組態」與「全景血統溯源檔」 (data.yaml & sub_obj_lineage.json)
    """
    # =============================================
    def __init__(self, sys_cfg):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「階層級聯協調引擎」，掛載全域路徑組態、演算法組態及階層策略字典。
        [參數] sys_cfg: [dict] 全域組態檢索，內含 path_inst, algo_inst 與 stage_inst 組態
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg   = sys_cfg
        self.path_cfg  = sys_cfg["path_inst"]
        self.algo_cfg  = sys_cfg["algo_inst"]
        self.stage_cfg = sys_cfg.get("stage_inst", {})

    # =============================================
    def _process_dataset_split(self, domain_dir, sub_out_dir, split, stage_rule, lineage_records):
        """
        [名稱] Func.B 單一劃分集切片處理函式
        [功能] 遍歷 train/valid/test 劃分集，執行主物件邊界裁切、次物件坐標投影與平移標籤。
        [參數] 共計 5 組參數，以下說明:
               - domain_dir      : [str/Path] 主領域物件標註資料集 (e.g., roboflow/VEH)
               - sub_out_dir     : [str/Path] 次領域物件標註資料集 (e.g., roboflow/VEH/sub)
               - split           : [str] 全景資料劃分集名稱 ('train', 'valid', 'test')
               - stage_rule      : [dict] 該領域階層策略字典
               - lineage_records : [list] 資料血統溯源紀錄累加清單
        [輸出] dict: 內含單一劃分集轉換統計之結構化字典，共計 2 組鍵值，以下說明:
               - crops_qty      : [int] 該劃分集所產出之主物件安全裁切影像數量
               - sub_labels_qty : [int] 該劃分集完成投影校正與類別平移之次目標標註實例總數
               - Remark-1: 透過 Python 可變物件特性，lineage_records 於函式內以 in-place 方式直接累加，呼叫端共享記憶體
        """
        # 全景資料劃分集路徑
        split_img_dir = Path(domain_dir) / split / "images"
        split_lbl_dir = Path(domain_dir) / split / "labels"

        split_stats = {"crops_qty": 0, "sub_labels_qty": 0}
        if not split_img_dir.exists():
            return split_stats

        cls_name_to_id = {}

        # 動態掛載 Stage-1 主階層名稱
        stage_1_cfg  = stage_rule["stage_1"]
        main_cls_ids = set(stage_1_cfg.get("gt_cls_ids", []))
        for cls_name, global_cls_id in zip(stage_1_cfg.get("cls_names", []), stage_1_cfg.get("gt_cls_ids", [])):
            cls_name_to_id[cls_name.lower().strip()] = global_cls_id

        # 動態掛載 Stage-2 次階層名稱與平移表 (全域 Global 類別 ID -> 區域 Local 類別 ID)
        transition   = stage_rule.get("transition", {})
        crop_margin  = transition.get("crop_margin", 0.05)
        stage_2_cfg  = stage_rule.get("stage_2", {})
        sub_tasks    = stage_2_cfg.get("sub_tasks", {})

        cls_id_remap = {}
        for sub_domain, sub_info in sub_tasks.items():
            for local_cls_id, (cls_name, global_cls_id) in enumerate(zip(sub_info.get("cls_names", []), sub_info.get("gt_cls_ids", []))):
                cls_name_to_id[cls_name.lower().strip()] = global_cls_id
                cls_id_remap[global_cls_id] = (sub_domain, local_cls_id)

        # 讀取 Roboflow 該領域的 data.yaml 取得 names 陣列
        robo_data_path = Path(domain_dir) / "data.yaml"
        robo_cls_names = []
        if robo_data_path.exists():
            with open(robo_data_path, "r", encoding="utf-8") as yaml_file:
                yaml_cfg = yaml.safe_load(yaml_file)
                robo_cls_names = [str(cls_name).lower().strip() for cls_name in yaml_cfg.get("names", [])]

        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        for img_path in sorted(split_img_dir.iterdir()):
            # 遍歷檢索「全景資料劃分集」影像與標註檔案 (Roboflow 下 train, valid, test)
            if not (img_path.is_file() and img_path.suffix.lower() in valid_extensions):
                continue
            lbl_path = split_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    obj_lbls = [l.strip() for l in f if l.strip()]
            except Exception:
                continue

            # 主物件座標資訊分流篩選
            lbl_items = [self._parse_object_label(label, robo_cls_names, cls_name_to_id) for label in obj_lbls]
            lbl_items = [item for item in lbl_items if item is not None]
            main_objs = [item for item in lbl_items if item["cls_id"] in main_cls_ids]
            sub_objs  = [item for item in lbl_items if item["cls_id"] in cls_id_remap]

            if not main_objs:
                continue
            # 載入全景影像像素資料矩陣
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            orig_h, orig_w = img_bgr.shape[:2]

            """ [STAGE-1] 主物件全景邊界定位與安全外擴裁切 """
            for crop_idx, main_obj in enumerate(main_objs):
                main_coords = main_obj["coords"]
                totl_coords = len(main_coords)
                # [STEP-1] 主物件座標資訊定位
                if main_obj["bbox_type"] == "HBB":
                    # 水平框(HBB)-座標定位 [xc, yc, w, h]
                    main_x_center = main_coords[0] * orig_w
                    main_y_center = main_coords[1] * orig_h
                    main_bbox_w   = main_coords[2] * orig_w
                    main_bbox_h   = main_coords[3] * orig_h
                    main_x_min, main_x_max = main_x_center - (main_bbox_w / 2.0), main_x_center + (main_bbox_w / 2.0)
                    main_y_min, main_y_max = main_y_center - (main_bbox_h / 2.0), main_y_center + (main_bbox_h / 2.0)

                elif main_obj["bbox_type"] == "OBB":
                    # 旋轉框(OBB)-座標定位 [x1, y1, ..., x4, y4]
                    main_pts_x = [main_coords[i] * orig_w for i in range(0, totl_coords, 2)]
                    main_pts_y = [main_coords[i+1] * orig_h for i in range(0, totl_coords, 2)]
                    main_x_min, main_x_max = min(main_pts_x), max(main_pts_x)
                    main_y_min, main_y_max = min(main_pts_y), max(main_pts_y)
                    main_bbox_w = main_x_max - main_x_min
                    main_bbox_h = main_y_max - main_y_min

                # [STEP-2] 限幅安全外擴邊界計算
                expand_w = main_bbox_w * crop_margin
                expand_h = main_bbox_h * crop_margin
                crop_x_min, crop_x_max = max(0, int(np.floor(main_x_min - expand_w))), min(orig_w, int(np.ceil(main_x_max + expand_w)))
                crop_y_min, crop_y_max = max(0, int(np.floor(main_y_min - expand_h))), min(orig_h, int(np.ceil(main_y_max + expand_h)))
                
                # [STEP-3] 主物件影像裁切幾何檢驗
                if (crop_x_max - crop_x_min) < 8 or (crop_y_max - crop_y_min) < 8:
                    # [NOTE] 裁切寬高若小於 8 像素屬於無效退化區域或嚴重變形，不具備特徵辨識價值予以捨棄
                    continue
                crop_patch = img_bgr[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                crop_stem  = f"{img_path.stem}_crop{crop_idx:03d}"
                crop_frame = f"{crop_stem}.jpg"

                """ [STAGE-2] 次物件坐標投影校正與類別編號平移 """
                sub_domain_lbls = {sub_domain: [] for sub_domain in sub_tasks.keys()}
                for sub_obj in sub_objs:
                    # [STEP-1] 次物件類別編號命名空間平移 (Global ID -> Local ID)
                    global_cls_id = sub_obj["cls_id"]
                    sub_domain, local_cls_id = cls_id_remap[global_cls_id]

                    # [STEP-2] 次物件座標投影校正 (全景坐標 -> 裁切切片相對坐標)
                    proj_coords = self._project_coords(
                        coords    = sub_obj["coords"],
                        bbox_type = sub_obj["bbox_type"],
                        orig_wh   = (orig_w, orig_h),
                        crop_rect = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
                    )
                    if proj_coords is not None:
                        coords_str = " ".join(f"{coord:.6f}" for coord in proj_coords)
                        sub_domain_lbls[sub_domain].append(f"{local_cls_id} {coords_str}")
                
                """ [STAGE-3] 次物件存在性檢驗與級聯資料集儲存 """
                crop_has_sub = any(len(lbls) > 0 for lbls in sub_domain_lbls.values())
                if crop_has_sub:
                    split_stats["crops_qty"] += 1
                    # 各次領域各自儲存對應之影像與標籤 (sub/<SUB_DOMAIN>/images/<split> 與 labels/<split>)
                    # 確保只有真正含有該次領域目標的切片，才會納入該次領域之獨立訓練集
                    save_sub_domains = []
                    for sub_domain, obj_lbls in sub_domain_lbls.items():
                        if not obj_lbls:
                            continue
                        
                        # [STEP-1] 次領域訓練集目錄建構
                        sub_split_img_dir = Path(sub_out_dir) / sub_domain / split / "images"
                        sub_split_lbl_dir = Path(sub_out_dir) / sub_domain / split / "labels"
                        sub_split_img_dir.mkdir(parents=True, exist_ok=True)
                        sub_split_lbl_dir.mkdir(parents=True, exist_ok=True)

                        # [STEP-2] 主物件局部裁切影像輸出儲存
                        # 輸出局部特徵切片作為次階專屬訓練底圖，排除全景大圖無關背景雜訊
                        cv2.imwrite(str(sub_split_img_dir / crop_frame), crop_patch)

                        # [STEP-3] 次物件投影坐標與標籤文字檔寫入
                        # 寫入平移歸零後的次領域局部標註，完全符合 YOLO 獨立資料集訓練規範
                        with open(sub_split_lbl_dir / f"{crop_stem}.txt", "w", encoding="utf-8") as f:
                            f.write("\n".join(obj_lbls) + "\n")

                        split_stats["sub_labels_qty"] += len(obj_lbls)
                        save_sub_domains.append(sub_domain)
                    
                    # [STEP-4] 資料血統溯源紀錄
                    lineage_records.append({
                        "crop_filename" : crop_frame,
                        "parent_image"  : img_path.name,
                        "split"         : split,
                        "sub_domain"    : save_sub_domains,
                        "crop_rect"     : [crop_x_min, crop_y_min, crop_x_max, crop_y_max],
                        "orig_wh"       : [orig_w, orig_h]
                    })
        
        return split_stats

    # =============================================
    def _parse_object_label(self, label, robo_cls_names=None, cls_name_to_id=None):
        """
        [名稱] Func.B-1 單筆標註幾何解析函式
        [功能] 將單行 YOLO 格式標籤拆解並動態轉譯類別識別，自動判定幾何形態：
               1. 藉由 data.yaml 索引映射反查類別名稱，動態對齊全域 ID (Global ID)。
               2. 嚴格過濾未定義於階層策略之無效或異常標籤 (返回 None)。
               3. 辨別 HBB-水平框 (4項) 與 OBB-旋轉框 (8項)，並相容 Roboflow 首尾閉合之 10 項多邊形坐標。
        [參數] 共計 3 組參數，以下說明:
               - label          : [str] YOLO 格式單筆標註文字字串
               - robo_cls_names : [list/None] Roboflow 標註中介資料 (data.yaml) 定義之類別名稱清單
               - cls_name_to_id : [dict/None] 階層策略字典動態建置之「類別名稱」-> 「全域 ID」映射表
        [輸出] dict/None: 內含幾何解析結果之結構化字典，若傳入為空行或純空白字串則回傳 None，共計 3 組鍵值，以下說明:
               - cls_id    : [int] 物件類別識別編號 (Class ID)
               - bbox_type : [str/None] 邊界框類型 ("HBB": 水平框, "OBB": 旋轉框, None: 非標準或未定義格式)
               - coords    : [list] 正規化幾何點位清單 (HBB 為 4 項 [xc, yc, w, h]，OBB 為 8 項 [x1, y1, ..., x4, y4])
        """
        parts = label.split()
        if not parts:
            return None
        lbl_cls_id = int(parts[0])

        # 嚴格校驗：必須存在於 data.yaml 的 names 且必須定義在 stage_rule 的對照表內
        if robo_cls_names and cls_name_to_id and lbl_cls_id < len(robo_cls_names):
            cls_name = robo_cls_names[lbl_cls_id]
            # 標註標籤未定義在 stage_rule 階層策略配置中，無效略過
            if cls_name not in cls_name_to_id:
                return None
            global_cls_id = cls_name_to_id[cls_name]
        else:
            # 無法映射的異常標籤，直接剔除
            return None

        coords = [float(c) for c in parts[1:]]
        
        if len(coords) == 4:
            bbox_type = "HBB"
        elif len(coords) >= 8:
            bbox_type = "OBB"
            # 若為 10 個座標 (Roboflow 多邊形首尾閉合)，保留前 8 個數值作為 4 頂點旋轉框
            if len(coords) == 10 and coords[0] == coords[8] and coords[1] == coords[9]:
                coords = coords[:8]
            elif len(coords) > 8:
                coords = coords[:8]
        else:
            return None
        
        lbl_item = {
            "cls_id"    : global_cls_id,
            "bbox_type" : bbox_type,
            "coords"    : coords
        }
        return lbl_item

    # =============================================
    def _project_coords(self, coords, bbox_type, orig_wh, crop_rect):
        """
        [名稱] Func.B-2 次物件坐標投影校正函式
        [功能] 以主物件裁切左上角為原點，依相對位置將次物件換算為切片正規化坐標並實施限幅。
        [參數] 共計 4 組參數，以下說明:
               - coords    : [list] 次物件全景相對坐標 (0.0~1.0)
               - bbox_type : [str] 次物件邊界框類型 ("HBB": 水平框, "OBB": 旋轉框)
               - orig_wh   : [tuple] 全景圖原始寬高像素維度 (orig_w, orig_h)
               - crop_rect : [tuple] 主物件裁切像素邊界 (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
        [輸出] list/None: 投影後之局部正規化相對坐標；若中心點落於「裁切視窗外」或「不明邊界框類型」則回傳 None
               - e.g. HBB 為 4 項 [xc, yc, w, h]
               - e.g. OBB 為 8 項 [x1, y1, ..., x4, y4]
        """
        # 全景圖原始寬高
        orig_w, orig_h = orig_wh
        # 主物件裁切圖素邊界＆寬高
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop_rect
        crop_w = crop_x_max - crop_x_min
        crop_h = crop_y_max - crop_y_min

        if crop_w <= 0 or crop_h <= 0:
            return None
        
        if bbox_type == "HBB":
            sub_x_center = coords[0] * orig_w
            sub_y_center = coords[1] * orig_h
            sub_bbox_w   = coords[2] * orig_w
            sub_bbox_h   = coords[3] * orig_h

            # 次物件中心點幾何檢驗，必須落於主物件裁切視窗內部
            is_center_in_crop = (
                crop_x_min <= sub_x_center <= crop_x_max and
                crop_y_min <= sub_y_center <= crop_y_max
            )
            if not is_center_in_crop:
                return None

            # 水平框(HBB)-座標校正回歸
            proj_x_center = (sub_x_center - crop_x_min) / crop_w
            proj_y_center = (sub_y_center - crop_y_min) / crop_h
            proj_w = min(sub_bbox_w / crop_w, 1.0)
            proj_h = min(sub_bbox_h / crop_h, 1.0)
            proj_hbb = [
                max(0.0, min(1.0, proj_x_center)),
                max(0.0, min(1.0, proj_y_center)),
                max(0.0, min(1.0, proj_w)),
                max(0.0, min(1.0, proj_h)),
            ]
            return proj_hbb

        elif bbox_type == "OBB":
            sub_pts = [(coords[i] * orig_w, coords[i+1] * orig_h) for i in range(0, len(coords), 2)]
            sub_center_x = sum(pt[0] for pt in sub_pts) / len(sub_pts)
            sub_center_y = sum(pt[1] for pt in sub_pts) / len(sub_pts)

            # 次物件中心點幾何檢驗，必須落於主物件裁切視窗內部
            is_center_in_crop = (
                crop_x_min <= sub_center_x <= crop_x_max and
                crop_y_min <= sub_center_y <= crop_y_max
            )
            if not is_center_in_crop:
                return None

            # 旋轉框(OBB)-座標校正回歸
            proj_coords = []
            for sub_pts_x, sub_pts_y in sub_pts:
                proj_pts_x = max(0.0, min(1.0, (sub_pts_x - crop_x_min) / crop_w))
                proj_pts_y = max(0.0, min(1.0, (sub_pts_y - crop_y_min) / crop_h))
                proj_coords.extend([proj_pts_x, proj_pts_y])

            return proj_coords
        
        else:
            return None

    # =============================================
    def _generate_sub_yaml(self, sub_domain_dir, sub_domain, task_cfg):
        """
        [名稱] Func.C 次階訓練組態生成函式
        [功能] 依據次任務規格自動產出合規之次階專屬訓練設定檔 (data.yaml)。
        [參數] 共計 3 組參數，以下說明:
               - sub_domain_dir : [Path] 次領域專屬根目錄 (e.g., roboflow/VEH/sub/LIC)
               - sub_domain     : [str] 次領域識別代碼 (e.g., 'LIC')
               - task_cfg       : [dict] 該次領域專屬任務規格配置 (包含 cls_names, gt_cls_ids, imgsz 等規格)
        [輸出] Path: 產出之 data.yaml 實體路徑
        """
        data_yaml_path = sub_domain_dir / "data.yaml"
        cls_names = task_cfg.get("cls_names", [])

        # 遵循 YOLO 鏡像尋找機制 (<split>/images 與 <split>/labels)
        sub_root_path = sub_domain_dir.resolve()
        yaml_content = {
            "path"  : str(sub_root_path),
            "train" : "train/images",
            "val"   : "valid/images",
            "test"  : "test/images",
            "nc"    : len(cls_names),
            "names" : cls_names
        }

        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False, allow_unicode=True)

        return data_yaml_path

    # =============================================
    def __call__(self, domain):
        """
        [名稱] Func.D 階層級聯協調調度函式
        [功能] 驅動階層級聯協調管線生命週期，統籌策略驗證、全景安全裁切、次目標投影校正與血統溯源檔案建置。
        [參數] domain: [str] 主領域識別代碼 (e.g., VEH, MARA)
        [輸出] dict: 內含執行結果之結構化字典，共計 4 組鍵值，以下說明:
               - stat_code : [int] 執行狀態代碼 (200-成功, 304-單階略過, 400-策略缺失, 404-目錄缺失, 500-系統異常)
               - stat_msge : [str] 狀態識別標籤 (e.g., COMPLETED, BYPASS_SINGLE_STAGE, MISSING_STAGE_RULE, MISSING_ROBOFLOW_DATASET, SYSTEM_ERROR)
               - rslt_msge : [str] 執行結果之詳細文字說明
               - details   : [dict] 共計 5 組任務指標明細，以下說明:
                 - domain        : [str] 主領域識別代碼
                 - stage_qty     : [int] 模型架構階層數 (1: 單階, 2: 雙階)
                 - sub_domains   : [list] 產出之次領域識別代碼清單
                 - totl_crops    : [int] 累計產出之主物件裁切影像張數
                 - totl_sub_lbls : [int] 累計投影校正之次物件標籤數量
        """
        domain = str(domain).strip().upper()
        stage_name = f"COORD-{domain}"
        log.BANNER(acnt="COORD-TASK", msin=f"階層級聯協調任務，載入 [{domain}] 資料集")

        roboflow_dir = self.path_cfg.train.roboflow
        domain_dir   = Path(roboflow_dir) / domain
        sub_out_dir  = domain_dir / "sub"

        try:
            """ [STAGE-1] 階層策略驗證與環境檢驗階段 """
            stage_rule = self.stage_cfg.get(domain)

            # 階層策略配置驗證
            if not stage_rule:
                log.CONTENT(
                    type = "STAGE",
                    targ = stage_name,
                    idnt = "VALIDATE_STAGE_RULE",
                    stat = "WARN",
                    msge = f"[領域: {domain}] 驗證警示 >>> 未定義階層策略規則，無法進行階層級聯協調"
                )
                log.FOOTER(acnt="COORD-HALT", rslt="階層級聯協調暫停流程")

                coord_mngr_info = {
                    "stat_code" : 400,
                    "stat_msge" : "MISSING_STAGE_RULE",
                    "rslt_msge" : f"[{domain}] 未定義階層策略規則，請檢查 stage_maps",
                    "details"   : {
                        "domain"        : domain,
                        "stage_qty"     : 0,
                        "sub_domains"   : [],
                        "totl_crops"    : 0,
                        "totl_sub_lbls" : 0
                    }
                }
                return coord_mngr_info

            # 單階層架構判定
            stage_qty = stage_rule.get("stage_qty", 1)
            if stage_qty < 2:
                log.CONTENT(
                    type = "STAGE",
                    targ = stage_name,
                    idnt = "BYPASS_SINGLE_STAGE_PIPELINE",
                    stat = "INFO",
                    msge = f"[領域: {domain}] 架構提示 >>> 此領域為單階層全圖訓練架構 (stage_qty=1)，自動略過階層級聯協調"
                )
                log.FOOTER(acnt="COORD-SKIP", rslt="階層級聯協調略過流程")

                coord_mngr_info = {
                    "stat_code" : 304,
                    "stat_msge" : "BYPASS_SINGLE_STAGE",
                    "rslt_msge" : f"[{domain}] 為單階層架構，自動略過階層級聯協調",
                    "details"   : {
                        "domain"        : domain,
                        "stage_qty"     : 1,
                        "sub_domains"   : [],
                        "totl_crops"    : 0,
                        "totl_sub_lbls" : 0
                    }
                }
                return coord_mngr_info

            # 標註資料集驗證
            if not domain_dir.exists():
                rel_path = input_resolver.to_root_relative(domain_dir)
                log.CONTENT(
                    type = "STAGE",
                    targ = stage_name,
                    idnt = "VALIDATE_ROBOFLOW_DATASET",
                    stat = "WARN",
                    msge = f"[目錄: {domain_dir.name}] 驗證警示 >>> 找不到指定的標註資料集目錄，請檢查路徑({rel_path})"
                )
                log.FOOTER(acnt="COORD-HALT", rslt="階層級聯協調暫停流程")

                coord_mngr_info = {
                    "stat_code" : 404,
                    "stat_msge" : "MISSING_ROBOFLOW_DATASET",
                    "rslt_msge" : f"[{domain}] 標註資料集目錄不存在，請檢查檔案路徑",
                    "details"   : {
                        "domain"        : domain,
                        "stage_qty"     : stage_qty,
                        "sub_domains"   : [],
                        "totl_crops"    : 0,
                        "totl_sub_lbls" : 0
                    }
                }
                return coord_mngr_info

            """ [STAGE-2] 級聯輸出環境清理與配置 """
            if sub_out_dir.exists():
                shutil.rmtree(sub_out_dir)
            sub_out_dir.mkdir(parents=True, exist_ok=True)
            
            # 次階層策略配置
            stage_2_cfg = stage_rule.get("stage_2", {})
            sub_tasks   = stage_2_cfg.get("sub_tasks", {})
            sub_domains = list(sub_tasks.keys())

            # 初始化級聯轉換統計指標與全域血統溯源累積清單
            totl_crops      = 0
            totl_sub_lbls   = 0
            lineage_records = []

            """ [STAGE-3] 主物件邊界裁切與次物件坐標投影校正 """
            log.CONTENT(
                type = "STAGE",
                targ = stage_name,
                idnt = "EXECUTE_CROP_AND_PROJECTION",
                stat = "INFO",
                msge = f"[任務: 主物件裁切與次物件投影] 啟動成功 >>> 涵蓋次領域: [{', '.join(sub_domains)}]"
            )
            
            for split in ["train", "valid", "test"]:
                stats = self._process_dataset_split(
                    domain_dir      = domain_dir,
                    sub_out_dir     = sub_out_dir,
                    split           = split,
                    stage_rule      = stage_rule,
                    lineage_records = lineage_records
                )
                totl_crops    += stats["crops_qty"]
                totl_sub_lbls += stats["sub_labels_qty"]

            """ [STAGE-4] 次階專屬訓練組態檔案生成 """
            for sub_domain, sub_info in sub_tasks.items():
                sub_domain_dir = sub_out_dir / sub_domain
                if sub_domain_dir.exists():
                    self._generate_sub_yaml(sub_domain_dir, sub_domain, sub_info)

            log.CONTENT(
                type = "STAGE",
                targ = stage_name,
                idnt = "GENERATE_SUB_DATA_YAML",
                stat = "SUCC",
                msge = f"[組態: data.yaml] 生成成功 >>> 已完成 {len(sub_domains)} 組次領域專屬訓練組態"
            )

            """ [STAGE-5] 資料血統溯源檔案寫入 """
            lineage_path = sub_out_dir / "sub_obj_lineage.json"
            with open(lineage_path, "w", encoding="utf-8") as jf:
                json.dump(lineage_records, jf, indent=2, ensure_ascii=False)
            
            log.CONTENT(
                type = "STAGE",
                targ = stage_name,
                idnt = "WRITE_SUB_OBJECT_LINEAGE",
                stat = "SUCC",
                msge = f"[記錄: {lineage_path.name}] 寫入成功 >>> 累計建立 {len(lineage_records)} 筆跨階層級聯血統紀錄"
            )
            
            summary_msge = (
                f"\n >>> 目標主領域: {domain} (雙階層架構)"
                f"\n >>> 涵蓋次領域: {', '.join(sub_domains)}"
                f"\n >>> 主物件裁切: {totl_crops} 張局部裁切影像"
                f"\n >>> 次物件校正: {totl_sub_lbls} 組投影平移標籤"
                f"\n >>> 血統溯源檔: {lineage_path.name}"
            )
            log.CONTENT(
                type = "STAGE",
                targ = stage_name,
                idnt = "COORDINATION_SUMMARY",
                stat = "SUCC",
                msge = f"[領域: {domain}] 協調成功 >>> 級聯統計如下: {summary_msge}"
            )
            log.FOOTER(acnt="COORD-DONE", rslt=f"[{domain}] 階層級聯協調全數完成，次階訓練資料集已就緒")

            coord_mngr_info = {
                "stat_code" : 200,
                "stat_msge" : "COMPLETED",
                "rslt_msge" : f"[{domain}] 階層級聯協調作業圓滿完成",
                "details"   : {
                    "domain"        : domain,
                    "stage_qty"     : stage_qty,
                    "sub_domains"   : sub_domains,
                    "totl_crops"    : totl_crops,
                    "totl_sub_lbls" : totl_sub_lbls
                }
            }
            return coord_mngr_info

        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "STAGE",
                targ = stage_name,
                idnt = "COORDINATION_EXCEPTION",
                stat = "FAIL",
                msge = f"[任務: 階層協調任務] 運行失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            log.FOOTER(acnt="COORD-FAIL", rslt="階層級聯協調發生異常，請調閱堆疊日誌排查問題")

            coord_mngr_info = {
                "stat_code" : 500,
                "stat_msge" : "SYSTEM_ERROR",
                "rslt_msge" : f"[{stage_name}] 協調過程遭遇非預期系統異常",
                "details"   : {
                    "domain"        : domain,
                    "stage_qty"     : 0,
                    "sub_domains"   : [],
                    "totl_crops"    : 0,
                    "totl_sub_lbls" : 0
                }
            }
            return coord_mngr_info