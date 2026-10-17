# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 階層級聯協調工具 (Hierarchical Cascade Coordination Tool)
# 維護日期: 2026-10-15
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
           1. 針對主物件實施「類別編號平移歸零」，並自動建立 Stage-1 專屬之 main/ 資料集
           2. 依據主物件全景邊界實施「安全外擴裁切」，產出次階層專屬訓練特徵切片
           3. 針對次物件實施「坐標動態投影與限幅」，將全景坐標換算為切片相對坐標
           4. 針對次物件實施「類別編號平移歸零」，並自動建立 Stage-2 專屬之 sub/ 資料集
           5. 自動建立「全景血統溯源檔」，記錄切片與母圖之映射關係以利後續回溯
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
    def _process_dataset_split(self, domain_dir, main_out_dir, sub_out_dir, split, stage_rule, lineage_records):
        """
        [名稱] Func.B 單一劃分集級聯處理函式
        [功能] 依據各 train/valid/test 劃分集，統籌主物件編號平移、安全邊界裁切、次目標坐標投影與編號平移。
        [參數] 共計 6 組參數，以下說明:
               - domain_dir      : [str/Path] 原始領域物件標註資料集根目錄 (e.g., roboflow/VEH)
               - main_out_dir    : [str/Path] Stage-1 主階層純主物件輸出目錄 (e.g., roboflow/VEH/main)
               - sub_out_dir     : [str/Path] Stage-2 次階層切片標註輸出目錄 (e.g., roboflow/VEH/sub)
               - split           : [str] 全景資料劃分集名稱 ('train', 'valid', 'test')
               - stage_rule      : [dict] 該領域階層策略字典
               - lineage_records : [list] 資料血統溯源紀錄累加清單
        [輸出] dict: 內含單一劃分集轉換統計之結構化字典，共計 3 組鍵值，以下說明:
               - crops_qty       : [int] 主物件裁切圖片總張數
               - main_labels_qty : [int] 主物件標註框總數 (main 標籤總行數)
               - sub_labels_qty  : [int] 次物件標註框總數 (sub 標籤總行數)
        """
        # 全景資料劃分集路徑
        split_img_dir = Path(domain_dir) / split / "images"
        split_lbl_dir = Path(domain_dir) / split / "labels"

        split_stats = {"crops_qty": 0, "main_labels_qty":0 ,"sub_labels_qty": 0}
        if not split_img_dir.exists():
            return split_stats
        
        # 類別名稱 (cls_name) -> 全域編號 (global_cls_id)
        cls_name_to_id = {}

        """ [STAGE-1] 建立 Stage-1 主階層類別編號對照表 """
        stage_1_cfg  = stage_rule.get("stage_1", {})
        stage_1_bbox = stage_1_cfg.get("bbox_type").upper()
        stage_1_cls_names = stage_1_cfg.get("cls_names", [])
        stage_1_cls_ids   = stage_1_cfg.get("gt_cls_ids", [])

        main_cls_id_remap = {}   # 全域編號 (global_cls_id) -> 本地編號 (local_cls_id)
        for local_cls_id, (cls_name, global_cls_id) in enumerate(zip(stage_1_cls_names, stage_1_cls_ids)):
            cls_name_to_id[cls_name.lower().strip()] = global_cls_id
            main_cls_id_remap[global_cls_id] = local_cls_id

        # 階層級聯銜接參數
        transition  = stage_rule.get("transition", {})
        crop_margin = transition.get("crop_margin", 0.05)

        """ [STAGE-2] 建立 Stage-2 次階層類別編號對照表 """
        stage_2_cfg = stage_rule.get("stage_2", {})
        sub_tasks   = stage_2_cfg.get("sub_tasks", {})

        sub_cls_id_remap = {}   # 全域編號 (global_cls_id) -> 次領域名稱 (sub_domain), 本地編號 (local_cls_id)
        for sub_domain, sub_info in sub_tasks.items():
            stage_2_cls_names = sub_info.get("cls_names", [])
            stage_2_cls_ids   = sub_info.get("gt_cls_ids", [])
            for local_cls_id, (cls_name, global_cls_id) in enumerate(zip(stage_2_cls_names, stage_2_cls_ids)):
                cls_name_to_id[cls_name.lower().strip()] = global_cls_id
                sub_cls_id_remap[global_cls_id] = (sub_domain, local_cls_id)

        """ [STAGE-3] 讀取 data.yaml 取得該領域下所有類別名稱 (names) """
        data_yaml_path = Path(domain_dir) / "data.yaml"
        yaml_cls_names = []
        if data_yaml_path.exists():
            with open(data_yaml_path, "r", encoding="utf-8") as yaml_file:
                yaml_cfg = yaml.safe_load(yaml_file)
                yaml_cls_names = [str(cls_name).lower().strip() for cls_name in yaml_cfg.get("names", [])]

        """ [STAGE-4] 建立 Stage-1 主領域輸出目錄 (main/<split>/images & labels) """
        main_split_img_dir = Path(main_out_dir) / split / "images"
        main_split_lbl_dir = Path(main_out_dir) / split / "labels"
        main_split_img_dir.mkdir(parents=True, exist_ok=True)
        main_split_lbl_dir.mkdir(parents=True, exist_ok=True)

        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        for img_path in sorted(split_img_dir.iterdir()):
            """ [STAGE-5] 檢索所有「全景資料劃分集」影像與標註檔案 (Roboflow 下 train, valid, test) """
            if not (img_path.is_file() and img_path.suffix.lower() in valid_extensions):
                continue
            lbl_path = split_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    sub_obj_line = [line.strip() for line in f if line.strip()]
            except Exception:
                continue

            """ [STAGE-6] 主物件座標資訊分流篩選 """
            lbl_items = [self._parse_object_label(label, yaml_cls_names, cls_name_to_id) for label in sub_obj_line]
            lbl_items = [item for item in lbl_items if item is not None]
            main_objs = [item for item in lbl_items if item["cls_id"] in stage_1_cls_ids]
            sub_objs  = [item for item in lbl_items if item["cls_id"] in sub_cls_id_remap]

            """ [STAGE-7] 主物件標註資料集建立 """
            if main_objs:
                # 建立主物件原始圖片 (images)
                main_obj_img = main_split_img_dir / img_path.name
                if not main_obj_img.exists():
                    shutil.copy2(img_path, main_obj_img)

                main_obj_lines = []
                for m_obj in main_objs:
                    # 驗證主物件邊框類型是否一致 (階層配置, 標註資料, 座標數量)
                    is_valid_hbb = (stage_1_bbox == "HBB" and m_obj["bbox_type"] == "HBB" and len(m_obj["coords"]) == 4)
                    is_valid_obb = (stage_1_bbox == "OBB" and m_obj["bbox_type"] == "OBB" and len(m_obj["coords"]) == 8)
                    if is_valid_hbb or is_valid_obb:
                        local_cls_id = main_cls_id_remap[m_obj["cls_id"]]
                        coord_str    = " ".join(f"{coord:.6f}" for coord in m_obj["coords"])
                        main_obj_lines.append(f"{local_cls_id} {coord_str}")

                # 建立主物件標註資料 (labels)
                if main_obj_lines:
                    with open(main_split_lbl_dir / f"{img_path.stem}.txt", "w", encoding="utf-8") as main_f:
                        main_f.write("\n".join(main_obj_lines) + "\n")
                    split_stats["main_labels_qty"] += len(main_obj_lines)

            if not main_objs:
                continue

            # 載入全景影像像素矩陣
            img_background = cv2.imread(str(img_path))
            if img_background is None:
                continue
            orig_h, orig_w = img_background.shape[:2]

            """ [STAGE-8] 主物件全景邊界定位與安全外擴裁切 """
            for crop_idx, m_obj in enumerate(main_objs):
                main_coords  = m_obj["coords"]
                total_coords = len(main_coords)
                # [STEP-1] 主物件座標資訊定位
                if m_obj["bbox_type"] == "HBB":
                    # 水平框類別(HBB) [xc, yc, w, h]
                    main_x_center = main_coords[0] * orig_w
                    main_y_center = main_coords[1] * orig_h
                    main_bbox_w   = main_coords[2] * orig_w
                    main_bbox_h   = main_coords[3] * orig_h
                    main_x_min, main_x_max = main_x_center - (main_bbox_w / 2.0), main_x_center + (main_bbox_w / 2.0)
                    main_y_min, main_y_max = main_y_center - (main_bbox_h / 2.0), main_y_center + (main_bbox_h / 2.0)

                elif m_obj["bbox_type"] == "OBB":
                    # 旋轉框類別(OBB) [x1, y1, ..., x4, y4]
                    main_pts_x = [main_coords[i] * orig_w for i in range(0, total_coords, 2)]
                    main_pts_y = [main_coords[i+1] * orig_h for i in range(0, total_coords, 2)]
                    main_x_min, main_x_max = min(main_pts_x), max(main_pts_x)
                    main_y_min, main_y_max = min(main_pts_y), max(main_pts_y)
                    main_bbox_w = main_x_max - main_x_min
                    main_bbox_h = main_y_max - main_y_min

                # [STEP-2] 安全外擴邊界計算
                expand_w = main_bbox_w * crop_margin
                expand_h = main_bbox_h * crop_margin
                crop_x_min, crop_x_max = max(0, int(np.floor(main_x_min - expand_w))), min(orig_w, int(np.ceil(main_x_max + expand_w)))
                crop_y_min, crop_y_max = max(0, int(np.floor(main_y_min - expand_h))), min(orig_h, int(np.ceil(main_y_max + expand_h)))
                
                # [STEP-3] 主物件影像裁切區塊檢驗
                if (crop_x_max - crop_x_min) < 8 or (crop_y_max - crop_y_min) < 8:
                    # [NOTE] 裁切寬高若小於 8 像素屬於無效退化區域或嚴重變形，不具備特徵辨識價值予以捨棄
                    continue
                crop_patch = img_background[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                crop_stem  = f"{img_path.stem}_crop{crop_idx:03d}"
                crop_frame = f"{crop_stem}.jpg"

                """ [STAGE-9] 次物件坐標投影校正與類別編號平移 """
                sub_obj_lines = {sub_domain: [] for sub_domain in sub_tasks.keys()}
                for s_obj in sub_objs:
                    # [STEP-1] 次物件類別編號平移 (Global ID -> Local ID)
                    sub_domain, local_cls_id = sub_cls_id_remap[s_obj["cls_id"]]

                    # [STEP-2] 次物件座標投影校正 (全景坐標 -> 裁切切片相對坐標)
                    proj_coords = self._project_coords(
                        coords    = s_obj["coords"],
                        bbox_type = s_obj["bbox_type"],
                        orig_wh   = (orig_w, orig_h),
                        crop_rect = (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
                    )
                    if proj_coords is not None:
                        coords_str = " ".join(f"{coord:.6f}" for coord in proj_coords)
                        sub_obj_lines[sub_domain].append(f"{local_cls_id} {coords_str}")
                
                """ [STAGE-10] 次物件存在性檢驗與級聯資料集儲存 """
                crop_has_sub = any(len(lines) > 0 for lines in sub_obj_lines.values())
                if crop_has_sub:
                    split_stats["crops_qty"] += 1
                    # 各次領域各自儲存對應之影像與標籤 (sub/<SUB_DOMAIN>/images/<split> 與 labels/<split>)
                    # 確保只有真正含有該次領域目標的切片，才會納入該次領域之獨立訓練集
                    save_sub_domains = []
                    for sub_domain, sub_obj_line in sub_obj_lines.items():
                        if not sub_obj_line:
                            continue
                        
                        # [STEP-1] 建立 Stage-2 次領域輸出目錄 (sub/<sub_domain>/<split>/images & labels)
                        sub_split_img_dir = Path(sub_out_dir) / sub_domain / split / "images"
                        sub_split_lbl_dir = Path(sub_out_dir) / sub_domain / split / "labels"
                        sub_split_img_dir.mkdir(parents=True, exist_ok=True)
                        sub_split_lbl_dir.mkdir(parents=True, exist_ok=True)

                        # [STEP-2] 主物件局部裁切影像輸出
                        # 輸出局部特徵切片作為次階專屬訓練底圖，排除全景影像之無關背景雜訊
                        cv2.imwrite(str(sub_split_img_dir / crop_frame), crop_patch)

                        # [STEP-3] 次物件投影坐標與標籤文字檔寫入
                        # 寫入平移歸零後的次領域局部標註，完全符合 YOLO 獨立資料集訓練規範
                        with open(sub_split_lbl_dir / f"{crop_stem}.txt", "w", encoding="utf-8") as f:
                            f.write("\n".join(sub_obj_line) + "\n")

                        split_stats["sub_labels_qty"] += len(sub_obj_line)
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
    def _parse_object_label(self, label, yaml_cls_names=None, cls_name_to_id=None):
        """
        [名稱] Func.B-1 單筆物件標註解析函式
        [功能] 拆解單行 YOLO 標註文字為結構化物件資訊 (cls_id, bbox_type, coords)，並自動轉譯類別編號及判定幾何形態：
               1. 藉由 data.yaml 索引值對應所屬類別名稱，自動對齊階層策略全域ID (Global ID)。
               2. 嚴格過濾未定義於階層策略之無效或異常標籤。
               3. 自動辨別 水平框 (HBB-4項) 與 旋轉框 (OBB-8項)，並相容 Roboflow 首尾閉合之 10 項多邊形坐標。
        [參數] 共計 3 組參數，以下說明:
               - label          : [str] YOLO 格式單筆標註文字字串
               - yaml_cls_names : [list/None] Roboflow 標註中介資料 (data.yaml) 定義之類別名稱清單
               - cls_name_to_id : [dict/None] 階層策略規範之「類別名稱」與「全域ID」對照表
        [輸出] dict/None: 內含幾何解析結果之結構化字典，若傳入為空行或純空白字串則回傳 None，共計 3 組鍵值，以下說明:
               - cls_id    : [int] 物件全域類別編號 (global_cls_id)
               - bbox_type : [str/None] 邊界框類型 ("HBB": 水平框, "OBB": 旋轉框, None: 非標準或未定義格式)
               - coords    : [list] 正規化幾何點位清單 (HBB 為 4 項 [xc, yc, w, h]，OBB 為 8 項 [x1, y1, ..., x4, y4])
        """
        parts = label.split()
        if not parts:
            return None
        lbl_cls_id = int(parts[0])

        if yaml_cls_names and cls_name_to_id and lbl_cls_id < len(yaml_cls_names):
            # 位於 data.yaml 的 names 必須被定義於 stage_rule 對照表中
            cls_name = yaml_cls_names[lbl_cls_id]
            # 標註標籤未定義在 stage_rule 階層策略中，則無效略過
            if cls_name not in cls_name_to_id:
                return None
            global_cls_id = cls_name_to_id[cls_name]
        else:
            # 無法對照之異常標籤，直接剔除
            return None

        coords = [float(coord) for coord in parts[1:]]
        
        if len(coords) == 4:
            bbox_type = "HBB"
        elif len(coords) >= 8:
            bbox_type = "OBB"
            # 若為 10 個座標，屬於 Roboflow 多邊形首尾閉合，保留前 8 個數值作為 4 頂點旋轉框頂點
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
    def _generate_yaml_config(self, target_dir, task_cfg):
        """
        [名稱] Func.C 階層訓練組態生成函式
        [功能] 依據階層任務規格自動產出合規之主階或次階專屬訓練設定檔 (data.yaml)。
        [參數] 共計 2 組參數，以下說明:
               - target_dir : [Path] 目標階層專屬根目錄 (e.g., roboflow/VEH/ main 或 sub/LIC)
               - task_cfg   : [dict] 該階層專屬任務規格配置 (包含 cls_names, gt_cls_ids 等規格)
        [輸出] Path: 產出之 data.yaml 實體路徑
        """
        data_yaml_path = target_dir / "data.yaml"
        cls_names = task_cfg.get("cls_names", [])

        # 遵循 YOLO 鏡像尋找機制 (<split>/images 與 <split>/labels)
        target_root_path = target_dir.resolve()
        yaml_content = {
            "path"  : str(target_root_path),
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
        [功能] 驅動階層級聯協調管線生命週期，統籌策略驗證、主次領域資料集拆分、主物件安全裁切、次目標投影校正與血統溯源檔案建置。
        [參數] domain: [str] 目標領域識別代碼 (e.g., "VEH", "MARA")
        [輸出] dict: 內含執行結果之結構化字典，共計 4 組鍵值，以下說明:
               - stat_code : [int] 執行狀態代碼 (200-成功, 304-單階略過, 400-策略缺失, 404-目錄缺失, 500-系統異常)
               - stat_msge : [str] 狀態識別標籤 (COMPLETED, BYPASS_SINGLE_STAGE, MISSING_STAGE_RULE, MISSING_ROBOFLOW_DATASET, SYSTEM_ERROR)
               - rslt_msge : [str] 執行結果之詳細文字說明
               - details   : [dict] 共計 5 組任務指標明細，以下說明:
                 - domain         : [str] 目標領域識別代碼 (e.g., "VEH")
                 - stage_qty      : [int] 模型架構階層數 (1: 單階全圖訓練, 2: 雙階級聯訓練)
                 - sub_domains    : [list] 產出之次領域識別代碼清單，若為單階層則為 []
                 - total_crops    : [int] 累計產出之主物件裁切影像總張數
                 - total_sub_lbls : [int] 累計投影校正之次物件標註框總數
        """
        domain = str(domain).strip().upper()
        stage_name = f"COORD-{domain}"
        log.BANNER(acnt="COORD-TASK", msin=f"階層級聯協調任務 (載入 {domain} 資料集)")

        roboflow_dir = self.path_cfg.train.roboflow
        domain_dir   = Path(roboflow_dir) / domain
        main_out_dir = domain_dir / "main"
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
                    msge = f"[領域: {domain}] 未定義階層策略規則，無法進行階層級聯協調"
                )
                log.FOOTER(acnt="COORD-HALT", rslt=f"[{domain}] 未定義階層策略規則，任務安全中止")

                coord_mngr_info = {
                    "stat_code" : 400,
                    "stat_msge" : "MISSING_STAGE_RULE",
                    "rslt_msge" : f"[{domain}] 未定義階層策略規則，請檢查 stage_maps",
                    "details"   : {
                        "domain"         : domain,
                        "stage_qty"      : 0,
                        "sub_domains"    : [],
                        "total_crops"    : 0,
                        "total_sub_lbls" : 0
                    }
                }
                return coord_mngr_info

            # 單階層架構判定
            stage_qty = stage_rule.get("stage_qty", 1)
            if stage_qty < 2:
                log.CONTENT(
                    type = "STAGE",
                    targ = stage_name,
                    idnt = "BYPASS_SINGLE_STAGE",
                    stat = "INFO",
                    msge = f"[領域: {domain}] 此領域為單階層全圖訓練架構 (stage_qty=1)，自動略過階層級聯協調"
                )
                log.FOOTER(acnt="COORD-SKIP", rslt=f"[{domain}] 單階層全圖訓練架構，階層協調安全略過")

                coord_mngr_info = {
                    "stat_code" : 304,
                    "stat_msge" : "BYPASS_SINGLE_STAGE",
                    "rslt_msge" : f"[{domain}] 為單階層架構，自動略過階層級聯協調",
                    "details"   : {
                        "domain"         : domain,
                        "stage_qty"      : 1,
                        "sub_domains"    : [],
                        "total_crops"    : 0,
                        "total_sub_lbls" : 0
                    }
                }
                return coord_mngr_info

            # 標註資料集驗證
            if not domain_dir.exists():
                rel_path = input_resolver.to_root_relative(domain_dir)
                log.CONTENT(
                    type = "STAGE",
                    targ = stage_name,
                    idnt = "VALIDATE_DATASET_DIR",
                    stat = "WARN",
                    msge = f"[目錄: {domain_dir.name}] 找不到指定的標註資料集目錄，請檢查路徑 ({rel_path})"
                )
                log.FOOTER(acnt="COORD-HALT", rslt=f"[{domain}] 標註資料集目錄不存在，任務安全中止")

                coord_mngr_info = {
                    "stat_code" : 404,
                    "stat_msge" : "MISSING_ROBOFLOW_DATASET",
                    "rslt_msge" : f"[{domain}] 標註資料集目錄不存在，請檢查檔案路徑",
                    "details"   : {
                        "domain"         : domain,
                        "stage_qty"      : stage_qty,
                        "sub_domains"    : [],
                        "total_crops"    : 0,
                        "total_sub_lbls" : 0
                    }
                }
                return coord_mngr_info

            """ [STAGE-2] 級聯輸出環境清理與配置 """
            if main_out_dir.exists():
                shutil.rmtree(main_out_dir)
            main_out_dir.mkdir(parents=True, exist_ok=True)

            if sub_out_dir.exists():
                shutil.rmtree(sub_out_dir)
            sub_out_dir.mkdir(parents=True, exist_ok=True)
            
            # 主階層與次階層策略組態
            stage_1_cfg = stage_rule.get("stage_1", {})
            stage_2_cfg = stage_rule.get("stage_2", {})
            sub_tasks   = stage_2_cfg["sub_tasks"]
            sub_domains = list(sub_tasks.keys())

            # 初始化級聯轉換統計指標與全域血統溯源累積清單
            total_crops     = 0
            total_main_lbls = 0
            total_sub_lbls  = 0
            lineage_records = []

            """ [STAGE-3] 主物件邊界裁切與次物件坐標投影校正 """
            log.CONTENT(
                type = "STAGE",
                targ = stage_name,
                idnt = "EXECUTE_CASCADE_CROP",
                stat = "INFO",
                msge = f"[任務: 主物件裁切與次物件投影] 啟動協調作業，涵蓋次領域: [{', '.join(sub_domains)}]"
            )
            
            for split in ["train", "valid", "test"]:
                stats = self._process_dataset_split(
                    domain_dir      = domain_dir,
                    main_out_dir    = main_out_dir,
                    sub_out_dir     = sub_out_dir,
                    split           = split,
                    stage_rule      = stage_rule,
                    lineage_records = lineage_records
                )
                total_crops     += stats["crops_qty"]
                total_main_lbls += stats["main_labels_qty"]
                total_sub_lbls  += stats["sub_labels_qty"]

            """ [STAGE-4] 主次階層專屬訓練組態檔案生成 """
            # [STEP-1] 主領域專屬訓練組態生成 (main/data.yaml)
            target_dir = main_out_dir
            self._generate_yaml_config(target_dir, stage_1_cfg)

            # [STEP-2] 次領域專屬訓練組態生成 (sub/<SUB_DOMAIN>/data.yaml)
            for sub_domain, sub_info in sub_tasks.items():
                target_dir = sub_out_dir / sub_domain
                if target_dir.exists():
                    self._generate_yaml_config(target_dir, sub_info)

            log.CONTENT(
                type = "STAGE",
                targ = stage_name,
                idnt = "GENERATE_STAGE_YAML",
                stat = "SUCC",
                msge = f"[組態: data.yaml] 已完成主領域及 {len(sub_domains)} 組次領域專屬訓練組態生成"
            )

            """ [STAGE-5] 資料血統溯源檔案寫入 """
            lineage_path = sub_out_dir / "sub_obj_lineage.json"
            with open(lineage_path, "w", encoding="utf-8") as jf:
                json.dump(lineage_records, jf, indent=2, ensure_ascii=False)
            
            log.CONTENT(
                type = "STAGE",
                targ = stage_name,
                idnt = "WRITE_SUB_LINEAGE",
                stat = "SUCC",
                msge = f"[記錄: {lineage_path.name}] 累計建立 {len(lineage_records)} 筆跨階層級聯血統紀錄"
            )
            
            summary_msge = (
                f"\n目標領域: {domain} (雙階層架構)"
                f"\n涵蓋次領域: {', '.join(sub_domains)}"
                f"\n主物件數量: {total_main_lbls} 個標註框"
                f"\n裁切圖片數: {total_crops} 張"
                f"\n次物件數量: {total_sub_lbls} 個標註框"
                f"\n血統溯源檔: {lineage_path.name}"
            )
            log.CONTENT(
                type = "STAGE",
                targ = stage_name,
                idnt = "COORDINATION_SUMMARY",
                stat = "SUCC",
                msge = f"[領域: {domain}] 階層協調完成，級聯統計如下:{summary_msge}"
            )
            log.FOOTER(acnt="COORD-DONE", rslt=f"[{domain}] 階層級聯協調全數完成，次階訓練資料集已就緒")

            coord_mngr_info = {
                "stat_code" : 200,
                "stat_msge" : "COMPLETED",
                "rslt_msge" : f"[{domain}] 階層級聯協調作業圓滿完成",
                "details"   : {
                    "domain"         : domain,
                    "stage_qty"      : stage_qty,
                    "sub_domains"    : sub_domains,
                    "total_crops"    : total_crops,
                    "total_sub_lbls" : total_sub_lbls
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
                msge = f"[任務: 階層協調任務] 遭遇非預期例外中斷，堆疊資訊如下: \n{sys_err}"
            )
            log.FOOTER(acnt="COORD-FAIL", rslt=f"[{domain}] 階層級聯協調非預期中斷，請調閱堆疊排查")

            coord_mngr_info = {
                "stat_code" : 500,
                "stat_msge" : "SYSTEM_ERROR",
                "rslt_msge" : f"[{domain}] 協調過程遭遇非預期系統異常",
                "details"   : {
                    "domain"         : domain,
                    "stage_qty"      : 0,
                    "sub_domains"    : [],
                    "total_crops"    : 0,
                    "total_sub_lbls" : 0
                }
            }
            return coord_mngr_info