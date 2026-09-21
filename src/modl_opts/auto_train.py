# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 自動化模型訓練工具 (Automated Model Training Tool)
# 維護日期: 2026-09-22
# 檔案路徑: MARS_Project/src/modl_opts/auto_train.py
# ##########################################################################################

# 掛載外部依賴
import re
import yaml
import shutil
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# 掛載內部依賴
from utils import log
from utils.env import input_resolver
from src.database import write_repo

# ==========================================================================================
class YoloModelTrainer:
    """
    [名稱] 自動化模型訓練引擎 (Automated Model Training Engine)
    [作用] 對接「全域階層策略-TRAIN」與「演算法路徑組態」，專職統籌單/雙階層架構之 YOLO 模型訓練管線:
           1. 自動識別「單階層」或「雙階層」訓練架構
           2. 依階層策略動態導引「全景資料集」或「主物件裁切資料集」 (data.yaml)
           3. 動態掛載對應之「預訓練骨幹權重」 (HBB / OBB Backbone)
           4. 驅動 YOLO 模型擬合、最佳權重保存與收斂指標批次寫入資料庫雙表 (modl_ver_info, modl_fit_eval)
    """
    # =============================================
    def __init__(self, sys_cfg):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「自動化模型訓練引擎」，並掛載演算法組態 (algo)、路徑組態 (path) 及階層策略字典 (stage)。
        [參數] sys_cfg: [dict] 全域組態檢索，內含 algo_inst, path_inst 與 stage_inst 組態
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg   = sys_cfg
        self.algo_cfg  = self.sys_cfg["algo_inst"]
        self.path_cfg  = self.sys_cfg["path_inst"]
        self.stage_cfg = self.sys_cfg["stage_inst"]

    # =============================================
    def _execute_stage_pipeline(self, domain, ver_id, train_date, stage_level, sub_domain, task_spec, train_params):
        """
        [名稱] Func.B 階層訓練管線工法函式
        [功能] 統籌單一階層之完整模型訓練流程 (中介驗證 -> 模型擬合 -> 指標解析 -> 資料庫同步)
        [參數] 共計 7 組參數，以下說明:
               - domain       : [str] 主領域識別代碼 (e.g., "VEH")
               - ver_id       : [str] 主領域版本代碼 (e.g., "v1.0.0")
               - train_date   : [str] 模型訓練日期 (YYYY-MM-DD)
               - stage_level  : [int] 階層識別代碼 (1: 主階層, 2: 次階層)
               - sub_domain   : [str] 次領域識別代碼 ("LIC", "BRAND")
               - task_spec    : [dict] 該階層之專屬策略配置 (內含 task_name, domain, bbox_type, gt_cls_ids 等 7 項配置)
               - train_params : [dict] 訓練參數字典 (batch_size, nbs, epoch_qty, learn_rate)
        [輸出] dict: 內含執行結果之結構化字典，共計 4 組鍵值，以下說明:
               - stat_code : [int] 執行狀態代碼 (200-成功, 206-局部落地, 400-格式錯誤, 404-缺檔, 500-系統異常)
               - stat_msge : [str] 狀態識別標籤 (e.g., COMPLETED, PARTIAL_SUCCESS, MISSING_DATA_YAML, SYSTEM_ERROR)
               - rslt_msge : [str] 執行結果之詳細文字說明
               - details   : [dict] 共計 6 組任務指標明細，以下說明:
                 - domain       : [str] 目標領域識別代碼 (e.g., "VEH")
                 - ver_id       : [str] 目標領域版本代碼 (e.g., v1.0.0)
                 - stage_level  : [int] 階層識別代碼 (1: 主階層, 2: 次階層)
                 - sub_domains  : [list] 次領域識別代碼清單，若為單階層則為 []
                 - final_map50s : [list] 模型收斂後之最佳 mAP@0.5 精度表現清單
                 - wght_paths   : [list] 產出之最佳權重絕對路徑清單
        """
        stage_name = f"YOLO-T-{domain}" if stage_level == 1 else f"YOLO-T-{domain}-{sub_domain}"

        """ [STAGE-1] 模型組態配置與路徑對應階段 """
        roboflow_dir = Path(self.path_cfg.train.roboflow)
        weights_dir  = Path(self.path_cfg.weights)

        yolo_t_cfg = self.algo_cfg.yolo_t
        default_hbb_bone = yolo_t_cfg.default_hbb_bone
        default_obb_bone = yolo_t_cfg.default_obb_bone
        proj_name    = yolo_t_cfg.proj_name
        bbox_type    = task_spec.get("bbox_type", "HBB" if stage_level == 1 else "OBB").upper()
        default_bone = default_hbb_bone if bbox_type == "HBB" else default_obb_bone
        bone_name    = task_spec.get("bone_name", default_bone)

        # 從「預訓練骨幹權重」檔名中動態提取架構名稱 (yolo11n.pt -> YOLO11, yolo11n-obb.pt -> YOLO11)
        # 組裝演算法架構名稱 (e.g., MARS-YOLO11)
        match = re.match(r"(yolov?\d+)", bone_name, re.IGNORECASE)
        arch_name = match.group(1).upper() if match else "YOLO"
        algo_name = f"{proj_name}-{arch_name}"

        if stage_level == 1:
            # 主階層路徑配置:
            # - 標註中介路徑: data/01_train/roboflow/<DOMAIN>/data.yaml
            data_yaml_path = roboflow_dir / domain / "data.yaml"
            # - 權重版本目錄: weights/<DOMAIN>/main/<VER_ID>/
            wght_ver_dir   = weights_dir / domain / "main" / ver_id 

        elif stage_level == 2:
            # 次階層路徑配置:
            # - 標註中介路徑: data/01_train/roboflow/<DOMAIN>/sub/<SUB_DOMAIN>/data.yaml
            data_yaml_path = roboflow_dir / domain / "sub" / sub_domain / "data.yaml"
            # - 權重版本目錄: weights/<DOMAIN>/sub/<SUB_DOMAIN>/<VER_ID>/
            wght_ver_dir   = weights_dir / domain / "sub" / sub_domain / ver_id

        # 最佳原生權重實體路徑
        best_wght_path = wght_ver_dir / "best.pt"

        # 模型訓練成果根目錄 (對應 YOLO 之 project 參數)
        yolo_rslt_dir  = wght_ver_dir

        """ [STAGE-2] 標註資料集中介資料驗證與解析 (Func.B-1) """
        data_cfg, err_ret = self._prepare_data_yaml(
            stage_name     = stage_name,
            domain         = domain,
            data_yaml_path = data_yaml_path
        )
        if err_ret:
            return err_ret

        """ [STAGE-3] 驅動 YOLO 模型擬合訓練 (Func.B-2) """
        default_imgsz = yolo_t_cfg.default_main_imgsz if stage_level == 1 else yolo_t_cfg.default_sub_imgsz
        img_size = int(task_spec.get("imgsz") or default_imgsz)

        yolo_run_dir, err_ret = self._execute_yolo_train(
            stage_name     = stage_name,
            domain         = domain,
            bone_name      = bone_name,
            data_yaml_path = data_yaml_path,
            img_size       = img_size,
            yolo_rslt_dir  = yolo_rslt_dir,
            train_params   = train_params
        )
        if err_ret:
            return err_ret

        """ [STAGE-4] 訓練指標時序日誌解析階段 (Func.B-3) """
        metrics, _ = self._resolve_train_metrics(
            stage_name   = stage_name,
            rslt_run_dir = yolo_run_dir
        )

        """ [STAGE-5] 模型版本與成效指標資料庫同步階段 (Func.B-4) """
        sub_ver_id    = ver_id if stage_level == 2 else "NONE"
        class_qty     = data_cfg["nc"]
        wght_rel_path = input_resolver.to_root_relative(best_wght_path)

        sync_rslt, err_ret = self._sync_train_database(
            stage_level  = stage_level,
            stage_name   = stage_name,
            domain       = domain,
            ver_id       = ver_id,
            sub_domain   = sub_domain,
            sub_ver_id   = sub_ver_id,
            bbox_type    = bbox_type,
            train_date   = train_date,
            algo_name    = algo_name,
            bone_name    = bone_name,
            wght_path    = str(wght_rel_path),  
            class_qty    = class_qty,
            img_size     = img_size,
            train_params = train_params,
            metrics      = metrics
        )
        if err_ret:
            return err_ret

        best_map50 = float(metrics.get("best_map50", 0.0))
        yolo_train_rslt = {
            "stat_code" : sync_rslt["stat_code"],
            "stat_msge" : sync_rslt["stat_msge"],
            "rslt_msge" : f"[{domain}] {sync_rslt['rslt_msge']}",
            "details"   : {
                "domain"       : domain,
                "ver_id"       : ver_id,
                "stage_level"  : stage_level,
                "sub_domains"  : [sub_domain] if stage_level == 2 else [],
                "final_map50s" : [best_map50],
                "wght_paths"   : [str(best_wght_path.resolve())]
            }
        }
        return yolo_train_rslt
        
    # =============================================
    def _prepare_data_yaml(self, stage_name, domain, data_yaml_path):
        """
        [名稱] Func.B-1 標註資料集中介資料驗證與解析函式
        [功能] 驗證 data.yaml 實體檔案狀態，提取真實目標類別數量 (nc)。
        [參數] 共計 3 組參數，以下說明:
               - stage_name     : [str] 階層任務識別名稱 (e.g., "YOLO-T-VEH")
               - domain         : [str] 目標領域識別代碼 (e.g., "VEH")
               - data_yaml_path : [Path] 標註資料集中介檔案實體路徑 (data.yaml)
        [輸出] tuple: (data_cfg: dict, err_ret: dict/None)
               - data_cfg : [dict/None] 成功時回傳 YAML 配置字典 (包含真實 nc 等)，失敗為 None
               - err_ret  : [dict/None] 失敗時回傳結構化錯誤字典 (含 404 缺檔或 400 格式錯誤)，成功為 None
        """
        data_yaml = Path(data_yaml_path)

        """ [STAGE-1] 標註資料集整備階段 """
        if not data_yaml.exists():
            # 驗證標註資料集 data.yaml 實體狀態，攔截檔案遺失異常
            rel_path = input_resolver.to_root_relative(data_yaml)
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "VALIDATE_DATA_YAML",
                stat = "WARN",
                msge = f"[檔案: {data_yaml.name}] 驗證警示 >>> 找不到指定的中介資料，請檢查檔案路徑({rel_path})"
            )
            log.FOOTER(acnt="YOLO-HALT", rslt="模型訓練暫停流程")

            err_ret = {
                "stat_code" : 404,
                "stat_msge" : "MISSING_DATA_YAML",
                "rslt_msge" : f"[{domain}] 指定的 {data_yaml.name} 檔案不存在，請重新檢視"
            }
            return None, err_ret    

        """ [STAGE-2] 標註資料集讀取階段 """
        try:
            with open(data_yaml, "r", encoding="utf-8") as f:
                # 載入 YAML 配置並提取類別數量
                data_cfg  = yaml.safe_load(f)

            # 提取類別數量: 優先取 nc，次之依據 names 清單長度計算
            class_qty = data_cfg.get("nc")
            if class_qty is None and "names" in data_cfg:
                class_qty = len(data_cfg["names"])

            if class_qty is None:
                # 驗證標註中介資料類別定義，攔截類別清單缺失異常
                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "PARSE_DATA_YAML",
                    stat = "WARN",
                    msge = f"[檔案: {data_yaml.name}] 解析警示 >>> 缺少類別數量 (nc) 或類別清單 (names) 定義"
                )
                log.FOOTER(acnt="YOLO-HALT", rslt="模型訓練暫停流程")

                err_ret = {
                    "stat_code" : 400,
                    "stat_msge" : "INVALID_DATA_YAML",
                    "rslt_msge" : f"[{domain}] 指定的 {data_yaml.name} 缺少類別定義，請重新檢查"
                }
                return None, err_ret

            data_cfg["nc"] = int(class_qty)

            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "READ_DATA_YAML",
                stat = "SUCC",
                msge = f"[檔案: {data_yaml.name}] 讀取成功 >>> 已指向標註資料集中介資料 (類別數: {class_qty})"
            )
            return data_cfg, None

        except Exception as e:
            # 檔案語法毀損或縮排錯誤攔截 (維持業務警示 WARN，精簡印出一行錯誤原因)
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "READ_DATA_YAML",
                stat = "FAIL",
                msge = f"[檔案: {data_yaml.name}] 讀取失敗 >>> 檔案內容毀損或語法錯誤: {e}"
            )
            log.FOOTER(acnt="YOLO-HALT", rslt="模型訓練暫停流程")

            err_ret = {
                "stat_code" : 400,
                "stat_msge" : "INVALID_DATA_YAML",
                "rslt_msge" : f"[{domain}] 指定的 {data_yaml.name} 格式不合法 ({e})，請重新檢查"
            }
            return None, err_ret

    # =============================================
    def _execute_yolo_train(self, stage_name, domain, bone_name, data_yaml_path, img_size, yolo_rslt_dir, train_params):
        """
        [名稱] Func.B-2 YOLO 模型擬合訓練函式
        [功能] 載入預訓練骨幹權重驅動 YOLO 模型擬合訓練，並於訓練完成時驗證且提取最佳權重 (best.pt) 歸檔至版本目錄。
        [參數] 共計 7 組參數，以下說明:
               - stage_name     : [str] 階層任務識別名稱 (e.g., "YOLO-T-VEH")
               - domain         : [str] 目標領域識別代碼 (e.g., "VEH")
               - bone_name      : [str] 基礎預訓練骨幹名稱 (e.g., "yolo11n.pt")
               - data_yaml_path : [Path] 標註資料集中介檔案實體路徑 (data.yaml)
               - img_size       : [int] 訓練輸入影像尺寸 (寬高像素)
               - yolo_rslt_dir  : [Path] YOLO 訓練資料之根目錄
               - train_params   : [dict] 訓練參數字典 (batch_size, nbs, epoch_qty, learn_rate)
        [輸出] tuple: (rslt_run_dir: Path, err_ret: dict/None)
               - rslt_run_dir : [Path/None] 成功時回傳 YOLO 產出資料目錄，失敗為 None
               - err_ret      : [dict/None] 失敗時回傳結構化錯誤字典，成功為 None
        """
        batch_size = train_params["batch_size"]
        nbs        = train_params["nbs"]
        epoch_qty  = train_params["epoch_qty"]
        learn_rate = train_params["learn_rate"]

        log.CONTENT(
            type = "MODEL",
            targ = stage_name,
            idnt = "EXECUTE_YOLO_TRAIN",
            stat = "INFO",
            msge = f"[任務: 模型擬合訓練] 啟動成功 >>> 骨幹權重({bone_name}), 輸入尺寸({img_size}px), 訓練輪次({epoch_qty})"
        )

        """ [STAGE-1] 驅動 YOLO 模型擬合訓練 """
        try:
            model = YOLO(bone_name)
            model.train(
                data       = str(data_yaml_path),
                project    = str(yolo_rslt_dir),
                name       = "run",
                exist_ok   = True,
                imgsz      = img_size,
                batch      = batch_size,
                nbs        = nbs,
                epochs     = epoch_qty,
                lr0        = learn_rate,
            )

            # 主階段最佳權重路徑: weights/<DOMAIN>/main/<VER_ID>/best.pt
            # 次階段最佳權重路徑: weights/<DOMAIN>/sub/<SUB_DOMAIN>/<VER_ID>/best.pt
            src_wght_path = yolo_rslt_dir / "run" / "weights" / "best.pt"
            dst_wght_path = yolo_rslt_dir / "best.pt"
            if src_wght_path.exists():
                dst_wght_path.parent.mkdir(parents=True, exist_ok=True)
                # 提取最佳權重檔案 (best.pt) 放入指定版本目錄
                shutil.copy2(src_wght_path, dst_wght_path)
                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "GENERATE_WEIGHTS",
                    stat = "SUCC",
                    msge = f"[檔案: {dst_wght_path.name}] 訓練成功 >>> 已完成最佳權重歸檔至版本目錄 ({dst_wght_path.parent.name})"
                )
                rslt_run_dir = yolo_rslt_dir / "run"
                return rslt_run_dir, None
                
            else:
                rel_path = input_resolver.to_root_relative(src_wght_path)
                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "GENERATE_WEIGHTS",
                    stat = "WARN",
                    msge = f"[檔案: {dst_wght_path.name}] 訓練警示 >>> 找不到指定的最佳權重，請檢查檔案路徑({rel_path})"
                )
                log.FOOTER(acnt="YOLO-HALT", rslt="模型訓練暫停流程")

                # 提早返回 404 結構化錯誤，防止後續無效流程與資料庫污染
                err_ret = {
                    "stat_code" : 404,
                    "stat_msge" : "MISSING_BEST_WEIGHTS",
                    "rslt_msge" : f"[{domain}] 未發現 [{dst_wght_path.name}] 最佳權重，請檢查檔案路徑({rel_path})",
                }
                return None, err_ret
                
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "EXECUTE_YOLO_TRAIN",
                stat = "FAIL",
                msge = f"[任務: 模型擬合訓練] 運行失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            log.FOOTER(acnt="YOLO-FAIL", rslt="模型訓練發生異常，請調閱堆疊日誌排查問題")

            err_ret = {
                "stat_code" : 500,
                "stat_msge" : "SYSTEM_ERROR",
                "rslt_msge" : f"[{stage_name}] 模型訓練過程遭遇非預期系統異常"
            }
            return None, err_ret
    
    # =============================================
    def _resolve_train_metrics(self, stage_name, rslt_run_dir):
        """
        [名稱] Func.B-3 訓練指標時序日誌解析函式
        [功能] 解析 results.csv 提取各輪次指標與最佳 mAP@0.5。
        [參數] 共計 2 組參數，以下說明:
               - stage_name   : [str] 階層任務識別名稱 (e.g., "YOLO-T-VEH")
               - rslt_run_dir : [Path] YOLO 訓練資料輸出目錄 (<WGHT_VER_DIR>/run/)
        [輸出] tuple: (metrics: dict, err_ret: dict/None)
               - metrics : [dict] 訓練指標字典 (best_map50, epoch_logs)，異常時採預設數值 (0.0)
               - err_ret : [None] 恆為 None (本函式採高韌性容錯設計，不中斷管線)
        """
        epoch_logs = []
        best_map50 = 0.0
        # 主階段時序收斂日誌路徑: weights/<DOMAIN>/main/<VER_ID>/run/results.csv
        # 次階段時序收斂日誌路徑: weights/<DOMAIN>/sub/<SUB_DOMAIN>/<VER_ID>/run/results.csv
        results_csv = rslt_run_dir / "results.csv"

        """ [STAGE-1] 時序日誌解析階段 """
        if results_csv.exists():
            try:
                rslt_csv = pd.read_csv(results_csv)
                rslt_csv.columns = rslt_csv.columns.str.strip()
                
                for _, row in rslt_csv.iterrows():
                    mAP50_csv = float(row.get("metrics/mAP50(B)", 0.0))
                    train_box_loss = float(row.get("train/box_loss", 0.0))
                    val_box_loss   = float(row.get("val/box_loss", train_box_loss))

                    epoch_logs.append({
                        "epoch_idx"  : int(row.get("epoch")),
                        "train_loss" : train_box_loss,
                        "val_loss"   : val_box_loss,
                        "map50"      : mAP50_csv,
                        "map95"      : float(row.get("metrics/mAP50-95(B)", 0.0)),
                        "prec_val"   : float(row.get("metrics/precision(B)", 0.0)),
                        "recall_val" : float(row.get("metrics/recall(B)", 0.0))
                    })
                    # 提取最佳 mAP50 數值
                    best_map50 = mAP50_csv if mAP50_csv > best_map50 else best_map50

                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "RESOLVE_RESULTS_LOGS",
                    stat = "SUCC",
                    msge = f"[日誌: {results_csv.name}] 解析成功 >>> 已取得各輪次 (Epoch) 訓練成果，最佳 mAP50: {best_map50:.4f}"
                )

            except Exception as e:
              # 檔案內容毀損或欄位異常攔截 (維持業務警示 WARN，精簡印出一行錯誤原因)
                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "RESOLVE_RESULTS_LOGS",
                    stat = "WARN",
                    msge = f"[日誌: {results_csv.name}] 解析警示 >>> 檔案內容毀損或格式異常: {e}，跳過詳細指標統計"
                )

        metrics = {
            "best_map50" : best_map50,
            "epoch_logs" : epoch_logs
        }
        
        return metrics, None

    # =============================================
    def _sync_train_database(self, stage_level, stage_name, domain, ver_id,
                             sub_domain, sub_ver_id, bbox_type, train_date,
                             algo_name, bone_name, wght_path, class_qty,
                             img_size, train_params, metrics):
        """
        [名稱] Func.B-4 訓練資料庫同步函式
        [功能] 將最佳模型權重詮釋資料 (modl_ver_info) 與輪次時序成效指標 (modl_fit_eval) 同步寫入資料庫。
        [參數] 共計 15 組參數，涵蓋階層識別、四主鍵定義、訓練參數與指標成果。
        [輸出] tuple: (sync_rslt: dict, err_ret: dict/None)
               - sync_rslt : [dict/None] 成功或局部同步時回傳狀態字典 (stat_code, stat_msge, rslt_msge)，失敗為 None
               - err_ret   : [dict/None] 失敗時回傳結構化錯誤字典 (包含 500 系統錯誤與 details)，成功為 None
        """
        proj_name = self.algo_cfg.yolo_t.proj_name
        db_name   = f"{proj_name}-DB"

        batch_size = train_params["batch_size"]
        nbs = train_params["nbs"]
        epoch_qty  = train_params["epoch_qty"]
        learn_rate = train_params["learn_rate"]

        """ [STAGE-1] 模型版本資訊彙整 """
        ver_info_record = {
            "main_domain" : domain,           # 主領域識別代碼 (e.g., "VEH")          ※ 次階層時恆保留所屬主領域
            "main_ver_id" : ver_id,           # 主領域版本代碼 (e.g., "v1.0.0")       ※ 次階層時恆保留所屬主版本
            "sub_domain"  : sub_domain,       # 次領域識別代碼 (e.g., "LIC", "BRAND") ※ Stage-1 主階層恆寫入 "NONE"
            "sub_ver_id"  : sub_ver_id,       # 次領域版本代碼 (e.g., "v1.1.0")       ※ Stage-1 主階層恆寫入 "NONE"
            "stage_level" : stage_level,      # 當前模型階層 (1: Stage-1 宏觀檢測, 2: Stage-2 微觀檢測)
            "bbox_type"   : bbox_type,        # 邊界框類型 ("HBB": 水平框, "OBB": 旋轉框)
            "train_date"  : train_date,       # 訓練完成日期戳記 (YYYY-MM-DD)
            "algo_name"   : algo_name,        # 演算法架構 (e.g., "MARS-YOLO11")
            "bone_name"   : bone_name,        # 基礎預訓練骨幹權重 (e.g., "yolo11n.pt", "yolo11n-obb.pt")
            "wght_path"   : wght_path,        # 最佳權重相對路徑 (weights/<DOMAIN>/.../best.pt)
            "class_qty"   : class_qty,        # 該領域物件類別數量
            "img_size"    : img_size,         # 輸入影像之解析度 (e.g., 640, 480)
            "batch_size"  : batch_size,       # 訓練批次大小
            "nbs"         : nbs,              # 梯度累積步數
            "epoch_qty"   : epoch_qty,        # 總訓練輪次數
            "learn_rate"  : learn_rate,       # 最佳化初始學習率
        }

        """ [STAGE-2] 模型成效指標彙整 """
        fit_eval_records = []
        for epoch_log in metrics.get("epoch_logs", []):
            fit_eval_records.append({
                "main_domain" : domain,       # 主領域識別代碼 (e.g., "VEH")          ※ 次階層時恆保留所屬主領域
                "main_ver_id" : ver_id,       # 主領域版本代碼 (e.g., "v1.0.0")       ※ 次階層時恆保留所屬主版本
                "sub_domain"  : sub_domain,   # 次領域識別代碼 (e.g., "LIC", "BRAND") ※ Stage-1 主階層恆寫入 "NONE"
                "sub_ver_id"  : sub_ver_id,   # 次領域版本代碼 (e.g., "v1.1.0")       ※ Stage-1 主階層恆寫入 "NONE"
                **epoch_log                   # 承接純量指標與輪次字典 (包含 epoch_idx, train_loss, val_loss, map50, map95, prec_val, recall_val)
            })

        """ [STAGE-3] 模型數據寫入與檢視 """
        try:
            ver_info_stat = write_repo.train.upsert_version_info(ver_info_record)
            if not fit_eval_records:
                fit_eval_stat = False
            else:
                fit_eval_stat = write_repo.train.upsert_fit_evaluation(fit_eval_records)

            if ver_info_stat and fit_eval_stat:
                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "SYNC_TRAIN_DATABASE",
                    stat = "SUCC",
                    msge = f"[資料庫: {db_name}] 寫入成功 >>> 模型版本資訊與 {len(fit_eval_records)} 筆成效指標數據「完全同步」"
                )
                sync_rslt = {
                    "stat_code" : 200,
                    "stat_msge" : "COMPLETED",
                    "rslt_msge" : f"模型訓練與數據寫入同步圓滿完成"
                }

            else:
                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "SYNC_TRAIN_DATABASE",
                    stat = "WARN",
                    msge = f"[資料庫: {db_name}] 寫入警示 >>> 模型版本資訊或成效指標數據「局部同步」，請調閱資料表紀錄"
                )
                sync_rslt = {
                    "stat_code" : 206,
                    "stat_msge" : "PARTIAL_SUCCESS",
                    "rslt_msge" : f"模型訓練完成，但數據寫入過程發生局部遺失"
                }

            return sync_rslt, None

        except Exception as e:
            # 資料庫連線或約束異常攔截 (推送業務錯誤 FAIL，精簡印出一行錯誤原因)
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "SYNC_TRAIN_DATABASE",
                stat = "FAIL",
                msge = f"[資料庫: {db_name}] 寫入失敗 >>> 模型訓練數據同步發生異常: {e}"
            )
            log.FOOTER(acnt="YOLO-FAIL", rslt="模型訓練數據寫入異常，請檢查資料庫狀態")
            err_ret = {
                "stat_code" : 500,
                "stat_msge" : "DATABASE_SYNC_ERROR",
                "rslt_msge" : f"[{domain}] 模型訓練數據同步寫入失敗: {e}",
                "details"   : {
                    "domain"       : domain,
                    "ver_id"       : ver_id,
                    "stage_level"  : stage_level,
                    "final_map50s" : [metrics.get("best_map50", 0.0)],
                    "wght_paths"   : [wght_path]
                }
            }   
            return None, err_ret

    # =============================================
    def __call__(self, domain, ver_id, train_date, epoch_qty=None, learn_rate=None):
        """
        [名稱] Func.C 自動化模型訓練函式
        [功能] 統籌單/雙階層架構之端到端訓練流水線，整合主次階層產出並提供統一執行成果。
        [參數] 共計 5 組參數，以下說明:
               - domain     : [str] 目標領域識別代碼 (e.g., "VEH")
               - ver_id     : [str] 目標領域版本代碼 (e.g., "v1.0.0")
               - train_date : [str] 模型訓練日期 (YYYY-MM-DD)
               - epoch_qty  : [int] 總訓練輪數，若未設置則預設使用全域組態
               - learn_rate : [float] 最佳化初始學習率；若未設置則預設使用全域組態
               # Remark-1 : batch_size 與 nbs 鎖定於 algo_config 中內部讀取，注意硬體規格，防範顯存溢出
        [輸出] dict: 內含執行結果之結構化字典，共計 4 組鍵值，以下說明:
               - stat_code : [int] 執行狀態代碼 (200-成功, 206-局部落地, 400-格式錯誤, 404-缺檔, 500-系統異常)
               - stat_msge : [str] 狀態識別標籤 (e.g., COMPLETED, PARTIAL_SUCCESS, MISSING_DATA_YAML, SYSTEM_ERROR)
               - rslt_msge : [str] 執行結果之詳細文字說明
               - details   : [dict] 共計 6 組任務指標明細，以下說明:
                 - domain       : [str] 目標領域識別代碼 (e.g., "VEH")
                 - ver_id       : [str] 目標領域版本代碼 (e.g., v1.0.0)
                 - stage_qty    : [int] 模型階層數量 (1: 單階層全圖訓練, 2: 二階層串聯裁切投影與微觀訓練)
                 - sub_domains  : [list] 次領域識別代碼清單，若為單階層則為 []
                 - final_map50s : [list] 模型收斂後之最佳 mAP@0.5 精度表現清單
                 - wght_paths   : [list] 產出之最佳權重絕對路徑清單
        """
        domain = str(domain).strip().upper()
        stage_name = f"YOLO-T-{domain}"
        log.BANNER(acnt="YOLO-TASK", msin=f"模型訓練任務，載入 [{domain}] 資料集 (核心版本-{ver_id})")

        try:
            """ [STAGE-1] 階層策略驗證階段 """
            stage_rule = self.stage_cfg.get(domain, {})

            if not stage_rule:
                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "VALIDATE_STAGE_RULE",
                    stat = "WARN",
                    msge = f"[領域: {domain}] 驗證警示 >>> 未定義階層策略規則，無法進行模型訓練"
                )
                log.FOOTER(acnt="YOLO-HALT", rslt="模型訓練暫停流程")

                yolo_mngr_info = {
                    "stat_code" : 400,
                    "stat_msge" : "MISSING_STAGE_RULE",
                    "rslt_msge" : f"[{domain}] 未定義階層策略規則，請檢查 stage_maps",
                    "details"   : {
                        "domain"       : domain,
                        "ver_id"       : ver_id,
                        "stage_qty"    : 0,
                        "sub_domains"  : [],
                        "final_map50s" : [],
                        "wght_paths"   : []
                    }
                }
                return yolo_mngr_info

            stage_qty = stage_rule.get("stage_qty", 1)

            """ [STAGE-2] 訓練參數解析與任務指標初始化階段 """
            yolo_t = self.algo_cfg.yolo_t
            train_params = {
                "batch_size" : yolo_t.batch_size,
                "nbs" : yolo_t.nbs,
                "epoch_qty"  : epoch_qty  if epoch_qty is not None else yolo_t.epoch_qty,
                "learn_rate" : learn_rate if learn_rate is not None else yolo_t.learn_rate
            }

            sub_domains  = []
            final_map50s = []
            wght_paths   = []
            has_partial  = False

            """ [STAGE-3] 主階層 (Stage-1) 模型擬合訓練 """
            stage_1_cfg = stage_rule["stage_1"]
            stage_1_train = self._execute_stage_pipeline(
                domain       = domain,
                ver_id       = ver_id,
                train_date   = train_date,
                stage_level  = 1,
                sub_domain   = "NONE",
                task_spec    = stage_1_cfg,
                train_params = train_params
            )

            if stage_1_train["stat_code"] not in [200, 206]:
                # 若已完成「模型擬合訓練」但寫入資料發生錯誤 (500)，仍自 details 繼承「評估指標」與「權重路徑」
                s1_details = stage_1_train.get("details", {})
                yolo_mngr_info = {
                    "stat_code" : stage_1_train["stat_code"],
                    "stat_msge" : stage_1_train["stat_msge"],
                    "rslt_msge" : stage_1_train["rslt_msge"],
                    "details"   : {
                        "domain"       : domain,
                        "ver_id"       : ver_id,
                        "stage_qty"    : stage_qty,
                        "sub_domains"  : [],
                        "final_map50s" : s1_details.get("final_map50s", []),
                        "wght_paths"   : s1_details.get("wght_paths", []),
                    }
                }
                return yolo_mngr_info
            
            if stage_1_train["stat_code"] == 206:
                has_partial = True

            final_map50s.extend(stage_1_train["details"]["final_map50s"])
            wght_paths.extend(stage_1_train["details"]["wght_paths"])

            """ [STAGE-4] 次階層 (Stage-2) 模型擬合訓練 """
            if stage_qty >= 2:
                stage_2_cfg = stage_rule["stage_2"]
                sub_tasks   = stage_2_cfg["sub_tasks"]

                for sub_domain, sub_info in sub_tasks.items():
                    sub_domains.append(sub_domain)
                    
                    # 將次階段配置 (stage_2) 共用配置與次領域專屬配置合併
                    merger_stage_2_cfg = {
                        "task_name"  : stage_2_cfg["task_name"],
                        **sub_info
                    }

                    stage_2_train = self._execute_stage_pipeline(
                        domain       = domain,
                        ver_id       = ver_id,
                        train_date   = train_date,
                        stage_level  = 2,
                        sub_domain   = sub_domain,
                        task_spec    = merger_stage_2_cfg,
                        train_params = train_params
                    )

                    if stage_2_train["stat_code"] not in [200, 206]:
                        # 若已完成「模型擬合訓練」但寫入資料發生錯誤 (500)，仍自 details 繼承「評估指標」與「權重路徑」
                        s2_details  = stage_2_train.get("details", {})
                        fail_map50s = list(final_map50s) + s2_details.get("final_map50s", [])
                        fail_wghtps = list(wght_paths) + s2_details.get("wght_paths", [])
                        yolo_mngr_info = {
                            "stat_code" : stage_2_train["stat_code"],
                            "stat_msge" : stage_2_train["stat_msge"],
                            "rslt_msge" : stage_2_train["rslt_msge"],
                            "details"   : {
                                "domain"       : domain,
                                "ver_id"       : ver_id,
                                "stage_qty"    : stage_qty,
                                "sub_domains"  : sub_domains,
                                "final_map50s" : fail_map50s,
                                "wght_paths"   : fail_wghtps
                            }
                        }
                        return yolo_mngr_info

                    if stage_2_train["stat_code"] == 206:
                        has_partial = True

                    final_map50s.extend(stage_2_train["details"]["final_map50s"])
                    wght_paths.extend(stage_2_train["details"]["wght_paths"])

            """ [STAGE-5] 訓練成果彙整階段 """
            avg_map50 = float(np.mean(final_map50s)) if final_map50s else 0.0
            log.FOOTER(acnt="YOLO-DONE", rslt=f"[{domain}] 全階層模型訓練全數完成，最佳平均精度值 (Mean mAP50) 為 {avg_map50:.4f}")

            if stage_qty == 1:
                task_stat_code = stage_1_train["stat_code"]
                task_stat_msge = stage_1_train["stat_msge"]
                task_rslt_msge = stage_1_train["rslt_msge"]
            else:
                task_stat_code = 206 if has_partial else 200
                task_stat_msge = "PARTIAL_SUCCESS" if has_partial else "COMPLETED"
                task_rslt_msge = (
                    f"[{domain}] 全階層模型訓練完成，但部分數據寫入過程發生局部遺失"
                    if has_partial
                    else f"[{domain}] 全階層模型訓練與數據寫入同步圓滿完成"
                )

            yolo_mngr_info = {
                "stat_code" : task_stat_code,
                "stat_msge" : task_stat_msge,
                "rslt_msge" : task_rslt_msge,
                "details"   : {
                    "domain"       : domain,
                    "ver_id"       : ver_id,
                    "stage_qty"    : stage_qty,
                    "sub_domains"  : sub_domains,
                    "final_map50s" : final_map50s,
                    "wght_paths"   : wght_paths
                }
            }
            return yolo_mngr_info

        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "YOLO_TRAIN_ERROR",
                stat = "FAIL",
                msge = f"[任務: 模型訓練任務] 運行失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            log.FOOTER(acnt="YOLO-FAIL", rslt="模型訓練發生異常，請調閱堆疊日誌排查問題")

            yolo_mngr_info = {
                "stat_code" : 500,
                "stat_msge" : "SYSTEM_ERROR",
                "rslt_msge" : f"[{stage_name}] 訓練過程遭遇非預期系統異常",
                "details"   : {
                    "domain"       : domain,
                    "ver_id"       : ver_id,
                    "stage_qty"    : 0,
                    "sub_domains"  : [],
                    "final_map50s" : [],
                    "wght_paths"   : []
                }
            }
            return yolo_mngr_info