# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 訓練管線調度工具 (Training Pipeline Orchestrator Tool)
# 維護日期: 2026-09-17
# 檔案路徑: MARS_Project/train_pipe.py
# ##########################################################################################

# 掛載外部依賴
import sys
import argparse
import pandas as pd
from pathlib import Path

# 掛載內部依賴
from config import algo_config, path_config, TRAIN_STAGE_RULES
from utils import log
from utils.env import file_manager, input_resolver
from src.data_prep import data_prep_insp, ConcurrentVideoExtractor
from src.modl_opts import modl_opts_insp, StageCoordinator, YoloModelTrainer, TensorRTExporter

# ==========================================================================================
class TrainPipeline:
    """
    [名稱] 訓練管線調度引擎 (Training Pipeline Orchestrator Engine)
    [作用] 負責自動探測 raw_vids 與 roboflow 資料狀態，提供 GUI 或 CLI 一鍵自動化：
           T1 影像抽幀、T2 光線調整
           T4 級聯協調、T5 模型訓練、T6 靜態編譯
           T7 資料寫入、T8 檔案歸檔
    """
    # =============================================
    def __init__(self, sys_cfg=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「訓練管線調度引擎」，掛載全域路徑組態、演算法組態及階層策略總表 (可允許外部注入自訂組態)。
        [參數] sys_cfg: [dict] 全域組態檢索，內含 algo_inst, path_inst 與 stage_inst 組態
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg  = sys_cfg if sys_cfg else {
            "algo_inst"  : algo_config,
            "path_inst"  : path_config,
            "stage_inst" : TRAIN_STAGE_RULES
        }
        self.algo_cfg  = self.sys_cfg["algo_inst"]
        self.path_cfg  = self.sys_cfg["path_inst"]
        self.stage_cfg = self.sys_cfg["stage_inst"]

    # =============================================
    def __call__(self, exec_stage="all", ver_id_dict=None, epoch_qty_dict=None, learn_rate_dict=None):
        """
        [名稱] Func.B 訓練管線調度函式
        [功能] 驅動自動化管線生命週期，自動驗證資料、原始影片抽幀、標註資料集載入、YOLO 模型擬合訓練與 TensorRT 靜態編譯，完成後自動歸檔。
        [參數] 共計 4 組參數，以下說明:
               - exec_stage      : [str] 訓練執行階段；'prepare'(僅抽幀), 'train'(僅訓練), 'all'(全流程)
               - ver_id_dict     : [dict] 必填參數；自定義模型版本識別代碼字典 (e.g., {"VEH": "v1.0.1", "MARA": "v2.0.0"}) 
               - epoch_qty_dict  : [dict] 選填參數；自定義總訓練輪次數字典，若未設置則預設使用全域組態 (e.g., {"VEH": 100, "MARA": 100})
               - learn_rate_dict : [dict] 選填參數；自定義最佳化初始學習率字典，若未設置則預設使用全域組態 (e.g., {"VEH": 0.02, "MARA": 0.05})
        [輸出] dict: 內含執行結果之結構化統整字典，共計 4 組鍵值，以下說明:
               - exec_stage : [str] 當次執行之管線階段 ('prepare', 'train', 'all')
               - stat_code  : [int] 全域執行之狀態代碼 (200-成功, 206-部分完成, 404-缺檔, 500-系統異常)
               - rslt_msge  : [str] 全域執行之文字說明
               - details    : [dict] 共計 5 組子階段明細，以下說明:
                 - extr_info  : [dict] 影像抽幀階段之結構化執行報告字典
                 - coord_info : [dict] 各領域階層級聯協調成果報告字典
                 - yolo_info  : [dict] 各領域 YOLO 模型訓練成果之結構化報告字典
                 - trt_info   : [dict] 各領域 TensorRT 靜態編譯成果之結構化報告字典
                 - arch_info  : [dict] 各領域 Archived 實體檔案歸檔成果之結構化報告字典
        """
        raw_vids_dir = self.path_cfg.train.raw_vids
        frames_dir   = self.path_cfg.train.frames
        roboflow_dir = self.path_cfg.train.roboflow
        archived_dir = self.path_cfg.train.archive

        # 彙整各子任務報告資料
        extr_info  = {}
        coord_info = {}
        yolo_info  = {}
        trt_info   = {}
        arch_info  = {}

        log.BANNER(acnt="TRAIN-PIPE", msin="啟動 MARS 自動化訓練管線")

        try:
            """ [STAGE-1] 資料整備階段 (T1-影像抽幀 & T2-光線調整) """
            if exec_stage in ["prepare", "all"]:
                try:
                    log.BANNER(acnt="DATA-PREP", msin="資料整備階段")

                    # [STEP-1] 檢驗有無「待訓練之標註資料集」(roboflow/)
                    if modl_opts_insp.check_unprocessed_datasets_exist(roboflow_dir):
                        active_datasets = modl_opts_insp.get_available_datasets(roboflow_dir)
                        domain_str = ", ".join(active_datasets)
                        log.CONTENT(
                            type = "PIPE",
                            targ = "DATA-PREP",
                            idnt = "CHECK_UNPROCESSED_DATASETS",
                            stat = "WARN",
                            msge = f"[領域: {domain_str}] 整備警示 >>> 偵測到「待訓練之標註資料集」，請先執行 --stage train 進行模型訓練"
                        )
                        log.FOOTER(acnt="PREP-HALT", rslt="資料整備暫停流程")
                        log.FOOTER(acnt="PIPE-HALT", rslt="訓練管線暫停流程")

                        extr_info = {
                            "stat_code" : 206,
                            "stat_msge" : "HALT_UNPROCESSED_ROBOFLOW",
                            "rslt_msge" : f"[{domain_str}] 偵測到待訓練標註資料集，請先執行 --stage train 進行模型訓練",
                            "details"   : active_datasets
                        }
                        pipe_mngr_info = {
                            "exec_stage" : exec_stage,
                            "stat_code"  : 206,
                            "rslt_msge"  : extr_info["rslt_msge"],
                            "details"    : {
                                "extr_info"  : extr_info,
                                "coord_info" : coord_info,
                                "yolo_info"  : yolo_info,
                                "trt_info"   : trt_info,
                                "arch_info"  : arch_info
                            }
                        }
                        return pipe_mngr_info

                    # [STEP-2] 檢驗有無「未標註之抽幀影像集」(frames/)
                    if data_prep_insp.check_unarchived_frames_exist(frames_dir):
                        unarchived_detail = data_prep_insp.get_unarchived_frames_detail(frames_dir)
                        domain_str = ", ".join(unarchived_detail.keys())
                        log.CONTENT(
                            type = "PIPE",
                            targ = "DATA-PREP",
                            idnt = "CHECK_UNARCHIVED_FRAMES",
                            stat = "WARN",
                            msge = f"[領域: {domain_str}] 整備警示 >>> 偵測到「未標註之抽幀影像集」，請先完成 roboflow 影像標註並上傳匯出，接續執行 --stage train 進行模型訓練"
                        )
                        log.FOOTER(acnt="PREP-HALT", rslt="資料整備暫停流程")
                        log.FOOTER(acnt="PIPE-HALT", rslt="訓練管線暫停流程")

                        extr_info = {
                            "stat_code" : 206,
                            "stat_msge" : "HALT_UNANNOTATED_FRAMES",
                            "rslt_msge" : f"[{domain_str}] 偵測到未標註之抽幀影像集，請先完成 roboflow 影像標註並上傳匯出",
                            "details"   : unarchived_detail
                        }
                        pipe_mngr_info = {
                            "exec_stage" : exec_stage,
                            "stat_code"  : 206,
                            "rslt_msge"  : extr_info["rslt_msge"],
                            "details"    : {
                                "extr_info"  : extr_info,
                                "coord_info" : coord_info,
                                "yolo_info"  : yolo_info,
                                "trt_info"   : trt_info,
                                "arch_info"  : arch_info
                            }
                        }
                        return pipe_mngr_info

                    # [STEP-3] 原始影片檢查與並行抽幀任務
                    if data_prep_insp.check_raw_video_exist(raw_vids_dir):
                        summary    = data_prep_insp.get_raw_video_summary(raw_vids_dir)
                        totl_vids  = summary.get("totl_vids", 0)
                        domain_str = ", ".join(summary.get("domain_cnt", {}).keys())
                        log.CONTENT(
                            type = "PIPE",
                            targ = "DATA-PREP",
                            idnt = "DETECT_RAW_VIDEOS",
                            stat = "INFO",
                            msge = f"[領域: {domain_str}] 狀態探測 >>> 偵測到「待抽幀之原始影片」共 {totl_vids} 部"
                        )
                        extractor   = ConcurrentVideoExtractor(sys_cfg=self.sys_cfg)
                        extr_report = extractor.launch_pipeline()
                        extr_info   = extr_report

                    else:
                        log.CONTENT(
                            type = "PIPE",
                            targ = "DATA-PREP",
                            idnt = "SKIP_VIDEO_EXTRACTION",
                            stat = "INFO",
                            msge = f"[目錄: {raw_vids_dir.name}] 整備提示 >>> 未發現「待抽幀之原始影片」，自動跳過抽幀階段"
                        )
                        extr_info = {
                            "stat_code" : 200,
                            "stat_msge" : "SKIPPED_NO_VIDEOS",
                            "rslt_msge" : f"[{raw_vids_dir.name}] 未發現待抽幀之原始影片，自動跳過抽幀階段",
                            "details"   : {}
                        }

                    # [STEP-4] 動態任務成果判別
                    if extr_info["stat_code"] == 200:
                        log.FOOTER(acnt="PREP-DONE", rslt="資料整備全數完成，原始影片抽幀順利到位")
                    elif extr_info["stat_code"] == 206:
                        log.FOOTER(acnt="PREP-WARN", rslt="資料整備部分完成，局部影片抽幀存在警示")
                    elif extr_info["stat_code"] == 400:
                        log.FOOTER(acnt="PREP-WARN", rslt="資料整備暫停流程，偵測到影片名稱格式不符規範")
                    elif extr_info["stat_code"] == 404:
                        log.FOOTER(acnt="PREP-WARN", rslt="資料整備跳過處理，未發現待抽幀原始影片")
                    else:
                        log.FOOTER(acnt="PREP-FAIL", rslt="資料整備發生異常，原始影片抽幀遭遇失敗")

                except Exception:
                    log.FOOTER(acnt="PREP-FAIL", rslt="資料整備發生異常，請調閱堆疊日誌排查問題")
                    raise

            """ [STAGE-2] 模型優化階段 (T4-級聯協調, T5-模型訓練, T6-靜態編譯, T7-資料寫入, T8-檔案歸檔) """
            if exec_stage in ["train", "all"]:
                try:
                    log.BANNER(acnt="MODL-OPTS", msin="模型優化階段")

                    # [STEP-1] 檢驗有無「待訓練之標註資料集 」(roboflow/)
                    if not modl_opts_insp.check_unprocessed_datasets_exist(roboflow_dir):
                        log.CONTENT(
                            type = "PIPE",
                            targ = "MODL-OPTS",
                            idnt = "CHECK_PENDING_DATASETS",
                            stat = "WARN",
                            msge = f"[目錄: {roboflow_dir.name}] 優化警示 >>> 未發現「待訓練之標註資料集」，請先完成 roboflow 影像標註並上傳匯出，接續執行 --stage train 進行模型訓練"
                        )
                        log.FOOTER(acnt="OPTS-HALT", rslt="模型優化暫停流程")
                        log.FOOTER(acnt="PIPE-HALT", rslt="訓練管線暫停流程")

                        pipe_mngr_info = {
                            "exec_stage" : exec_stage,
                            "stat_code"  : 404,
                            "rslt_msge"  : f"[{roboflow_dir.name}] 未發現待訓練之標註資料集，請先完成 roboflow 影像標註並上傳匯出",
                            "details"    : {
                                "extr_info"  : extr_info,
                                "coord_info" : coord_info,
                                "yolo_info"  : yolo_info,
                                "trt_info"   : trt_info,
                                "arch_info"  : arch_info
                            }
                        }
                        return pipe_mngr_info

                    # [STEP-2] 檢驗有無「領域版本標籤」(ver_id_dict)
                    if not ver_id_dict:
                        log.CONTENT(
                            type = "PIPE",
                            targ = "MODL-OPTS",
                            idnt = "VALIDATE_VERSION_PARAMS",
                            stat = "WARN",
                            msge = f"[參數: ver_id_dict] 優化警示 >>> 未提供「領域版本標籤」，無法執行模型訓練"
                        )
                        log.FOOTER(acnt="OPTS-HALT", rslt="模型優化暫停流程")
                        log.FOOTER(acnt="PIPE-HALT", rslt="訓練管線暫停流程")

                        pipe_mngr_info = {
                            "exec_stage" : exec_stage,
                            "stat_code"  : 404,
                            "rslt_msge"  : "[MODL-OPTS] 未提供 ver_id_dict 參數 (領域版本標籤)，無法執行模型訓練",
                            "details"    : {
                                "extr_info"  : extr_info,
                                "coord_info" : coord_info,
                                "yolo_info"  : yolo_info,
                                "trt_info"   : trt_info,
                                "arch_info"  : arch_info
                            }
                        }
                        return pipe_mngr_info
                                        
                    # [STEP-3] 檢驗「待訓練領域」與「版本標籤」對齊完整性
                    active_datasets = modl_opts_insp.get_available_datasets(roboflow_dir)
                    missing_domains = [d for d in active_datasets if d not in ver_id_dict]
                    if missing_domains:
                        missing_str = ", ".join(missing_domains)
                        log.CONTENT(
                            type = "PIPE",
                            targ = "MODL-OPTS",
                            idnt = "ALIGN_DOMAIN_VERSIONS",
                            stat = "WARN",
                            msge = f"[領域: {missing_str}] 優化警示 >>> 缺少部分「領域版本標籤」，無法執行模型訓練"
                        )
                        log.FOOTER(acnt="OPTS-HALT", rslt="模型優化暫停流程")
                        log.FOOTER(acnt="PIPE-HALT", rslt="訓練管線暫停流程")

                        pipe_mngr_info = {
                            "exec_stage" : exec_stage,
                            "stat_code"  : 404,
                            "rslt_msge"  : f"[{missing_str}] 尚未指定版本標籤 (ver_id_dict)，無法執行模型訓練",
                            "details"    : {
                                "extr_info"  : extr_info,
                                "coord_info" : coord_info,
                                "yolo_info"  : yolo_info,
                                "trt_info"   : trt_info,
                                "arch_info"  : arch_info
                            }
                        }
                        return pipe_mngr_info

                    # [STEP-4] 循環執行各領域最佳化流程
                    train_date = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
                    for domain in active_datasets:
                        ver_id     = ver_id_dict[domain]
                        epoch_qty  = epoch_qty_dict.get(domain) if epoch_qty_dict else None
                        learn_rate = learn_rate_dict.get(domain) if learn_rate_dict else None

                        # 預設該領域未完成歸檔
                        arch_info[domain] = False

                        # [STEP-4-1] Coord 階層級聯協調任務
                        coordinator  = StageCoordinator(sys_cfg=self.sys_cfg)
                        coord_report = coordinator(domain=domain)
                        coord_info[domain] = coord_report
                        if coord_report.get("stat_code") not in [200, 304]:
                            continue

                        # [STEP-4-2] YOLO 模型訓練任務
                        trainer = YoloModelTrainer(sys_cfg=self.sys_cfg)
                        yolo_t_report = trainer(
                            domain     = domain,
                            ver_id     = ver_id,
                            train_date = train_date,
                            epoch_qty  = epoch_qty,
                            learn_rate = learn_rate
                        )
                        yolo_info[domain] = yolo_t_report
                        if yolo_t_report.get("stat_code") not in [200, 206]:
                            continue

                        # [STEP-4-3] TensorRT 靜態編譯任務
                        src_wght_paths = yolo_t_report.get("details", {}).get("wght_paths", [])
                        exporter = TensorRTExporter()
                        trt_report = exporter(
                            domain         = domain,
                            ver_id         = ver_id,
                            src_wght_paths = src_wght_paths,
                        )
                        trt_info[domain] = trt_report
                        if trt_report.get("stat_code") != 200:
                            continue
                        
                        # [STEP-4-4] Archive 資料歸檔任務 (需將級聯協調、模型訓練及靜態編譯全數完成才可實施)
                        has_arch_vids     = False   # 至少一部影片處理
                        all_vids_archived = True    # 全數影片完成處理 
                        frames_domain_dir = frames_dir / domain

                        if frames_domain_dir.exists():
                            for stem_dir in list(frames_domain_dir.iterdir()):
                                if stem_dir.is_dir() and not stem_dir.name.startswith("."):
                                    match_vids = [
                                        vid for vid in raw_vids_dir.glob(f"{stem_dir.name}.*")
                                        if vid.is_file() and vid.suffix.lower() in [".mp4", ".avi", ".mkv"]
                                    ]
                                    if match_vids:
                                        has_arch_vids = True
                                        is_archived = file_manager.archive_train_file(
                                            sgl_vid_path = match_vids[0],
                                            dst_base_dir = archived_dir,
                                            train_date   = train_date,
                                            ver_id       = ver_id
                                        )
                                        if not is_archived:
                                            all_vids_archived = False
                        arch_info[domain] = has_arch_vids and all_vids_archived

                    # [STEP-5] 動態任務成果判別
                    opts_reports  = list(coord_info.values()) + list(yolo_info.values()) + list(trt_info.values())
                    has_opts_fail = any(task_info.get("stat_code", 500) >= 400 for task_info in opts_reports)
                    has_opts_warn = any(task_info.get("stat_code") == 206 for task_info in opts_reports)

                    # 精準蒐集異常與警示領域 (TRT 無 206 警示，400/404/500 皆歸類為 Failure)
                    # 1. 蒐集異常領域 (含狀態代碼)
                    coord_fail_domains = [f"{domain}:{info.get('stat_code')}" for domain, info in coord_info.items() if info.get("stat_code", 500) >= 400]
                    yolo_fail_domains  = [f"{domain}:{info.get('stat_code')}" for domain, info in yolo_info.items() if info.get("stat_code", 500) >= 400]
                    trt_fail_domains   = [f"{domain}:{info.get('stat_code')}" for domain, info in trt_info.items() if info.get("stat_code", 500) >= 400]
                    # 2. 蒐集警示領域 (含狀態代碼，保留 coord 與 yolo 的 206 彈性，trt 若未來有需要也順手容納)
                    coord_warn_domains = [f"{domain}:{info.get('stat_code')}" for domain, info in coord_info.items() if info.get("stat_code") == 206]
                    yolo_warn_domains  = [f"{domain}:{info.get('stat_code')}" for domain, info in yolo_info.items() if info.get("stat_code") == 206]
                    trt_warn_domains   = [f"{domain}:{info.get('stat_code')}" for domain, info in trt_info.items() if info.get("stat_code") == 206]

                    if has_opts_fail:
                        fail_reason = []
                        if coord_fail_domains : fail_reason.append(f"級聯協調失敗({', '.join(coord_fail_domains)})")
                        if yolo_fail_domains  : fail_reason.append(f"模型訓練失敗({', '.join(yolo_fail_domains)})")
                        if trt_fail_domains   : fail_reason.append(f"靜態編譯失敗({', '.join(trt_fail_domains)})")
                        if has_opts_warn:
                            if coord_warn_domains : fail_reason.append(f"級聯協調警示({', '.join(coord_warn_domains)})")
                            if yolo_warn_domains  : fail_reason.append(f"模型訓練警示({', '.join(yolo_warn_domains)})")
                            if trt_warn_domains   : fail_reason.append(f"靜態編譯警示({', '.join(trt_warn_domains)})")
                        reason_str = "、".join(fail_reason)
                        log.FOOTER(acnt="OPTS-FAIL", rslt=f"模型優化發生異常，領域分布如下: {reason_str}")

                    elif has_opts_warn:
                        warn_reason = []
                        if coord_warn_domains : warn_reason.append(f"級聯協調警示({', '.join(coord_warn_domains)})")
                        if yolo_warn_domains  : warn_reason.append(f"模型訓練警示({', '.join(yolo_warn_domains)})")
                        if trt_warn_domains   : warn_reason.append(f"靜態編譯警示({', '.join(trt_warn_domains)})")
                        reason_str = "、".join(warn_reason)
                        log.FOOTER(acnt="OPTS-WARN", rslt=f"模型優化部分完成，領域分布如下: {reason_str}")

                    else:
                        log.FOOTER(acnt="OPTS-DONE", rslt="模型優化全數完成，各領域級聯協調、模型訓練與靜態編譯全數就緒，並自動歸檔")

                except Exception:
                    log.FOOTER(acnt="OPTS-FAIL", rslt="模型優化發生異常，請調閱堆疊日誌排查問題")
                    raise

            """ [STAGE-3] 全域管線輸出與代碼收斂 """
            if exec_stage == "prepare":
                pipe_stage = "自動化訓練管線(僅整備)"
            elif exec_stage == "train":
                pipe_stage = "自動化訓練管線(僅訓練)"
            else:
                pipe_stage = "自動化訓練管線(全流程)"

            # 依照執行階段不同，動態評估抽幀與訓練子任務之成敗狀態，以下說明:
            # 成功代碼: 200 | 警示代碼: 206 | 錯誤代碼: 涵蓋 400/404/500 等

            # 資料整備階段執行情況
            prep_fail = extr_info.get("stat_code", 200) >= 400   # 預設 200 是因為當抽幀階段被跳過，此時 extr_info 是空的字典
            prep_warn = extr_info.get("stat_code") == 206

            # 模型優化階段執行情況
            opts_reports = list(coord_info.values()) + list(yolo_info.values()) + list(trt_info.values())
            opts_fail = any(task_info.get("stat_code", 500) >= 400 for task_info in opts_reports) if opts_reports else False
            opts_warn = any(task_info.get("stat_code") == 206 for task_info in opts_reports) if opts_reports else False

            # 全域子任務異常與警示彙整判別
            has_all_fail = prep_fail or opts_fail
            has_all_warn = prep_warn or opts_warn
            if has_all_fail:
                stage_stat_code = 500
                pipe_acnt = "PIPE-FAIL"
                if prep_fail and opts_fail:
                    stage_rslt_msge = f"[{pipe_stage}] 發生異常，影片抽幀與模型優化皆遭遇失敗"
                elif prep_fail:
                    stage_rslt_msge = f"[{pipe_stage}] 發生異常，原始影片抽幀作業遭遇失敗"
                else:
                    stage_rslt_msge = f"[{pipe_stage}] 發生異常，部分領域模型優化作業遭遇失敗"

            elif has_all_warn:
                stage_stat_code = 206
                pipe_acnt = "PIPE-WARN"
                if prep_warn and opts_warn:
                    stage_rslt_msge = f"[{pipe_stage}] 部分完成，影片抽幀與模型優化均存在警示"
                elif prep_warn:
                    stage_rslt_msge = f"[{pipe_stage}] 部分完成，局部影片抽幀處理存在警示"
                else:
                    stage_rslt_msge = f"[{pipe_stage}] 部分完成，全領域模型訓練成功，局部資料庫同步存在警示"

            else:
                stage_stat_code = 200
                pipe_acnt = "PIPE-DONE"
                if exec_stage == "prepare":
                    stage_rslt_msge = f"[{pipe_stage}] 順利完成，原始影片抽幀與光學整備作業全數就緒"
                elif exec_stage == "train":
                    stage_rslt_msge = f"[{pipe_stage}] 順利完成，各領域級聯協調、模型訓練與靜態編譯全數就緒"
                else:
                    stage_rslt_msge = f"[{pipe_stage}] 順利完成，資料整備與各領域模型優化全數就緒"

            log.FOOTER(acnt=pipe_acnt, rslt=stage_rslt_msge)

            pipe_mngr_info = {
                "exec_stage" : exec_stage,
                "stat_code"  : stage_stat_code,
                "rslt_msge"  : stage_rslt_msge,
                "details"    : {
                    "extr_info"  : extr_info,
                    "coord_info" : coord_info,
                    "yolo_info"  : yolo_info,
                    "trt_info"   : trt_info,
                    "arch_info"  : arch_info
                }
            }
            return pipe_mngr_info

        except Exception:
            log.FOOTER(acnt="PIPE-FAIL", rslt="訓練管線發生異常，請調閱堆疊日誌排查問題")
            pipe_mngr_info = {
                "exec_stage" : exec_stage,
                "stat_code"  : 500,
                "rslt_msge"  : "[TRAIN-PIPE] 訓練管線遭遇非預期系統異常，已強制中止",
                "details"    : {
                    "extr_info"  : extr_info,
                    "coord_info" : coord_info,
                    "yolo_info"  : yolo_info,
                    "trt_info"   : trt_info,
                    "arch_info"  : arch_info
                }
            }
            return pipe_mngr_info

# =============================================
# ⭐｜主程式進入點｜CLI 命令列控制台
# =============================================
if __name__ == "__main__":
    """ [STAGE-1] 命令提示字元定義與解析 """
    parser = argparse.ArgumentParser(description="MARS 多重目標自動辨識系統 - 訓練管線 CLI 控制台")
    parser.add_argument(
        "--stage", type=str, default="all", choices=["prepare", "train", "all"],
        help="執行訓練階段；類型 prepare(僅抽幀), train(僅訓練與歸檔), all(全流程)"
    )
    parser.add_argument(
        "--ver_id", type=str, nargs="+", default=[],
        help="模型版本代號；格式 DOMAIN:VER_ID (e.g., --ver_id VEH:v1.0.1 MARA:v1.0.0)"
    )
    parser.add_argument(
        "--epochs", type=str, nargs="*", default=[],
        help="總訓練輪次數；格式 DOMAIN:EPOCHS (e.g., --epochs VEH:100 MARA:50)"
    )
    parser.add_argument(
        "--lr", type=str, nargs="*", default=[],
        help="初始學習率；格式 DOMAIN:LR (e.g., --lr VEH:0.01 MARA:0.02)"
    )
    args = parser.parse_args()
    # 自動解析 CLI 傳入之參數清單轉為字典
    ver_id_dict     = input_resolver.parse_kv_pair(args.ver_id, val_type=str)
    epoch_qty_dict  = input_resolver.parse_kv_pair(args.epochs, val_type=int)
    learn_rate_dict = input_resolver.parse_kv_pair(args.lr, val_type=float)

    """ [STAGE-2] 待訓練領域資料集與參數完整性校驗 """
    # 僅在執行模型優化階段 (train 或 all) 時進行完整性檢查
    if args.stage in ["train", "all"]:
        # [STEP-1] 探測當前 roboflow 目錄下所有待訓練之領域
        roboflow_dir = path_config.train.roboflow
        active_datasets = modl_opts_insp.get_available_datasets(roboflow_dir)

        # [STEP-2] 檢查是否遺漏 --ver_id 參數情況
        if not ver_id_dict:
            parser.error("執行模型優化期間 (--stage train 或 all)，必須提供 --ver_id 參數 (e.g., --ver_id VEH:v1.0.1)")

        # [STEP-3] 嚴格比對階段，是否遺漏待訓練領域之版本代號
        missing_domain = [domain for domain in active_datasets if domain not in ver_id_dict]
        if missing_domain:
            missing_str = ", ".join(missing_domain)
            parser.error(
                f"偵測到待訓練領域 ({missing_str}) 未指定版本！請補全指令 (--ver_id " + 
                " ".join(f"{d}:v1.0.0" for d in active_datasets) + ")"
            )
            
    """ [STAGE-3] 執行訓練管線任務 """
    pipeline = TrainPipeline()
    train_report = pipeline(
        exec_stage      = args.stage,
        ver_id_dict     = ver_id_dict,
        epoch_qty_dict  = epoch_qty_dict,
        learn_rate_dict = learn_rate_dict
    )
    sys.exit(0 if train_report.get("stat_code") in [200, 206] else 1)