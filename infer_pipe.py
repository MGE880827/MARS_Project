# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 辨識管線調度工具 (Inference Pipeline Orchestrator Tool)
# 維護日期: 2026-09-20
# 檔案路徑: MARS_Project/infer_pipe.py
# ##########################################################################################

# 掛載外部依賴
import sys
import argparse
import traceback
from pathlib import Path
import pandas as pd

# 掛載內部依賴
from config import algo_config, path_config, INFER_STAGE_RULES
from utils import log
from utils.env import file_manager, input_resolver
from src.vis_recg import vis_recg_insp, VisionModelPreloader, VideoStreamer, ObjectDetector, PipelineDispatcher

# ==========================================================================================
class InferPipeline:
    """
    [名稱] 辨識管線調度引擎 (Inference Pipeline Orchestrator Engine)
    [作用] 負責自動探測 vids_todo 待辨識影片，提供 GUI 或 CLI 一鍵自動化：
           I1 模型預載、I2 串流抽幀、I3 物件偵測、I4 特徵辨識
           I5 資料寫入、I6 檔案歸檔、I7 資源釋放
    """
    # =============================================
    def __init__(self, sys_cfg=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「辨識管線調度引擎」，掛載全域路徑組態、演算法組態及階層辨識策略總表 (可允許外部注入自訂組態)。
        [參數] sys_cfg: [dict] 全域組態檢索，內含 algo_inst, path_inst 與 stage_inst 組態
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg   = sys_cfg if sys_cfg else {
            "algo_inst"  : algo_config,
            "path_inst"  : path_config,
            "stage_inst" : INFER_STAGE_RULES
        }
        self.algo_cfg  = self.sys_cfg["algo_inst"]
        self.path_cfg  = self.sys_cfg["path_inst"]
        self.stage_cfg = self.sys_cfg["stage_inst"]

    # =============================================
    def __call__(self, exec_stage="infer", ver_id_dict=None, use_trt=True):
        """
        [名稱] Func.B 辨識管線調度函式
        [功能] 驅動自動化管線生命週期，自動探測待辨識影片、預載模型權重、逐幀解碼串流、執行主次階層特徵辨識，完成後自動歸檔至 vids_done。
        [參數] 共計 3 組參數，以下說明:
               - exec_stage  : [str] 辨識執行階段；預設為 'infer' (專職辨識與歸檔)
               - ver_id_dict : [dict] 選填參數；自定義各領域模型版本識別代碼字典 (e.g., {"VEH": "v1.0.0"})，未指定則自動探測最新版本
               - use_trt     : [bool] 選填參數；是否優先啟用 TensorRT 靜態加速引擎，預設為 True
        [輸出] dict: 內含執行結果之結構化統整字典，共計 4 組鍵值，以下說明:
               - exec_stage : [str] 當次執行之管線階段 ('infer')
               - stat_code  : [int] 全域執行之狀態代碼 (200-成功, 206-部分完成, 404-缺檔, 500-系統異常)
               - rslt_msge  : [str] 全域執行之文字說明
               - details    : [dict] 共計 3 組子任務明細，以下說明:
                 - preload_info : [dict] 各領域視覺辨識模型預載成果報告字典
                 - recg_info    : [dict] 各影片幀數特徵辨識成果統計字典
                 - arch_info    : [dict] 各影片 Archived 實體檔案歸檔成果字典
        """
        vids_todo_dir = Path(self.path_cfg.infer.vids_todo)
        vids_done_dir = Path(self.path_cfg.infer.vids_done)

        preload_info = {}
        recg_info    = {}
        arch_info    = {}

        log.BANNER(acnt="INFER-PIPE", msin="啟動 MARS 自動化辨識管線")

        try:
            """ [STAGE-1] 環境檢查與待辨識影片狀態探測階段 """
            if not vids_todo_dir.exists():
                vids_todo_dir.mkdir(parents=True, exist_ok=True)

            # [STEP-1] 檢驗有無「待辨識之影片檔案」(vids_todo/)
            if not vis_recg_insp.check_pending_videos_exist(vids_todo_dir):
                rel_path = input_resolver.to_root_relative(vids_todo_dir)
                log.CONTENT(
                    type = "PIPE",
                    targ = "INFER-EXEC",
                    idnt = "CHECK_PENDING_VIDEOS",
                    stat = "WARN",
                    msge = f"[目錄: {vids_todo_dir.name}] 辨識警示 >>> 未發現「待辨識之影片檔案」，請放置影片至 ({rel_path})"
                )
                log.FOOTER(acnt="PIPE-HALT", rslt="辨識管線暫停流程")

                return {
                    "exec_stage" : exec_stage,
                    "stat_code"  : 404,
                    "rslt_msge"  : f"[{vids_todo_dir.name}] 未發現待辨識之影片檔案",
                    "details"   : {
                        "preload_info" : preload_info,
                        "recg_info"    : recg_info,
                        "arch_info"    : arch_info
                    }
                }

            # [STEP-2] 掃描 vids_todo 目錄，統計待辦辨識影片總量及各領域分佈比例
            summary    = vis_recg_insp.get_pending_video_summary(vids_todo_dir)
            totl_vids  = summary.get("totl_vids", 0)
            domain_cnt = summary.get("domain_cnt", {})

            # 抽離未知領域影片，並且額外標註 (檔案名稱不合法)
            valid_domains = [k for k in domain_cnt.keys() if k != "UNK"]
            domain_str = ", ".join(valid_domains) if valid_domains else "無"
            unk_cnt = domain_cnt.get("UNK", 0)
            log.CONTENT(
                type = "PIPE",
                targ = "INFER-EXEC",
                idnt = "DETECT_PENDING_VIDEOS",
                stat = "INFO",
                msge = f"[領域: {domain_str}] 狀態探測 >>> 偵測到影片總計 {totl_vids} 部 (含格式異常/未知領域 {unk_cnt} 部)"
            )

            # [STEP-3] 解析與過濾有效影片任務與指定領域清單
            cand_vids = [
                vid for vid in vids_todo_dir.iterdir()
                if vid.is_file() and not vid.name.startswith(".") and vid.suffix.lower() in [".mp4", ".avi", ".mkv"]
            ]

            valid_vid_tasks = []
            for vid_path in cand_vids:
                meta = input_resolver.analyze_video_filename(vid_path)
                if not meta.get("is_valid", False):
                    log.CONTENT(
                        type = "PIPE",
                        targ = vid_path.name,
                        idnt = "VALIDATE_VIDEO_FILENAME",
                        stat = "WARN",
                        msge = f"[影片: {vid_path.name}] 命名警示 >>> 檔案名稱不符辨識命名規範，跳過該影片"
                    )
                    continue

                main_domain = meta["domain"]
                if ver_id_dict and main_domain not in ver_id_dict:
                    log.CONTENT(
                        type = "PIPE",
                        targ = vid_path.name,
                        idnt = "FILTER_UNASSIGNED_DOMAIN",
                        stat = "INFO",
                        msge = f"[影片: {vid_path.name}] 略過提示 >>> 該領域 ({main_domain}) 未在指定辨識清單內，自動跳過"
                    )
                    continue

                valid_vid_tasks.append((vid_path, main_domain))

            if not valid_vid_tasks:
                unk_cnt = domain_cnt.get("UNK", 0)
                missing_str = "UNK" if unk_cnt > 0 else "無合格影片"
                log.CONTENT(
                    type = "PIPE",
                    targ = "INFER-EXEC",
                    idnt = "CHECK_VALID_TASKS",
                    stat = "WARN",
                    msge = f"[領域: {missing_str}] 辨識警示 >>> 待辦影片均未通過命名規範檢驗 (皆為格式異常或未知領域)，無法執行目標辨識"
                )
                log.FOOTER(acnt="PIPE-HALT", rslt="辨識管線暫停流程")

                return {
                    "exec_stage" : exec_stage,
                    "stat_code"  : 404,
                    "rslt_msge"  : f"[{missing_str}] 待辦影片均不符合辨識規範，流程終止",
                    "details"    : {
                        "preload_info" : preload_info,
                        "recg_info"    : recg_info,
                        "arch_info"    : arch_info
                    }
                }

            """ [STAGE-2] 模型辨識全階段 (I1 模型預載, I2 串流抽幀, I3 物件偵測, I4 特徵辨識, I5 資料寫入, I6 檔案歸檔, I7 資源釋放) """
            infer_date = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
            preloader  = VisionModelPreloader(sys_cfg=self.sys_cfg)
            streamer   = VideoStreamer(sys_cfg=self.sys_cfg)
            detector   = ObjectDetector(sys_cfg=self.sys_cfg)
            dispatcher = PipelineDispatcher(sys_cfg=self.sys_cfg)

            domain_ready = {}  # 領域模型可用性狀態表 {"VEH": True/False, "MARA": True/False}

            # 循環執行各影片辨識與歸檔流程
            for vid_path, main_domain in valid_vid_tasks:
                vid_name = vid_path.name

                # [STEP-1] 視覺辨識模型預載任務
                assigned_ver = ver_id_dict.get(main_domain) if ver_id_dict else None
                if main_domain not in domain_ready:
                    # 檢查該領域是否為首次遇到；若是，則僅執行一次預載任務
                    load_report = preloader(domain=main_domain, ver_id=assigned_ver, use_trt=use_trt)
                    preload_info[main_domain] = load_report

                    # 唯有狀態碼為 200 (全數成功) 才能標記為 True，206 或異常一律標記為 False
                    is_ready = (load_report.get("stat_code") == 200)
                    domain_ready[main_domain] = is_ready

                    if not is_ready:
                        stat_code = load_report.get("stat_code", 500)
                        log.CONTENT(
                            type = "PIPE",
                            targ = main_domain,
                            idnt = "PRELOAD_FAILED",
                            stat = "FAIL",
                            msge = f"[領域: {main_domain}] 預載未達完整就緒 (代碼: {stat_code}) >>> 終止該領域所有影片之辨識排程"
                        )

                if not domain_ready[main_domain]:
                    # 若該領域已判定不可用，直接乾淨略過該影片 (不再呼叫 preloader，也不重複刷屏報錯)
                    continue

                log.BANNER(acnt="INFER-TASK", msin=f"開始處理影片 [{vid_name}] (主領域: {main_domain})")

                # 解析影片檔名元數據以獲取攝影機代碼
                meta = input_resolver.analyze_video_filename(vid_path)
                cam_id = meta.get("cam_brand", "CAM_01")
                
                # 初始化當前影片之執行追蹤紀錄
                arch_info[vid_name] = False
                recg_info[vid_name] = {"totl_frames": 0, "totl_objs": 0}

                # 取出確定成功的載入報告與版本資訊
                actual_ver = preload_info[main_domain].get("details", {}).get("ver_id") or assigned_ver

                # [STEP-2] 索取 YOLO 主模型實例
                main_model = preloader.get_yolo_model(main_domain)
                if main_model is None:
                    log.CONTENT(
                        type = "PIPE",
                        targ = main_domain,
                        idnt = "MISSING_MAIN_MODEL",
                        stat = "FAIL",
                        msge = f"[領域: {main_domain}] 預載失敗 >>> 未找到主階層 YOLO 模型實例，跳過影片 ({vid_name})"
                    )
                    continue

                # [STEP-3] 提供領域管線所需之模型實例
                pipe_inst = dispatcher(domain=main_domain, preloader=preloader)
                if pipe_inst is None:
                    log.CONTENT(
                        type = "PIPE",
                        targ = main_domain,
                        idnt = "DISPATCH_PIPE_FAILED",
                        stat = "FAIL",
                        msge = f"[領域: {main_domain}] 配發失敗 >>> 查無註冊之特徵提取管線，跳過影片 ({vid_name})"
                    )
                    continue

                stage_1_rule  = self.stage_cfg[main_domain]["stage_1"]
                frame_count   = 0   # 影片串流總幀數
                main_obj_extr = 0   # 主物件辨識數量

                # [STEP-4] 視覺辨識暨歸檔任務
                try:
                    # [STEP-4-1] Stream 影片時序串流任務
                    for packet in streamer(src_vid_path=vid_path):
                        frame     = packet["frame"]
                        curr_frms = packet["curr_frms"]
                        recg_ts   = packet["recg_ts"]
                        frame_count += 1

                        # [STEP-4-2] Object 影像物件偵測任務
                        main_detections = detector(
                            model     = main_model,
                            frame     = frame,
                            task_rule = stage_1_rule
                        )

                        # [STEP-4-3] Pipeline 領域特徵提取任務 (進行次階層特徵提取)
                        frame_meta = {
                            "cam_id"    : cam_id,
                            "vid_name"  : vid_name,
                            "frame_idx" : curr_frms,
                            "recg_ts"   : recg_ts
                        }

                        proc_results, sync_rslt, err_ret = pipe_inst(
                            frame      = frame,
                            detections = main_detections,
                            frame_meta = frame_meta
                        )
                        if proc_results:
                            main_obj_extr += len(proc_results)
                            log.CONTENT(
                                type = "FRAME",
                                targ = f"{vid_name}-{curr_frms:05d}",
                                idnt = "EXTRACT_SUCCESS",
                                stat = "SUCC",
                                msge = f"[時間: {recg_ts.strftime('%H:%M:%S')}] 成功辨識 {len(proc_results)} 組主物件特徵"
                            )

                    # 辨識數據統計
                    recg_info[vid_name] = {
                        "totl_frames" : frame_count,    # 影片串流總幀數
                        "totl_objs"   : main_obj_extr   # 主物件辨識數量
                    }

                    # [STEP-4-4] Archive 資料歸檔任務 (需將模型加載、時序串流、物件偵測及管線配發全數完成才可實施)
                    is_archived = file_manager.archive_infer_file(
                        sgl_vid_path = vid_path,
                        dst_base_dir = vids_done_dir,
                        infer_date   = infer_date,
                        ver_id       = actual_ver
                    )
                    arch_info[vid_name] = is_archived

                except Exception:
                    # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
                    sys_err = traceback.format_exc()
                    log.CONTENT(
                        type = "PIPE",
                        targ = vid_name,
                        idnt = "VIDEO_PROCESS_EXCEPTION",
                        stat = "FAIL",
                        msge = f"[影片: {vid_name}] 辨識失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
                    )
                    continue

            # [STEP-5] 顯式釋放 GPU 顯存與預載快取資源 (對應 I7 資源釋放)
            preloader.clear_cache()

            """ [STAGE-3] 全域管線輸出與代碼收斂 """
            pipe_stage = "自動化辨識管線"

            # 依照執行階段不同，動態評估模型預載與影片辨識子任務之成敗狀態，以下說明:
            # 成功代碼: 200 | 警示代碼: 206 | 錯誤代碼: 涵蓋 400/404/500 等

            # 模型預載成敗判定
            preload_fail = any(info.get("stat_code", 500) >= 400 for info in preload_info.values()) if preload_info else True
            preload_warn = any(info.get("stat_code") == 206 for info in preload_info.values()) if preload_info else False

            # 影片特徵辨識與歸檔成敗判定
            has_arch_vids     = any(arch_info.values()) if arch_info else False   # 至少一部影片完成處理
            all_vids_archived = all(arch_info.values()) if arch_info else False   # 全數影片完成處理

            infer_fail = not has_arch_vids                         # 全部影片皆未歸檔 (全數失敗)
            infer_warn = has_arch_vids and not all_vids_archived   # 部分影片歸檔成功 (局部警示)

            # 全域子任務異常與警示彙整判別
            has_all_fail = preload_fail or infer_fail
            has_all_warn = preload_warn or infer_warn
            if has_all_fail:
                stage_stat_code = 500
                pipe_acnt       = "PIPE-FAIL"
                if preload_fail and infer_fail:
                    stage_rslt_msge = f"[{pipe_stage}] 發生異常，模型預載與影片辨識歸檔皆遭遇失敗"
                elif preload_fail:
                    stage_rslt_msge = f"[{pipe_stage}] 發生異常，部分領域模型預載作業遭遇失敗"
                else:
                    stage_rslt_msge = f"[{pipe_stage}] 發生異常，影片辨識作業遭遇失敗，無檔案成功歸檔"

            elif has_all_warn:
                stage_stat_code = 206
                pipe_acnt       = "PIPE-WARN"
                if preload_warn and infer_warn:
                    stage_rslt_msge = f"[{pipe_stage}] 部分完成，模型預載與影片辨識歸檔均存在警示"
                elif preload_warn:
                    stage_rslt_msge = f"[{pipe_stage}] 部分完成，局部領域模型預載存在警示"
                else:
                    stage_rslt_msge = f"[{pipe_stage}] 部分完成，全域模型辨識成功，局部影片特徵提取或歸檔存在警示"

            else:
                stage_stat_code = 200
                pipe_acnt       = "PIPE-DONE"
                stage_rslt_msge = f"[{pipe_stage}] 順利完成，各領域模型辨識、特徵提取與檔案歸檔全數就緒"

            log.FOOTER(acnt=pipe_acnt, rslt=stage_rslt_msge)

            pipe_mngr_info = {
                "exec_stage" : exec_stage,
                "stat_code"  : stage_stat_code,
                "rslt_msge"  : stage_rslt_msge,
                "details"    : {
                    "preload_info" : preload_info,
                    "recg_info"    : recg_info,
                    "arch_info"    : arch_info
                }
            }
            return pipe_mngr_info

        except Exception:
            log.FOOTER(acnt="PIPE-FAIL", rslt="辨識管線發生異常，請調閱堆疊日誌排查問題")
            pipe_mngr_info = {
                "exec_stage" : exec_stage,
                "stat_code"  : 500,
                "rslt_msge"  : "[INFER-PIPE] 辨識管線遭遇非預期系統異常，已強制中止",
                "details"    : {
                    "preload_info" : preload_info,
                    "recg_info"    : recg_info,
                    "arch_info"    : arch_info
                }
            }
            return pipe_mngr_info


# =============================================
# ⭐｜主程式進入點｜CLI 命令列控制台
# =============================================
if __name__ == "__main__":
    """ [STAGE-1] 命令提示字元定義與解析 """
    parser = argparse.ArgumentParser(description="MARS 多重目標自動辨識系統 - 辨識管線 CLI 控制台")
    parser.add_argument(
        "--stage", type=str, default="infer", choices=["infer"],
        help="執行辨識階段；預設為 infer (專職辨識與歸檔)"
    )
    parser.add_argument(
        "--ver_id", type=str, nargs="*", default=[],
        help="模型版本代號；格式 DOMAIN:VER_ID (e.g., --ver_id VEH:v1.0.1 MARA:v1.0.0)，未設置則自動檢測最新版本"
    )
    parser.add_argument(
        "--no_trt", action="store_true",
        help="停用 TensorRT 靜態加速引擎，改採 PyTorch 原生引擎 (預設優先啟用 TRT)"
    )
    args = parser.parse_args()

    # 自動解析 CLI 傳入之參數清單轉為字典
    ver_id_dict = input_resolver.parse_kv_pair(args.ver_id, val_type=str)
    use_trt     = not args.no_trt

    """ [STAGE-2] 執行辨識管線任務 """
    pipeline = InferPipeline()
    infer_report = pipeline(
        exec_stage  = args.stage,
        ver_id_dict = ver_id_dict,
        use_trt     = use_trt
    )
    sys.exit(0 if infer_report.get("stat_code") in [200, 206] else 1)