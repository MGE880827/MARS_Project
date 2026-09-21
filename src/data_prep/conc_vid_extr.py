# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 影片並行抽幀工具 (Concurrent Video Extraction Tool)
# 維護日期: 2026-09-21
# 檔案路徑: MARS_Project/src/data_prep/conc_vid_extr.py
# ##########################################################################################

# 掛載外部依賴
import traceback
import multiprocessing
import concurrent.futures
from pathlib import Path

# 掛載內部依賴
from utils import log
from utils.env import input_resolver
from .sgl_vid_extr import SingleVideoExtractor   # [NOTE] 同層資料夾，必須使用相對路徑

# ==========================================================================================
def global_pipeline_runner(cls_blueprint, init_args, exec_args):
    """ 
    [名稱] 跨進程任務引導函式
    [功能] 負責接收未實例化之類別藍圖，於獨立之子進程記憶體中完成物件初始化，並直接執行影片抽幀作業。
    [參數] 共計 3 組參數，以下說明:
           - cls_blueprint : [class] 尚未實例化之影片抽幀類別 (SingleVideoExtractor)
           - init_args     : [dict] 初始化該類別物件所需之配置參數
           - exec_args     : [dict] 執行影片抽幀所需之輸入與輸出路徑參數
    [輸出] dict: 內含執行結果之結構化統整字典，共計 3 組鍵值，以下說明:
            - stat_code : [int] 執行狀態代碼 (200-成功, 404-讀取失敗, 500-系統異常)
            - details   : [dict] 共計 2 組任務指標明細，以下說明:
                - totl_frms : [int] 影片抽幀之總幀數
                - totl_save : [int] 成功擷取並儲存之圖片張數
            - log_info  : [dict] 共計 5 組系統日誌標籤，以下說明:
                - log_type  : [str] 操作目標類型
                - log_targ  : [str] 具體操作目標
                - log_idnt  : [str] 動作識別標籤
                - log_stat  : [str] 執行完畢狀態
                - log_msge  : [str] 補充描述資訊       
    """
    # 在隔離之子進程記憶體中當場實例化，產生具備獨立記憶體指標之 self
    extractor_inst = cls_blueprint(**init_args)

    # 在隔離之子進程記憶體中，執行影片抽幀作業
    extr_frms_info = extractor_inst(**exec_args)
    return extr_frms_info

# ==========================================================================================
class ConcurrentVideoExtractor:
    """
    [名稱] 影片並行抽幀引擎 (Concurrent Video Extraction Engine)
    [作用] 對接全域組態環境，實施原始影片掃描、特徵路由決策，並分發任務至多核心 CPU 平行運作。
    """
    # =============================================
    def __init__(self, sys_cfg):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「影片並行抽幀引擎」，掛載並解構全域路徑組態與演算法參數。
        [參數] sys_cfg: [dict] 全域組態檢索，內含 algo_inst 與 path_inst 組態
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg  = sys_cfg
        self.path_cfg = sys_cfg["path_inst"]
        self.algo_cfg = sys_cfg["algo_inst"]

    # =============================================
    def _build_task_queue(self):    
        """
        [名稱] Func.B 任務佇列建構函式
        [功能] 掃描原始影片目錄 (raw_vids)，依據檔案名稱特徵執行自動場景分流，並配置對應之抽幀頻率。
        [參數] None: 直接讀取內部掛載之全域組態
        [輸出] tuple: 共計 2 組任務與檔名清單，以下說明:
               - task_queue   : [list] 並行抽幀任務字典清單，每個任務元素包含以下兩組核心配置:
                   - init_args  : [dict] 影片抽幀器實例化參數 (step_frms, clip_limit, grid_size)
                   - exec_args  : [dict] 抽幀作業輸入與輸出路徑 (src_vid_path, dst_frms_dir)
               - invalid_vids : [list] 命名格式不符規範之影片名稱清單 (格式: [<VID_NAME>, ...])
        """
        # 取出訓練階段所需之全域組態
        src_vids_dir = self.path_cfg.train.raw_vids     # 待抽幀之模型訓練影片
        dst_base_dir = self.path_cfg.train.frames       # 待標註之時序抽幀全景影像
        intervals    = self.algo_cfg.stream.intervals   # 視角抽幀間距
        clip_limit   = self.algo_cfg.clahe.clip_limit   # 對比限制門檻
        grid_size    = self.algo_cfg.clahe.grid_size    # 局部切塊大小

        task_queue   = []   # 並行抽幀任務字典清單
        invalid_vids = []   # 命名格式不符規範之影片名稱清單

        if not src_vids_dir.exists():
            # 驗證來源目錄狀態，攔截資料夾不存在或路徑遺失
            return task_queue, invalid_vids

        # 依照「拍攝視角」差異進行參數指派
        for sgl_vid_path in sorted(src_vids_dir.iterdir()):
            sgl_vid_name = sgl_vid_path.name
            if sgl_vid_path.is_file() and sgl_vid_path.suffix.lower() in [".mp4", ".avi", ".mkv"]:
                sgl_vid_stem = sgl_vid_path.stem

                # [STEP-1] 執行檔名特徵拆解與中介資料提取
                meta = input_resolver.analyze_video_filename(sgl_vid_path)

                if not meta.get("is_valid", False):
                    # 驗證影片檔名解析狀態，攔截命名格式不符規範者
                    invalid_vids.append(sgl_vid_name)
                    continue
                
                domain = meta["domain"]
                angle  = meta["angle"]

                step_frms    = intervals[domain][angle]
                dst_frms_dir = Path(dst_base_dir) / domain / sgl_vid_stem

                # [STEP-2] 建構並行任務佇列
                task_queue.append({
                    "init_args" : {
                        "step_frms"  : step_frms,
                        "clip_limit" : clip_limit,
                        "grid_size"  : grid_size
                    },
                    "exec_args" : {
                        "src_vid_path" : str(sgl_vid_path),
                        "dst_frms_dir" : str(dst_frms_dir)
                    }
                })
            
        return task_queue, invalid_vids

    # =============================================
    def launch_pipeline(self):
        """
        [名稱] Func.C 任務調度啟動函式
        [功能] 啟動「多核心並行抽幀」處理程序，負責分派任務並彙整執行報告。
        [參數] None: 無需使用外部參數，並自動讀取內部建構之任務佇列
        [輸出] dict: 內含執行結果之結構化字典，共計 4 組鍵值，以下說明:
               - stat_code : [int] 執行狀態代碼 (200-成功, 206-部分成功, 400-檔名異常, 404-無待執行任務, 500-系統異常)
               - stat_msge : [str] 狀態識別標籤 (e.g., COMPLETED, PARTIAL_SUCCESS, INVALID_NAME, NO_TASKS, ALL_FAILED)
               - rslt_msge : [str] 執行結果之詳細文字說明
               - details   : [dict] 共計 5 組任務指標明細，以下說明:
                 - succ_cnts : [int] 成功完成抽幀之影片數量
                 - fail_cnts : [int] 抽幀失敗或異常之影片數量
                 - fail_vids : [list] 抽幀失敗或異常之影片名稱清單
                 - totl_frms : [int] 所有任務影片之累計解碼總幀數
                 - totl_save : [int] 成功擷取儲存之累計影像總張數
        """
        src_vids_dir = self.path_cfg.train.raw_vids

        try:
            """ [STAGE-1] 任務佇列初始化階段 """
            task_queue, invalid_vids = self._build_task_queue()
            total_task = len(task_queue)
            log.BANNER(acnt="CONC-TASK", msin=f"並行抽幀任務，共計載入 {total_task} 部影片")

            if invalid_vids:
                # 驗證影片命名格式狀態，攔截檔名格式不符規範者
                invalid_vids_str = ", ".join(invalid_vids)
                log.CONTENT(
                    type = "QUEUE",
                    targ = "TaskQueue",
                    idnt = "INVALID_FILENAME_FORMAT",
                    stat = "WARN",
                    msge = f"[目錄: {src_vids_dir.name}] 並行警示 >>> 偵測到 {len(invalid_vids)} 部影片名稱格式不符規範，強制終止抽幀程序，以防訓練樣本遺漏，異常影片: [{invalid_vids_str}]"
                )
                log.FOOTER(acnt="CONC-HALT", rslt="並行抽幀暫停流程")

                conc_mngr_info = {
                    "stat_code" : 400,
                    "stat_msge" : "INVALID_FILENAME_FORMAT",
                    "rslt_msge" : f"[{src_vids_dir.name}] 偵測到 {len(invalid_vids)} 部影片名稱格式不符規範，強制終止抽幀",
                    "details"   : {
                        "succ_cnts" : 0,
                        "fail_cnts" : len(invalid_vids),
                        "fail_vids" : invalid_vids,
                        "totl_frms" : 0,
                        "totl_save" : 0
                    }
                }
                return conc_mngr_info

            if not task_queue:
                # 驗證任務佇列狀態，攔截目錄下無任何影像檔案
                log.CONTENT(
                    type = "QUEUE",
                    targ = "TaskQueue", 
                    idnt = "VALIDATE_TASK_QUEUE",
                    stat = "WARN", 
                    msge = f"[目錄: {src_vids_dir.name}] 並行警示 >>> 目錄下無任何有效之影片檔案"
                )
                log.FOOTER(acnt="CONC-HALT", rslt="並行抽幀暫停流程")

                conc_mngr_info = {
                    "stat_code" : 404,
                    "stat_msge" : "EMPTY_TASK_QUEUE",
                    "rslt_msge" : f"[{src_vids_dir.name}] 目錄下無任何有效之影片檔案",
                    "details"   : {
                        "succ_cnts" : 0,
                        "fail_cnts" : 0,
                        "fail_vids" : [],
                        "totl_frms" : 0,
                        "totl_save" : 0
                    }
                }
                return conc_mngr_info
            
            """ [STAGE-2] 並行抽幀啟動階段 """
            succ_vids_cnt = 0    # 成功完成抽幀之影片數量
            fail_vids_cnt = 0    # 抽幀失敗或異常之影片數量
            fail_vids     = []   # 抽幀失敗或異常之影片名稱清單
            totl_frms     = 0    # 所有任務影片之累計解碼總幀數
            totl_save     = 0    # 成功擷取儲存之累計影像總張數
            
            # 保留 1 顆核心給作業系統，防止當機情形發生
            max_cores = max(1, multiprocessing.cpu_count() -1)
            # 子進程映射字典，將 Future 記憶體物件與對應之實體影片路徑綁定，以利任務進度查詢與例外追蹤
            future_to_vid = {}

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
                futures = []
                for task in task_queue:
                    future = executor.submit(
                        global_pipeline_runner,
                        cls_blueprint = SingleVideoExtractor,
                        init_args     = task["init_args"],
                        exec_args     = task["exec_args"]
                    )
                    future_to_vid[future] = task["exec_args"]["src_vid_path"]
                    futures.append(future)

                # 集中式日誌管理機制
                for future in concurrent.futures.as_completed(futures):
                    src_vid_path  = future_to_vid[future]
                    src_vid_name  = Path(src_vid_path).name
                    try:
                        resp = future.result()
                        # 並行任務數據統計
                        if resp.get("stat_code") == 200:
                            succ_vids_cnt += 1
                            details = resp.get("details", {})
                            totl_frms += details.get("totl_frms", 0)
                            totl_save += details.get("totl_save", 0)
                        else:
                            fail_vids.append(src_vid_name)
                            fail_vids_cnt += 1

                        # 提取子進程日誌紀錄
                        log_info = resp.get("log_info", {})
                        log.CONTENT(
                            type = log_info.get("log_type", "VIDEO"),
                            targ = log_info.get("log_targ", src_vid_name),
                            idnt = log_info.get("log_idnt", "UNKNOWN_ACTION_IDENTIFIER"),
                            stat = log_info.get("log_stat", "WARN"),
                            msge = log_info.get("log_msge", f"[影片: {src_vid_name}] 抽幀警示 >>> 抽幀日誌資訊未完整定義，請檢查 sgl_vid_extr.py")
                        )

                    except Exception:
                        # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
                        fail_vids.append(src_vid_name)
                        fail_vids_cnt += 1
                        sys_err = traceback.format_exc()
                        log.CONTENT(
                            type = "CLASS",
                            targ = "SingleVideoExtractor",
                            idnt = "HANDLE_SUBPROCESS_EXCEPTION",
                            stat = "WARN",
                            msge = f"[影片: {src_vid_name}] 抽幀警示 >>> 子進程異常崩潰(可能因 OOM/Segfault/IPC_BRK 問題)，自動跳過該影片，堆疊資訊如下: \n{sys_err}"
                        )
            
            if fail_vids:
                fail_vids_str = ", ".join(fail_vids)
            else:
                fail_vids_str = "None"
            
            # 彙整並行任務處理之結果
            summary_msge = (
                f"\n >>> 處理成功: {succ_vids_cnt} 部影片"
                f"\n >>> 處理失敗: {fail_vids_cnt} 部影片"
                f"\n >>> 失敗影片: {fail_vids_str}"
                f"\n >>> 累計數量: {totl_frms} 幀影像"
                f"\n >>> 抽幀數量: {totl_save} 張訓練圖像"
            )
            
            # 三階動態狀態判定: 全勝 SUCC / 部分成功 WARN / 全滅 FAIL
            if fail_vids_cnt == 0:
                # 系統日誌組態
                logC_stat = "SUCC"
                logC_msge = "成功"
                logF_acnt = "CONC-DONE"
                logF_rslt = f"並行抽幀全數完成，請上傳 Roboflow 完成標註並匯出"
                # 任務狀態報告
                task_stat_code = 200
                task_stat_msge = "COMPLETED"
                task_rslt_msge = f"[{src_vids_dir.name}] 並行抽幀全數完成，共計處理 {succ_vids_cnt} 部影片"
            elif succ_vids_cnt > 0:
                # 系統日誌組態
                logC_stat = "WARN"
                logC_msge = "警示"
                logF_acnt = "CONC-WARN"
                logF_rslt = f"並行抽幀部分完成，請檢視失敗影片之堆疊日誌"
                # 任務狀態報告
                task_stat_code = 206
                task_stat_msge = "PARTIAL_SUCCESS"
                task_rslt_msge = f"[{src_vids_dir.name}] 並行抽幀部分完成，成功 {succ_vids_cnt} 部，失敗 {fail_vids_cnt} 部影片"
            else:
                # 系統日誌組態
                logC_stat = "FAIL"
                logC_msge = "失敗"
                logF_acnt = "CONC-FAIL"
                logF_rslt = f"並行抽幀全數失敗，請調閱堆疊日誌排查問題"
                # 任務狀態報告
                task_stat_code = 500
                task_stat_msge = "ALL_FAILED"
                task_rslt_msge = f"[{src_vids_dir.name}] 並行抽幀全數失敗，共計 {fail_vids_cnt} 部影片處理異常"

            log.CONTENT(
                type = "CLASS",
                targ = "ConcurrentVideoExtractor", 
                idnt = "SUMMARIZE_EXTRACTION_TASKS",
                stat = logC_stat,
                msge = f"[目錄: {src_vids_dir.name}] 並行{logC_msge} >>> 任務統計如下: {summary_msge}"
            )
            log.FOOTER(acnt=logF_acnt, rslt=logF_rslt)

            conc_mngr_info = {
                "stat_code" : task_stat_code,
                "stat_msge" : task_stat_msge,
                "rslt_msge" : task_rslt_msge,
                "details"   : {
                    "succ_cnts" : succ_vids_cnt,
                    "fail_cnts" : fail_vids_cnt,
                    "fail_vids" : fail_vids,
                    "totl_frms" : totl_frms,
                    "totl_save" : totl_save
                }
            }
            return conc_mngr_info
            
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "CLASS",
                targ = "ConcurrentVideoExtractor",
                idnt = "CONCURRENT_EXTRACTION_EXCEPTION",
                stat = "FAIL",
                msge = f"[目錄: {src_vids_dir.name}] 並行失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            log.FOOTER(acnt="CONC-FAIL", rslt="流程異常中斷，請調閱堆疊排除故障")
            raise RuntimeError(f"[FAIL] 錯誤來源 >>> [DIR] data_prep >> [FILE] conc_vid_extr.py >> [Func.C] launch_pipeline") from None