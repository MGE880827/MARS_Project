# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 影片時序串流工具 (Video Sequential Stream Tool)
# 維護日期: 2026-09-19
# 檔案路徑: MARS_Project/src/vis_recg/vid_stream.py
# ##########################################################################################

# 掛載外部依賴
import cv2
import traceback
from pathlib import Path
from datetime import datetime, timedelta

# 掛載內部依賴
from utils import log
from utils.env import input_resolver

# ==========================================================================================
class VideoStreamer:
    """
    [名稱] 影片時序串流引擎 (Video Sequential Stream Engine)
    [作用] 對接全域組態環境，負責解析實體影片路徑，提供穩定之 cv2 逐幀讀取迭代器，並透過影片檔名特徵還原「每一幀之真實地理時間戳記」。
    """
    # =============================================
    def __init__(self, sys_cfg):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「影片時序串流引擎」，並掛載演算法組態 (algo) 與路徑組態 (path)。
        [參數] sys_cfg: [dict] 全域組態檢索，內含 algo_inst 與 path_inst 組態
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg  = sys_cfg
        self.algo_cfg = self.sys_cfg["algo_inst"]
        self.path_cfg = self.sys_cfg["path_inst"]

    # =============================================
    def _parse_base_time(self, src_vid_path):
        """
        [名稱] Func.B 基礎時間解析函式
        [功能] 利用檔名分析函式解析拍攝日期與時間，還原出影片第 0 幀的起始時間點。
        [參數] src_vid_path: [str/Path] 單一來源影片檔案之路徑，位於 vids_todo 下
        [輸出] datetime: 影片起始時間物件；若檔名不符規範則回傳當下時間
        """
        meta = input_resolver.analyze_video_filename(src_vid_path)
        if meta.get("is_valid"):
            date_str = meta.get("cap_date")
            time_str = meta.get("cap_time")
            try:
                base_time = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                return base_time
            except (ValueError, TypeError):
                pass
        return datetime.now()

    # =============================================
    def _get_domain_frame_step(self, src_vid_path):
        """
        [名稱] Func.C 影片採樣間距提取函式
        [功能] 解析來源影片領域與視角，獲取對應之採樣間距 (frame_step)；若無法精準匹配則推送警示並熔斷中斷。
        [參數] src_vid_path: [str/Path] 單一來源影片檔案之路徑，位於 vids_todo 下
        [輸出] int/None: 採樣間距 (1 表示不跳幀全抽；大於 1 表示每隔 N 幀抽 1 幀)；查無合法配置時回傳 None
        """
        src_vid_path = Path(src_vid_path)
        src_vid_name = src_vid_path.name
        
        meta   = input_resolver.analyze_video_filename(src_vid_path)
        domain = meta.get("domain", "UNK")
        angle  = meta.get("angle", "UNK")

        stream_cfg = self.algo_cfg.stream
        angle_step = stream_cfg.intervals.get(domain, {}).get(angle)
        if angle_step is not None:
            return max(1, int(angle_step))

        log.CONTENT(
            type = "VIDEO",
            targ = src_vid_name,
            idnt = "MISSING_STREAM_INTERVAL",
            stat = "WARN",
            msge = f"[影片: {src_vid_name}] 抽幀警示 >>> 未定義領域或視角之抽幀間距 ({domain}/{angle})，終止該影片串流"
        )
        return None

    # =============================================
    def __call__(self, src_vid_path):
        """
        [名稱] Func.D 單一影片逐幀讀取函式
        [功能] 啟動單一影片之 OpenCV 串流解析程序，依領域、視角不同自動提取採樣間距，逐幀於記憶體產出含影像矩陣、幀數編號與真實時間戳記之結構化字典。
        [參數] src_vid_path: [str/Path] 單一來源影片檔案之路徑，位於 vids_todo 下
        [輸出] generator: 每次迭代 yield 釋出 1 組結構化字典，共計 3 組鍵值，以下說明:
               - frame     : [numpy.ndarray] 當前幀數之 BGR 影像陣列 (H, W, C)
               - curr_frms : [int] 當前影像幀數編號；自 0 開始累加
               - recg_ts   : [datetime] 當前幀數所屬之時間戳記
        """
        src_vid_path = Path(src_vid_path)
        src_vid_name = src_vid_path.name

        # [STEP-1] 驗證來源影片檔案狀態，攔截路徑遺失錯誤以確保串流通道暢通
        if not src_vid_path.exists():
            log.CONTENT(
                type = "VIDEO",
                targ = src_vid_name,
                idnt = "FILE_NOT_FOUND",
                stat = "WARN",
                msge = f"[影片: {src_vid_name}] 驗證警示 >>> 該檔案不存在，無法啟動串流解析程序"
            )
            return

        cap = cv2.VideoCapture(str(src_vid_path))
        # [STEP-2] 驗證影片讀取狀態，攔截影片毀損或無法開啟者
        if not cap.isOpened():
            log.CONTENT(
                type = "VIDEO",
                targ = src_vid_name,
                idnt = "OPEN_STREAM_FAILED",
                stat = "WARN",
                msge = f"[影片: {src_vid_name}] 串流警示 >>> 無法開啟影片串流或該檔案已毀損"
            )
            return

        # [STEP-3] 提取並驗證原生 FPS 數值
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0:
            valid_fps = 30.0
            log.CONTENT(
                type = "VIDEO",
                targ = src_vid_name,
                idnt = "FPS_READ_FAILED",
                stat = "WARN",
                msge = f"[影片: {src_vid_name}] 串流警示 >>> 無法讀取有效原生 FPS (讀取值: {native_fps})，時間戳可能產生漂移偏差，自動套用安全值 ({valid_fps})"
            )
        else:
            valid_fps = float(native_fps)

        # [STEP-4] 提取採樣間距，若配置缺失則直接熔斷攔截
        step_frms = self._get_domain_frame_step(src_vid_path)   # 影片採樣間距
        if step_frms is None:
            cap.release()
            return

        totl_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))      # 影片總幀數
        base_time = self._parse_base_time(src_vid_path)         # 影片起始時間
        log.CONTENT(
            type = "VIDEO",
            targ = src_vid_name,
            idnt = "INIT_STREAM_SUCCESS",
            stat = "SUCC",
            msge = f"[影片: {src_vid_name}] 串流成功 >>> 原生FPS({valid_fps:.1f})、總幀數({totl_frms})、採樣間距({step_frms})、起始時間({base_time.strftime('%Y-%m-%d %H:%M:%S')})"
        )

        curr_frms = 0   # 當前讀取幀數進度
        succ_extr = 0   # 成功執行抽幀次數
        try:
            while True:
                ret, frm = cap.read()
                if not ret:
                    break   # 影片正常解碼完畢，跳出迴圈

                # 依據領域採樣間距 (step_frms) 進行影像過濾
                if curr_frms % step_frms == 0:
                    # 透過「累積幀數」與「有效FPS」換算經過秒數，標註影像所屬時間戳記
                    time_offset = timedelta(seconds=(curr_frms / valid_fps))
                    recg_ts = base_time + time_offset

                    # 利用 yield 釋出結構化字典封包，供辨識管線調用
                    yield {
                        "frame"     : frm,
                        "curr_frms" : curr_frms,
                        "recg_ts"   : recg_ts
                    }
                    succ_extr += 1
                curr_frms += 1

            log.CONTENT(
                type = "VIDEO",
                targ = src_vid_name,
                idnt = "READ_STREAM_COMPLETED",
                stat = "SUCC",
                msge = f"[影片: {src_vid_name}] 串流完畢 >>> 影片總幀數({totl_frms})，共計記憶體採樣載入 {succ_extr} 張影格"
            )
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "VIDEO",
                targ = src_vid_name,
                idnt = "STREAM_EXCEPTION",
                stat = "FAIL",
                msge = f"[影片: {src_vid_name}] 串流失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )

        finally:
            # 釋放 OpenCV 視訊解碼通道與硬體資源
            cap.release()
            log.CONTENT(
                type = "VIDEO",
                targ = src_vid_name,
                idnt = "RELEASE_STREAM_SUCCESS",
                stat = "SUCC",
                msge = f"[影片: {src_vid_name}] 串流結束 >>> 逐幀讀取資源已安全釋放"
            )