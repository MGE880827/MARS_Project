# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 單一影片抽幀工具 (Video Frame Extraction Tool)
# 維護日期: 2026-09-17
# 檔案路徑: MARS_Project/src/data_prep/sgl_vid_extr.py
# ##########################################################################################

# 掛載外部依賴
import cv2
import traceback
from pathlib import Path

# ==========================================================================================
class SingleVideoExtractor:
    """
    [名稱] 單一影片抽幀引擎 (Video Frame Extraction Engine)
    [作用] 負責單一影片之 OpenCV 串流解析、依指定間隔執行抽幀，並於寫入硬碟前完成高品質之影像預處理與整備。
    """
    # =============================================
    def __init__(self, step_frms, clip_limit, grid_size):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「單一影片抽幀引擎」組態，配置抽幀間隔頻率，並預載自適應光線調整處理流程。
        [參數] 共計 3 組參數，以下說明:
               - step_frms  : [int] 抽幀間隔 (單位:幀)，預設為每 N 幀抽取 1 張影像
               - clip_limit : [float] CLAHE 對比度限制閾值，數值越高明暗對比越強，但過高易導致雜訊爆光，建議值 2.0
               - grid_size  : [tuple] CLAHE 網格分割尺寸，決定局部光線運算之細膩度區塊，建議值 (8, 8)
        [輸出] None: 依定義之屬性完成初始化
        """ 
        self.step_frms = step_frms
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # ============================================= 
    def _frame_quality_filter(self, frame):
        """ 
        [名稱] Func.B 影像品質過濾函式
        [功能] 過濾抽幀影像品質，判斷當前擷取之矩陣影像是否具備整備價值。
        [參數] frame: [numpy.ndarray] OpenCV 讀取之原始影像矩陣
        [輸出] bool: 是否通過品質檢驗 (True/False)
        """
        if frame is None:
            return False
        # [NOTE] 未來可擴充 cv2.Laplacian 變異數計算: 自動過濾動態模糊過高之車牌影像

        return True
    
    # ============================================= 
    def _apply_light_alignment(self, frame):
        """ 
        [名稱] Func.C 影像光學調整函式
        [功能] 對通過檢驗之影像實施 YUV-CLAHE 直方圖均衡化，自適應優化強光反光與暗處陰影。
        [參數] frame: [numpy.ndarray] 通過品質過濾之標準影像矩陣
        [輸出] numpy.ndarray: 光線調整優化完成之影像矩陣
        """
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = self.clahe.apply(yuv[:, :, 0])
        proc_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        return proc_frame
    
    # ============================================= 
    def __call__(self, src_vid_path, dst_frms_dir):
        """ 
        [名稱] Func.D 單一影片抽幀函式
        [功能] 啟動單一影片之 OpenCV 串流解析程序，依預設間隔執行抽幀、品質過濾與光線優化，最終將合格影像轉存至指定目錄。
        [參數] 共計 2 組參數，以下說明:
               - src_vid_path : [str/Path] 來源影片檔案之路徑，位於 raw_vids 下
               - dst_frms_dir : [str/Path] 存放時序抽幀影像之目錄路徑，位於 frame/<DOMAIN>/<VID_STEM> 下
        [輸出] dict: 內含執行結果之結構化字典，共計 3 組鍵值，以下說明:
               - stat_code : [int]  執行狀態代碼 (200-成功, 404-讀取失敗, 500-系統異常)
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
        src_vid_path = Path(src_vid_path)
        dst_frms_dir = Path(dst_frms_dir)
        src_vid_name = src_vid_path.name
        src_vid_stem = src_vid_path.stem
        cap = None

        try:
            # [STEP-1] 影片讀取配置
            if not dst_frms_dir.exists():
                # 驗證抽幀影像輸出目錄狀態，攔截路徑遺失錯誤以確保寫入通道暢通
                dst_frms_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(src_vid_path))
            if not cap.isOpened():
                # 驗證影片讀取狀態，攔截影片毀損或無法開啟者
                extr_frms_info = {
                    "stat_code" : 404,
                    "details"   : {
                        "totl_frms" : 0,
                        "totl_save" : 0,
                    },
                    "log_info"  : {
                        "log_type"  : "VIDEO",
                        "log_targ"  : src_vid_name,
                        "log_idnt"  : "OPEN_VIDEO_STREAM",
                        "log_stat"  : "WARN",
                        "log_msge"  : f"[影片: {src_vid_name}] 讀取警示 >>> 無法開啟影片串流或該檔案已毀損"
                    }
                }
                return extr_frms_info

            totl_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 影片總幀數
            curr_frms = 0   # 當前讀取幀數進度
            save_frms = 0   # 成功儲存圖片數量
            
            # [STEP-2] 影像數據生成
            while True:
                ret, frame = cap.read()
                if not ret:
                    break   # 影片正常解碼完畢，跳出迴圈

                # 執行影片抽幀，含品質過濾及光線調整
                if curr_frms % self.step_frms == 0:
                    if self._frame_quality_filter(frame):
                        img = self._apply_light_alignment(frame)
                        # 存放時序抽幀影像之目錄路徑: data/01_train/frames/<DOMAIN>/<VID_STEM>
                        # 抽幀影像檔案名稱規範: <VID_STEM>_<FRM_INDX>.jpg
                        img_name = f"{src_vid_stem}_{save_frms:05d}.jpg"  
                        cv2.imwrite(str(dst_frms_dir / img_name), img)
                        save_frms += 1
                curr_frms += 1
                
            extr_frms_info = {
                "stat_code" : 200,
                "details"   : {
                    "totl_frms" : totl_frms,
                    "totl_save" : save_frms,
                },
                "log_info"  : {
                    "log_type"  : "VIDEO",
                    "log_targ"  : src_vid_name,
                    "log_idnt"  : "EXTRACT_FRAMES",
                    "log_stat"  : "SUCC",
                    "log_msge"  : f"[影片: {src_vid_name}] 抽幀成功 >>> 影片長度共計 {totl_frms} 幀，成功抽幀儲存 {save_frms} 張影像"
                }
            }
            return extr_frms_info

        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            extr_frms_info = {
                "stat_code" : 500,
                "details"   : {
                    "totl_frms" : 0,
                    "totl_save" : 0,
                },
                "log_info"  : {
                    "log_type"  : "VIDEO",
                    "log_targ"  : src_vid_name,
                    "log_idnt"  : "EXTRACT_FRAME_EXCEPTION",
                    "log_stat"  : "FAIL",
                    "log_msge"  : f"[影片: {src_vid_name}] 抽幀失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
                }
            }
            return extr_frms_info

        finally:
            # 確保無論正常離開或遭遇中斷，皆安全釋放底層影片串流控制代碼
            if cap is not None and cap.isOpened():
                cap.release()