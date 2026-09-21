# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 系統日誌紀錄工具 (System Logging Record Tool)
# 維護日期: 2026-07-28
# 檔案路徑: MARS_Project/utils/log/sys_log.py
# ##########################################################################################

# 掛載外部依賴
import os
import logging
import pytz
from datetime import datetime
from pathlib import Path

# 掛載內部依賴
from config.path_cfg import PROJECT_ROOT

# ==========================================================================================
class SystemLog:
    """
    [名稱] 系統日誌紀錄引擎 (System Logging Record Engine)
    [作用] 定義全專案通用之數據流監控排版邏輯，確保日誌輸出的結構化與一致性。
    """
    # =============================================
    def __init__(self):
        """
        [名稱] 類別組態配置
        [功能] 初始化「系統日誌紀錄引擎」，配置全局寬度規格、時區參數及監控紀錄。
        [參數] None: 無需使用外部參數。
        [輸出] None: 依定義之屬性完成初始化，並啟動實體日誌路徑路由。
        """
        self.ind_mod = False     # 預設關閉縮排模式
        self.fst_wdt = 170       # 分割字元預設寬度
        self.ind_wdt = " " * 4   # 縮排字元預設寬度

        self.tw_tz = pytz.timezone("Asia/Taipei")   
        
        # 建立全局數據監控
        self.logger = logging.getLogger("SystemLogger")
        self.logger.setLevel(logging.INFO)
        # 啟動動態日期路由
        self._rotate_daily_log()

    # =============================================
    def _get_log_path(self):
        """
        [名稱] 日誌路徑生成函式 (Func.A)
        [功能] 根據執行當下的時區時間，建立結構化的日誌檔案路徑 (MARS_Project/<YYYY-MM>/<YYYY-MM-DD>.log)。
        [參數] None: 無需使用外部參數。
        [輸出] Path: 建立日誌檔案路徑物件 (pathlib.Path)。
        """
        now = datetime.now(self.tw_tz)

        # 建立日誌層級資料夾: MARS_Project/logs/<YYYY-MM>/
        log_mnth_dir = PROJECT_ROOT / "logs" / now.strftime("%Y-%m")
        log_mnth_dir.mkdir(parents=True, exist_ok=True)

        return log_mnth_dir / f"{now.strftime('%Y-%m-%d')}.log"

    # =============================================
    def _rotate_daily_log(self):
        """
        [名稱] 日誌檔案輪轉函式 (Func.B)
        [功能] 檢查目前 Handler 指向的檔案路徑，若跨日則自動切換寫入目標。
        [參數] None: 無需使用外部參數。
        [輸出] None: 若跨日則內部直接更新 self.logger 屬性，不產生外部回傳值。
        """
        curr_log_path = self._get_log_path()   # 取得當下日誌檔案路徑
        needs_new_handler = True               # 預設切換旗標 (True)

        # 取得 Handler 所指向的日誌檔案，且格式須為 FileHandler
        if self.logger.handlers:
            current_handler = self.logger.handlers[0]
            if isinstance(current_handler, logging.FileHandler):
                if os.path.abspath(current_handler.baseFilename) == os.path.abspath(curr_log_path):
                    needs_new_handler = False   # 如遇日期相同，旗標值設為不切換 (False)

        # 執行跨日 Handler 切換流程
        if needs_new_handler:
            # 1.移除舊有 Handlers
            for old_handler in self.logger.handlers[:]:
                self.logger.removeHandler(old_handler)
            # 2.建立新的 Handler
            new_handler = logging.FileHandler(curr_log_path, encoding="utf-8")
            new_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
            self.logger.addHandler(new_handler)

    # =============================================    
    def _indent_on(self):
        """
        [名稱] 日誌縮排啟用函式 (Func.C)
        [功能] 啟用日誌段落之層級縮排。
        [參數] None: 無需使用外部參數。
        [輸出] None: 更新內部狀態。
        """
        self.ind_mod = True

    # =============================================
    def _indent_off(self):
        """
        [名稱] 日誌縮排停用函式 (Func.D)
        [功能] 停用日誌段落之層級縮排。
        [參數] None: 無需使用外部參數。
        [輸出] None: 更新內部狀態。
        """
        self.ind_mod = False

    # =============================================
    def _display_split(self):
        """
        [名稱] 視覺區隔輸出函式 (Func.E)
        [功能] 輸出視覺區隔線，物理隔離不同子任務之執行範疇。
        [參數] None: 無需使用外部參數。
        [輸出] None: 依定義之格式將結果紀錄於日誌，並同步顯示於終端機上。
        """
        self._rotate_daily_log()

        # 分割線格式定義，如遇縮排情況，分割字元寬度需減四，使其結尾對齊
        prefix = self.ind_wdt if self.ind_mod else ""
        width  = (self.fst_wdt-4) if self.ind_mod else self.fst_wdt
        split_line = prefix + " " * width

        self.logger.info(split_line)
        print(split_line)

    # =============================================
    def _log_row(self, type, targ, idnt, stat, msge=""):
        """
        [名稱] 結構數據記錄函式 (Func.F)
        [功能] 寫入結構化數據列紀錄，確保追蹤日誌之格式一致性。
        [參數] 共計 5 組參數，以下說明:
               - type : [str] 操作目標類型。
               - targ : [str] 具體操作目標。
               - idnt : [str] 動作識別標籤，由 <VERB>_<NOUN> 組成。 
               - stat : [str] 執行完畢狀態，由 SUCC, WARN, FAIL 三種標籤組成。
               - msge : [str] 補充描述資訊，格式定義 [<TYPE>: <TARGET>] <C-VERB><成功/失敗/警示> >>> description
        [輸出] None: 依定義之格式將結果紀錄於日誌，並同步顯示於終端機上。
        """
        self._rotate_daily_log()

        # 記錄格式定義，如遇縮排情況，記錄橫列寬度需減四，使其結尾對齊
        prefix  = self.ind_wdt if self.ind_mod else ""        
        log_msg = prefix + f"| {type:<12} | {targ:<22} | {str(idnt):<13} | {stat:<8} | {msge}"
        
        if stat == "SUCC":
            self.logger.info(log_msg)
        elif stat == "FAIL":
            self.logger.error(log_msg)
        elif stat == "WARN":
            self.logger.warning(log_msg)
        print(log_msg)
    
    # =============================================
    def _display_banner(self, acnt, msin):
        """ 
        [名稱] 任務啟始橫幅函式 (Func.G)
        [功能] 輸出任務起始橫幅，建立具備時戳資訊之起始邊界。
        [參數] 共計 2 組參數，以下說明:
               - acnt : [str] 任務動作描述，由 <VERB>-<NOUN> 組成。 
               - msin : [str] 任務標題名稱。
        [輸出] None: 依定義之格式將結果紀錄於日誌，並同步顯示於終端機上。
        """
        self._rotate_daily_log()
        curr_datetime = datetime.now(self.tw_tz).strftime("%Y-%m-%d %H:%M:%S")
        
        # 記錄格式定義，如遇縮排情況，記錄橫列寬度需減四，使其結尾對齊
        prefix = self.ind_wdt if self.ind_mod else ""
        width  = (self.fst_wdt-4) if self.ind_mod else self.fst_wdt

        top_split  = prefix + "=" * width
        banner_msg = prefix + f"| {acnt:<12} | 任務名稱: {msin:<15} | 啟動時間: {curr_datetime}"
        btm_split  = prefix + "-" * width
 
        self.logger.info(top_split)
        self.logger.info(banner_msg)
        self.logger.info(btm_split)
        print(top_split)
        print(banner_msg)
        print(btm_split)

    # =============================================
    def _display_footer(self, acnt, rslt):
        """ 
        [名稱] 任務終止橫幅函式 (Func.H)
        [功能] 輸出任務結束頁尾，建立具備執行統計之終止邊界。
        [參數] 共計 2 組參數，以下說明:
               - acnt : [str] 任務動作描述，由 <VERB>-<DONE/STOP> 組成。
               - rslt : [str] 任務成果統計。
        [輸出] None: 依定義之格式將結果紀錄於日誌，並同步顯示於終端機上。
        """
        self._rotate_daily_log()
        curr_datetime = datetime.now(self.tw_tz).strftime("%Y-%m-%d %H:%M:%S")

        # 記錄格式定義，如遇縮排情況，記錄橫列寬度需減四，使其結尾對齊
        prefix = self.ind_wdt if self.ind_mod else ""
        width  = (self.fst_wdt-4) if self.ind_mod else self.fst_wdt

        top_split  = prefix + "-" * width
        footer_msg = prefix + f"| {acnt:<12} | 執行結果: {rslt:<12} | 結束時間: {curr_datetime}"
        btm_split  = prefix + "=" * width

        self.logger.info(top_split)
        self.logger.info(footer_msg)
        self.logger.info(btm_split)
        print(top_split)
        print(footer_msg)
        print(btm_split)

# =============================================
# ⭐｜模組導出｜實例化管理對象
# =============================================
""" 使用底線標記為內部私有實例，防止外部 import * 時污染命名空間 """
_system_log = SystemLog()

""" 對外開放的正式單例調度 (公開名稱) """
# 1. system_log : [實例] 系統日誌紀錄引擎 >>> 負責管控全專案數據流之結構化排版與日誌路由。
# 2. IND_ON     : [封裝] 日誌縮排啟用函式 >>> 啟用日誌段落之層級縮排，提升子任務之視覺層次感。
# 3. IND_OFF    : [封裝] 日誌縮排停用函式 >>> 停用日誌段落之層級縮排，恢復全域標準對齊狀態。
# 4. SPLIT      : [封裝] 視覺區隔輸出函式 >>> 輸出視覺區隔線，物理隔離不同任務區塊。
# 5. BANNER     : [封裝] 任務啟始橫幅函式 >>> 輸出任務起始橫幅，建立具備時戳與任務名稱之起始視覺邊界。
# 6. CONTENT    : [封裝] 結構數據記錄函式 >>> 寫入結構化數據列紀錄，確保追蹤日誌之格式一致性。
# 7. FOOTER     : [封裝] 任務終止橫幅函式 >>> 輸出任務結束頁尾，建立具備成果統計與總結之任務結束視覺邊界。

system_log = _system_log
IND_ON     = _system_log._indent_on
IND_OFF    = _system_log._indent_off
SPLIT      = _system_log._display_split
BANNER     = _system_log._display_banner
CONTENT    = _system_log._log_row
FOOTER     = _system_log._display_footer