# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 系統日誌紀錄工具 (System Logging Record Tool)
# 維護日期: 2026-10-11
# 檔案路徑: MARS_Project/utils/log/sys_log.py
# ##########################################################################################

# 掛載外部依賴
import os
import re
import sys
import pytz
import logging
import unicodedata
from datetime import datetime
from pathlib import Path

# 掛載內部依賴
from config.path_cfg import PROJECT_ROOT

# ==========================================================================================
class LogColors:
    """
    [名稱] 終端機色彩代碼表 (Terminal ANSI Colors)
    [作用] 提供現代 CLI 指令式介面色彩定義，區分各類資訊狀態。
    """
    RESET   = "\x1B[0m"    # 重設
    BOLD    = "\x1B[1m"    # 粗體
    DIM     = "\x1B[2m"    # 微暗

    # 文字顏色
    CYAN    = "\x1B[36m"   # 青色
    BLUE    = "\x1B[34m"   # 藍色
    GREEN   = "\x1B[32m"   # 綠色
    YELLOW  = "\x1B[33m"   # 黃色
    RED     = "\x1B[31m"   # 紅色
    MAGENTA = "\x1B[35m"   # 洋紅色
    WHITE   = "\x1B[37m"   # 白色
    GRAY    = "\x1B[90m"   # 灰色

# ==========================================================================================
class SystemLog:
    """
    [名稱] 系統日誌紀錄引擎 (System Logging Record Engine)
    [作用] 定義全專案通用之數據流監控排版邏輯，確保日誌輸出的結構化與一致性。
    """
    # 移除 ANSI 色彩代碼之正規表示式 (用於純文字存檔)
    ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    # =============================================
    def __init__(self):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「系統日誌紀錄引擎」，配置欄位寬度規格、字寬校準規則、時區參數及監控紀錄。
        [參數] None: 無需使用外部參數。
        [輸出] None: 依定義之屬性完成初始化，並啟動實體日誌路徑路由。
        """
        # 啟用 Windows 終端機 (PowerShell / CMD) 的 ANSI 色彩轉義支援
        if os.name == "nt":
            os.system("")
            
        self.ind_mod = False   # 預設關閉縮排模式
        self.ind_wdt = "  "    # 縮排採用 2 空格階層式縮排
        self.fst_wdt = 140     # 分割線與邊框基準寬度

        # 欄位規格固定寬度 (全形佔 2，半形佔 1)
        self.COL_STAT = 4      # 執行狀態
        self.COL_IDNT = 22     # 動作識別
        self.COL_TYPE = 8      # 目標類型
        self.COL_TARG = 18     # 操作對象
        self.COL_ACNT = 12     # 任務動作
        self.COL_MSIN = 38     # 任務標題
        self.COL_RSLT = 38     # 任務成果


        # 建立當地時區 
        self.tw_tz = pytz.timezone("Asia/Taipei")   
        
        # 建立全局數據監控
        self.logger = logging.getLogger("SystemLogger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # 啟動動態日期路由
        self._rotate_daily_log()

    # =============================================
    @staticmethod
    def _get_display_width(text):
        """
        [名稱] Func.B-1 字元寬度精確計算函式
        [功能] 計算中英混雜字串在終端機與等寬字型下的真實佔位寬度 (全形=2, 半形=1)。
        [參數] text : [str] 欲計算長度之原始字串。
        [輸出] int  : 回傳該字串在螢幕上的真實字元寬度。
        """
        clean_text = SystemLog.ANSI_ESCAPE.sub("", str(text))
        width = 0
        for char in str(clean_text):
            east_asian = unicodedata.east_asian_width(char)
            if east_asian in ("F", "W"):   # Fullwidth, Wide (中文全形字)
                width += 2
            else:
                width += 1
        return width

    # =============================================
    def _pad(self, text, width, align="left"):
        """
        [名稱] Func.B-2 文字等寬對齊函式
        [功能] 依據真實顯示寬度進行左右對齊與空白補齊，徹底解決中文字元破壞排版問題。
        [參數] 共計 3 組參數，以下說明:
               - text  : [str] 待格式化文字。
               - width : [int] 目標欄位規範寬度。
               - align : [str] 對齊方式，可選 'left', 'right', 'center'，預設為 'left'。
        [輸出] str : 補足空格後具備精確視覺寬度的字串。
        """
        text_str   = str(text)
        real_width = self._get_display_width(text_str)
        pad_size   = max(0, width - real_width)

        if align == "right":
            return (" " * pad_size) + text_str
        elif align == "center":
            left_pad  = pad_size // 2
            right_pad = pad_size - left_pad
            return (" " * left_pad) + text_str + (" " * right_pad)
        return text_str + (" " * pad_size)

    # =============================================
    def _get_log_path(self):
        """
        [名稱] Func.C-1 日誌路徑生成函式
        [功能] 根據執行當下的時區日期，建立結構化的日誌檔案路徑，位於 MARS_Project/logs 下。
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
        [名稱] Func.C-2 日誌檔案輪轉函式
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
            # 1.移除實體檔案 Handler，切勿動到 GUI 轉發或其他 Handler
            for old_handler in self.logger.handlers[:]:
                if isinstance(old_handler, logging.FileHandler):
                    self.logger.removeHandler(old_handler)
                    old_handler.close()   # 釋放檔案控制代碼，避免 Windows 鎖檔
            # 2.建立新的 Handler
            new_handler = logging.FileHandler(curr_log_path, encoding="utf-8")
            new_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
            self.logger.addHandler(new_handler)

    # =============================================    
    def _indent_on(self):
        """
        [名稱] Func.D 日誌縮排啟用函式
        [功能] 啟用日誌段落之層級縮排。
        [參數] None: 無需使用外部參數。
        [輸出] None: 更新內部狀態。
        """
        self.ind_mod = True

    # =============================================
    def _indent_off(self):
        """
        [名稱] Func.E 日誌縮排停用函式
        [功能] 停用日誌段落之層級縮排。
        [參數] None: 無需使用外部參數。
        [輸出] None: 更新內部狀態。
        """
        self.ind_mod = False

    # =============================================
    def _write_log(self, stat, log_msge):
        """
        [名稱] Func.F 日誌輸出函式
        [功能] 將文字寫入 Logger (自動過濾 ANSI 色彩碼存入 .log 檔) 並同步輸出於終端機。
        [參數] 共計 2 組參數，以下說明:
               - stat     : [str] 執行完畢狀態標籤 (SUCC, WARN, FAIL, INFO)。
               - log_msge : [str] 待輸出與寫入之日誌內容字串。
        [輸出] None: 直接寫入日誌檔案與輸出終端機，無外部回傳值。
        """
        clean_msge = self.ANSI_ESCAPE.sub("", log_msge)

        if stat == "SUCC":
            self.logger.info(clean_msge)
        elif stat == "FAIL":
            self.logger.error(clean_msge)
        elif stat == "WARN":
            self.logger.warning(clean_msge)
        else:
            self.logger.info(clean_msge)

        print(log_msge)

    # =============================================
    def _log_row(self, type, targ, idnt, stat, msge=""):
        """
        [名稱] Func.G 結構數據記錄函式
        [功能] 寫入結構化數據列紀錄，確保追蹤日誌之格式一致性。
        [參數] 共計 5 組參數，以下說明:
               - type : [str] 操作目標類型。
               - targ : [str] 具體操作目標。
               - idnt : [str] 動作識別標籤。 
               - stat : [str] 執行完畢狀態，由 SUCC, WARN, FAIL, INFO 四種標籤組成。
               - msge : [str] 補充描述資訊。
        [輸出] None: 依定義之格式將結果紀錄於日誌，並同步顯示於終端機上。
        """
        self._rotate_daily_log()

        # 1.狀態標籤色彩映照
        color_map = {
            "SUCC": LogColors.GREEN,
            "WARN": LogColors.YELLOW,
            "FAIL": LogColors.RED,
            "INFO": LogColors.CYAN,
        }
        stat_color = color_map.get(str(stat).strip(), LogColors.RESET)

        # 2.前段固定寬度欄位格式化 (確保中英文等寬校準)
        col_stat = self._pad(str(stat), self.COL_STAT, "left")
        col_idnt = self._pad(str(idnt), self.COL_IDNT, "left")
        col_type = self._pad(str(type), self.COL_TYPE, "left")
        col_targ = self._pad(str(targ), self.COL_TARG, "left")

        # 3.組合第一行前綴 (Header: stat  idnt  type  targ  -)
        prefix = self.ind_wdt if self.ind_mod else ""
        header_colored = (
            f"{prefix}{stat_color}{col_stat}{LogColors.RESET}  "
            f"{col_idnt}  "
            f"{LogColors.CYAN}{col_type}{LogColors.RESET}  "
            f"{col_targ}  "
            f"{LogColors.GRAY}-{LogColors.RESET}"
        )

        # 4.處理訊息內容與多行縮排
        lines = str(msge).split("\n")

        # 5.首行輸出
        first_line = f"{header_colored} {lines[0]}" if lines[0] else header_colored
        self._write_log(stat, first_line)

        # 6.多行輸出
        if len(lines) > 1:
            raw_header = f"{prefix}{col_stat}  {col_idnt}  {col_type}  {col_targ}  -"
            padding_space = " " * (self._get_display_width(raw_header) - 1)
            for line in lines[1:]:
                other_line = f"{padding_space}{LogColors.GRAY}|{LogColors.RESET} {line}"
                self._write_log(stat, other_line)

    # =============================================
    def _display_banner(self, acnt, msin):
        """ 
        [名稱] Func.H 任務啟始橫幅函式
        [功能] 輸出任務起始橫幅，建立具備時戳資訊之起始邊界。
        [參數] 共計 2 組參數，以下說明:
               - acnt : [str] 任務動作描述。 
               - msin : [str] 任務標題名稱。
        [輸出] None: 依定義之格式將結果紀錄於日誌，並同步顯示於終端機上。
        """
        self._rotate_daily_log()
        curr_datetime = datetime.now(self.tw_tz).strftime("%Y-%m-%d %H:%M:%S")
        
        # 1.固定寬度欄位格式化 (確保中英文等寬校準)
        col_acnt = self._pad(str(acnt), self.COL_ACNT, "left")
        col_msin = self._pad(str(msin), self.COL_MSIN, "left")

        if self.ind_mod:
            # 次階層 (Sub-Task): 內縮 2 空格、灰色單橫線(-)、黃色圓圈指示符(●)
            prefix     = self.ind_wdt
            split_wdt  = self.fst_wdt - self._get_display_width(prefix)
            banner_sep = f"{prefix}{LogColors.GRAY}{'-' * split_wdt}{LogColors.RESET}"
            banner_msg = (
                f"{prefix}{LogColors.YELLOW}●{LogColors.RESET} "
                f"[次任務啟動] {col_acnt}  "
                f"{LogColors.GRAY}時間:{LogColors.RESET} {curr_datetime}  "
                f"{LogColors.GRAY}名稱:{LogColors.RESET} {LogColors.BOLD}{col_msin}{LogColors.RESET}"
            )

        else:
            # 主階層 (Pipeline): 頂格無縮排、青色雙等號線(=)、青色指示符(>>>)
            prefix     = ""
            banner_sep = f"{LogColors.CYAN}{LogColors.BOLD}{'=' * self.fst_wdt}{LogColors.RESET}"
            banner_msg = (
                f"{LogColors.CYAN}{LogColors.BOLD}>>> [主管線啟動]{LogColors.RESET} {col_acnt}  "
                f"{LogColors.GRAY}時間:{LogColors.RESET} {curr_datetime}  "
                f"{LogColors.GRAY}名稱:{LogColors.RESET} {LogColors.BOLD}{col_msin}{LogColors.RESET}"
            )

        self._write_log("INFO", banner_sep)
        self._write_log("INFO", banner_msg)
        self._write_log("INFO", banner_sep)

    # =============================================
    def _display_footer(self, acnt, rslt):
        """ 
        [名稱] Func.I 任務終止橫幅函式
        [功能] 輸出任務結束頁尾，建立具備執行統計之終止邊界。
        [參數] 共計 2 組參數，以下說明:
               - acnt : [str] 任務動作描述。
               - rslt : [str] 任務成果統計。
        [輸出] None: 依定義之格式將結果紀錄於日誌，並同步顯示於終端機上。
        """
        self._rotate_daily_log()
        curr_datetime = datetime.now(self.tw_tz).strftime("%Y-%m-%d %H:%M:%S")

        # 1.固定寬度欄位格式化 (確保中英文等寬校準)
        col_acnt = self._pad(str(acnt), self.COL_ACNT, "left")
        col_rslt = self._pad(str(rslt), self.COL_RSLT, "left")

        if self.ind_mod:
            # 次階層 (Sub-Task): 內縮 2 空格、灰色單橫線(-)、綠色圓圈指示符(●)
            prefix     = self.ind_wdt
            split_wdt  = self.fst_wdt - self._get_display_width(prefix)
            footer_sep = f"{prefix}{LogColors.GRAY}{'-' * split_wdt}{LogColors.RESET}"
            footer_msg = (
                f"{prefix}{LogColors.GREEN}○{LogColors.RESET} "
                f"[次任務結束] {col_acnt}  "
                f"{LogColors.GRAY}時間:{LogColors.RESET} {curr_datetime}  "
                f"{LogColors.GRAY}結果:{LogColors.RESET} {col_rslt}  "
            )

        else:
            # 主階層 (Pipeline): 頂格無縮排、青色雙等號線(=)、青色指示符(<<<)
            prefix     = ""
            footer_sep = f"{LogColors.CYAN}{LogColors.BOLD}{'=' * self.fst_wdt}{LogColors.RESET}"
            footer_msg = (
                f"{LogColors.CYAN}{LogColors.BOLD}<<< [主管線結束]{LogColors.RESET} {col_acnt}  "
                f"{LogColors.GRAY}時間:{LogColors.RESET} {curr_datetime}  "
                f"{LogColors.GRAY}結果:{LogColors.RESET} {col_rslt}"
            )

        self._write_log("INFO", footer_sep)
        self._write_log("INFO", footer_msg)
        self._write_log("INFO", footer_sep)

# =============================================
# ⭐｜模組導出｜實例化管理對象
# =============================================
""" 使用底線標記為內部私有實例，防止外部 import * 時污染命名空間 """
_system_log = SystemLog()

""" 對外開放的正式單例調度 (公開名稱) """
# 1. system_log : [實例] 系統日誌紀錄引擎 >>> 負責管控全專案數據流之結構化排版與日誌路由。
# 2. IND_ON     : [封裝] 日誌縮排啟用函式 >>> 啟用日誌段落之層級縮排，提升子任務之視覺層次感。
# 3. IND_OFF    : [封裝] 日誌縮排停用函式 >>> 停用日誌段落之層級縮排，恢復全域標準對齊狀態。
# 4. BANNER     : [封裝] 任務啟始橫幅函式 >>> 輸出任務起始橫幅，建立具備時戳與任務名稱之起始視覺邊界。
# 5. CONTENT    : [封裝] 結構數據記錄函式 >>> 寫入結構化數據列紀錄，確保追蹤日誌之格式一致性。
# 6. FOOTER     : [封裝] 任務終止橫幅函式 >>> 輸出任務結束頁尾，建立具備成果統計與總結之任務結束視覺邊界。

system_log = _system_log
IND_ON     = _system_log._indent_on
IND_OFF    = _system_log._indent_off
BANNER     = _system_log._display_banner
CONTENT    = _system_log._log_row
FOOTER     = _system_log._display_footer