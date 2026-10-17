# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 系統日誌轉發工具 (System Log Forwarding Tool)
# 維護日期: 2026-10-14
# 檔案路徑: MARS_Project/src/gui/bridge/log_handler.py
# ##########################################################################################

# 掛載外部依賴
import logging
from PyQt6.QtCore import QObject, pyqtSignal

# ==========================================================================================
def attach_gui_log_handler(target_logger):
    """
    [名稱] 系統日誌轉發掛載函式
    [功能] 將 QtLogHandler 註冊至指定的 Logger 實例中，自動啟用日誌向 GUI 的即時轉發。
    [參數] target_logger : [logging.Logger] 欲掛載監聽的目標日誌紀錄器。
    [輸出] QtLogHandler  : 已完成掛載並啟用的轉發引擎實例。
    """
    # 防止重複掛載機制，若已存在 QtLogHandler 則直接回傳
    for handler in target_logger.handlers:
        if isinstance(handler, QtLogHandler):
            return handler
        
    handler = QtLogHandler(_log_emitter)
    # 預設僅轉發原始訊息文字 (格式化已由 sys_log 內部統一管線處理)
    handler.setFormatter(logging.Formatter("%(message)s"))
    target_logger.addHandler(handler)
    return handler

# ==========================================================================================
def detach_gui_log_handler(target_logger):
    """
    [名稱] 系統日誌轉發卸載函式
    [功能] 於 GUI 視窗關閉退出時，將 QtLogHandler 從 Logger 中安全移除，避免後續日誌觸發已銷毀之 Qt 物件。
    [參數] target_logger : [logging.Logger] 欲卸載監聽的目標日誌紀錄器。
    [輸出] None: 完成轉發處理器之卸載。
    """
    for handler in target_logger.handlers:
        if isinstance(handler, QtLogHandler):
            target_logger.removeHandler(handler)

# ==========================================================================================
class QtLogEmitter(QObject):
    """
    [名稱] 系統日誌信號發射引擎 (Log Event Emitter Engine)
    [作用] 提供基於事件迴圈之信號通道，確保背景工作產生的日誌能安全投遞至 UI 介面。
    """
    # 宣告日誌發射信號: 攜帶 2 組字串參數 (日誌層級標籤, 格式化訊息字串)
    sig_log_emitted = pyqtSignal(str, str)

# ==========================================================================================
class QtLogHandler(logging.Handler):
    """
    [名稱] 系統日誌轉發引擎 (Log Forwarding Handler)
    [作用] 繼承標準 logging.Handler，攔截全系統日誌輸出並轉換為事件信號廣播至圖形主控台。
    """
    # =============================================
    def __init__(self, emitter):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「系統日誌轉發引擎」，配置專屬之日誌信號發射對象。
        [參數] emitter : [QtLogEmitter] 負責派發日誌事件之信號發射器實例。
        [輸出] None    : 完成轉發引擎組態設置。
        """
        super().__init__()
        self.emitter = emitter

    # =============================================
    def emit(self, record):
        """
        [名稱] Func.B 系統日誌截獲函式
        [功能] 攔截日誌核心 (Logger) 產生的每筆 LogRecord，格式化為文字後透過信號向外部派發。
        [參數] record : [logging.LogRecord] 系統日誌內部標準紀錄物件。
        [輸出] None   : 觸發信號發射，不產生外部回傳值。
        """
        try:
            # 取得經由 Formatter 格式化後的純文字日誌
            msg   = self.format(record)
            level = record.levelname

            # 向 UI 介面發射日誌訊號
            self.emitter.sig_log_emitted.emit(level, msg)

        except Exception:
            # 發生非預期異常時調用標準處理避免崩潰
            self.handleError(record)

# =============================================
# ⭐｜模組導出｜實例化管理對象
# =============================================
""" 使用底線標記為內部私有實例，防止外部 import * 時污染命名空間 """
_log_emitter = QtLogEmitter()

""" 對外開放的正式單例接口 (公開名稱) """
# 1. log_emitter            : [實例] 系統日誌信號發射引擎 >>> 提供 UI 面板連接 .connect() 接收日誌訊息。
# 2. attach_gui_log_handler : [函式] 系統日誌轉發掛載函式 >>> 將系統日誌無縫接入 Qt 事件迴圈。
# 3. detach_gui_log_handler : [函式] 系統日誌轉發卸載函式 >>> 於視窗關閉時解除監聽，避免物件銷毀引發異常。

log_emitter            = _log_emitter
attach_gui_log_handler = attach_gui_log_handler
detach_gui_log_handler = detach_gui_log_handler