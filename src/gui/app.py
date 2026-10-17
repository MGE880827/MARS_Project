# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 使用者介面啟動工具 (User Interface Launcher Tool)
# 維護日期: 2026-10-14
# 檔案路徑: MARS_Project/src/gui/app.py
# ##########################################################################################

# 掛載外部依賴
import sys
import ctypes
import traceback
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

# 掛載內部依賴
from utils.log import system_log
from .bridge import attach_gui_log_handler, detach_gui_log_handler
from .main_window import MainWindow

# ==========================================================================================
def _setup_windows_app_id():
    """
    [名稱] Windows 應用程式識別碼註冊函式
    [功能] 向 Windows 作業系統註冊獨立 Application User Model ID，確保工作列圖示與群組獨立化。
    [參數] None: 無需外部參數
    [輸出] None: 完成系統註冊
    """
    if sys.platform == "win32":
        try:
            app_id = "mars.auto_recognition.gui.v1"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass

# ==========================================================================================
def _handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """
    [名稱] 未知例外攔截函式
    [功能] 捕捉系統中未被 try-catch 處理之嚴重錯誤，先寫入日誌檔並彈窗提示使用者，防止視窗無預警瞬間消失。
    [參數] 共計 3 組標準例外參數:
           - exc_type      : [type] 例外類別 (e.g., ZeroDivisionError、TypeError)
           - exc_value     : [Exception] 錯誤詳細訊息
           - exc_traceback : [traceback] 程式出錯行數追蹤物件
    [輸出] None: 完成錯誤寫入日誌與彈窗警示
    """
    # 若為使用者在終端機按下 Ctrl+C 手動結束程式，則正常放行退出，不觸發警示
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    system_log.logger.critical(f"GUI 異常攔截資訊:\n{err_msg}")

    # 若已有 Qt 實例則彈出警告視窗
    if QApplication.instance():
        QMessageBox.critical(
            None,
            "系統嚴重異常",
            f"應用程式發生非預期錯誤:\n{str(exc_value)}\n\n詳細除錯資訊已寫入系統日誌。"
        )

# ==========================================================================================
def run_app():
    """
    [名稱] Func.A 使用者介面啟動函式
    [功能] 初始化 Qt 應用程式實例、配置高 DPI 支援、掛載全域日誌通道、建立主視窗框架並啟動主事件迴圈。
    [參數] None: 無需外部參數
    [輸出] int : 系統退出碼 (0: 正常退出)
    """
    # Windows 平台環境識別配置
    _setup_windows_app_id()

    # 掛載未知例外攔截
    sys.excepthook = _handle_uncaught_exception

    # 配置高解析度 (High-DPI) 向量螢幕縮放策略，避免 2K/4K 螢幕或 125%/150% 縮放時造成版面模糊與位移
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # 建立 Qt 應用程式實例
    app = QApplication(sys.argv)
    app.setApplicationName("MARS-System")

    # 連接後端日誌系統，使其顯示於使用者介面
    attach_gui_log_handler(system_log.logger)

    # 實例化並展開主視窗核心框架
    main_window = MainWindow()
    main_window.show()

    # 啟動介面運作 (維持視窗開啟待命)，並於關閉後安全卸載日誌處理器
    try:
        return app.exec()
    finally:
        detach_gui_log_handler(system_log.logger)
    

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. run_app : [函式] 使用者介面啟動工具 >>> 提供系統桌面端環境配置、視窗啟動與常駐運作入口。

__all__ = ["run_app"]

# =============================================
# 🧩｜單元測試｜直接執行此工具啟動 GUI
# =============================================
if __name__ == "__main__":
    sys.exit(run_app())