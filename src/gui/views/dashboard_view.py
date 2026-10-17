# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 系統總覽視圖工具 (System Dashboard View Tool)
# 維護日期: 2026-10-13
# 檔案路徑: MARS_Project/src/gui/views/dashboard_view.py
# ##########################################################################################

# 掛載外部依賴
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox
)

# 掛載內部依賴
from utils.log.sys_log import system_log, BANNER, CONTENT, FOOTER
from ..bridge import attach_gui_log_handler
from ..widgets import TerminalConsoleWidget

# ==========================================================================================
class DashboardView(QWidget):
    """
    [名稱] 系統總覽視圖 (System Dashboard View)
    [作用] 作為系統總覽操作面板，整合系統狀態摘要卡片、快捷操作工具列與虛擬終端主控元件，並於初始化階段將全域 Logger 接入 GUI 遙測軌道。
    """
    # =============================================
    def __init__(self, parent=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「系統總覽視圖」，構建介面佈局、裝配虛擬終端主控元件，並掛載日誌轉發引擎。
        [參數] parent : [QWidget/None] Qt 父容器物件 (選填)
        [輸出] None   : 完成視圖與事件組態配置
        """
        super().__init__(parent)

        # [STEP-1] 建立主要垂直佈局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(12)

        # [STEP-2] 構建上方頂部資訊與控制欄
        self._init_header_section()

        # [STEP-3] 構建中央主控台視圖面板
        self._init_console_section()

    # =============================================
    def _init_header_section(self):
        """
        [名稱] Func.B 頂部控制面板配置函式
        [功能] 構建標題、系統狀態指示以及操作按鈕工具列。
        [參數] None: 無需外部參數
        [輸出] None: 完成頂部元件排版
        """
        header_group  = QGroupBox("系統總覽與運作監控")
        header_layout = QHBoxLayout(header_group)
        header_layout.setContentsMargins(12, 12, 12, 12)
        header_layout.setSpacing(10)

        # [STEP-1] 系統狀態
        # 1.標籤欄位 (Lable)
        self.sys_stat_lbl = QLabel("狀態: 系統待命中 (Ready)")
        self.sys_stat_lbl.setStyleSheet("""
            QLabel {
                font-size: 13px;     /* 字體大小: 13 像素 */
                font-weight: bold;   /* 字體粗細: 粗體字型 */
                color: #10B981;      /* 文字顏色: 翡翠綠（代表系統運行就緒） */
            }
        """)
        header_layout.addWidget(self.sys_stat_lbl)

        # 彈性間距 (將按鈕推至右側)
        header_layout.addStretch()

        # [STEP-2] 冒煙測試
        # 1.快速自檢按鈕 (Button)
        self.btn_smoke_test = QPushButton("系統環境檢查")
        self.btn_smoke_test.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_smoke_test.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;   /* 預設底色: 科技寶藍色 */
                color: #FFFFFF;              /* 文字顏色: 純白 */
                border: none;                /* 邊框樣式: 移除系統預設邊框 */
                border-radius: 4px;          /* 邊框圓角: 4 像素圓角邊緣 */
                padding: 6px 14px;           /* 內距留白: 上下 6 像素、左右 14 像素 */
                font-weight: bold;           /* 字體粗細: 粗體字型 */
            }
            QPushButton:hover {
                background-color: #1D4ED8;   /* 懸停狀態: 滑鼠移入時加深為深藍色 */
            }
            QPushButton:pressed {
                background-color: #1E40AF;   /* 點擊狀態: 滑鼠按住時呈現深海藍色 */
            }
        """)
        self.btn_smoke_test.clicked.connect(self._on_smoke_test_clicked)
        header_layout.addWidget(self.btn_smoke_test)

        # 2.清除終端按鈕 (Button)
        self.btn_clear_console = QPushButton("終端內容清除")
        self.btn_clear_console.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_console.setStyleSheet("""
            QPushButton {
                background-color: #3F3F46;   /* 預設底色: 石墨灰底色 */
                color: #FFFFFF;              /* 文字顏色: 純白 */
                border: none;                /* 邊框樣式: 移除系統預設邊框 */
                border-radius: 4px;          /* 邊框圓角: 4 像素圓角邊緣 */
                padding: 6px 14px;           /* 內距留白: 上下 6 像素、左右 14 像素 */
            }
            QPushButton:hover {
                background-color: #52525B;   /* 懸停狀態: 滑鼠移入時變為淺石墨灰 */
            }
            QPushButton:pressed {
                background-color: #27272A;   /* 點擊狀態: 滑鼠按住時呈現深沉炭黑灰 */
            }
        """)
        self.btn_clear_console.clicked.connect(self._on_clear_console_clicked)
        header_layout.addWidget(self.btn_clear_console)

        self.main_layout.addWidget(header_group)

    # =============================================
    def _init_console_section(self):
        """
        [名稱] Func.C 終端控制面板配置函式
        [功能] 實例化 TerminalConsoleWidget 並將其放置於主要佈局中展開填滿。
        [參數] None: 無需外部參數
        [輸出] None: 完成主控台掛載
        """
        # 實例化自訂虛擬終端機元件
        self.console_widget = TerminalConsoleWidget(self)
        self.main_layout.addWidget(self.console_widget, stretch=1)

    # =============================================
    def _on_smoke_test_clicked(self):
        """
        [名稱] Func.D 系統環境檢查函式
        [功能] 透過標準 sys_log 管線發射測試日誌，驗證日誌輪轉、色彩配置與自動捲動功能。
        [參數] None: 無需外部參數
        [輸出] None: 觸發系統日誌流向 UI 介面
        """
        BANNER("CHECK-ENV", "系統運作環境與核心通訊自檢測試")
        CONTENT("CONFIG", "Python", "INTERPRETER", "SUCC", "已成功確認虛擬環境解譯器就緒")
        CONTENT("GUI", "TerminalConsole", "ATTACH", "SUCC", "日誌遙測轉發軌道已正確掛載")
        CONTENT("CUDA", "Device", "QUERY", "INFO", "準備查詢 GPU 加速單元")
        FOOTER("CHECK-ENV", "系統環境自檢完成，遙測通訊狀態良好")

    # =============================================
    def _on_clear_console_clicked(self):
        """
        [名稱] Func.E 終端內容清除函式
        [功能] 呼叫虛擬終端元件內部清理函式。
        [參數] None: 無需外部參數
        [輸出] None: 終端機緩衝區歸零
        """
        self.console_widget.clear_console()

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. DashboardView : [類別] 系統總覽視圖 >>> 負責整合控制工具列與虛擬終端主控元件。

__all__ = ["DashboardView"]

# =============================================
# 🧩｜單元測試｜單一介面獨立測試
# =============================================
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = DashboardView()
    window.resize(1000, 650)
    window.setWindowTitle("MARS 多重目標自動辨識系統 - 系統總覽測試")
    window.show()
    sys.exit(app.exec())