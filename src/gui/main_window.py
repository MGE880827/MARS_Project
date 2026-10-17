# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 主視窗核心框架工具 (Main Window Core Frame Tool)
# 維護日期: 2026-10-13
# 檔案路徑: MARS_Project/src/gui/main_window.py
# ##########################################################################################

# 掛載外部依賴
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QButtonGroup, QMessageBox
)

# 掛載內部依賴
from .bridge import PipelineWorker
from .views import DashboardView, TrainView, InferView
# from .views.database_view import DatabaseView

# ==========================================================================================
class MainWindow(QMainWindow):
    """
    [名稱] 主視窗核心框架 (Main Window Core Frame)
    [作用] 作為系統桌面端的主控台容器，統籌導覽選單、主畫面切換，並串接後端管線橋接執行緒以實現非同步任務調度。
    """
    # =============================================
    def __init__(self):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「主視窗核心框架」，設定視窗基本屬性、建立導覽按鈕群組，並完成主視窗階層式架構之設計。
        [參數] None: 無需外部傳入參數
        [輸出] None: 完成主視窗初始化
        """
        super().__init__()

        # [STEP-1] 設置視窗基本屬性
        self.setWindowTitle("MARS 多重目標自動辨識系統 (Multi-Object Auto-Recognition System)")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 680)

        # [STEP-2] 追蹤當前運行之背景橋接執行緒
        self.active_worker = None

        # [STEP-3] 建立導覽按鈕群組 (確保按鈕處於單選互斥模式)
        self.nav_button_group = QButtonGroup(self)
        self.nav_button_group.setExclusive(True)

        # [STEP-4] 建置主視窗框架
        self._init_window_structure()

        # [STEP-5] 串接後端橋接信號
        self._connect_bridge_signals()

    # =============================================
    def _init_window_structure(self):
        """
        [名稱] Func.B 主視窗框架建構函式
        [功能] 構建主視窗中央核心區塊，以水平佈局整合「左側導覽欄」與「右側多視圖分頁」。
        [參數] None: 無需外部傳入參數
        [輸出] None: 完成畫面架構之配置
        """
        # [STEP-1] 建立中央區塊與水平佈局
        central_container = QWidget(self)
        self.setCentralWidget(central_container)
        self.root_layout = QHBoxLayout(central_container)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        # [STEP-2] 左側導覽欄
        self.sidebar_container = self._create_sidebar()
        self.root_layout.addWidget(self.sidebar_container)

        # [STEP-3] 右側多視圖分頁 (Stretch Factor=1 使其填滿剩餘空間)
        self.content_stack = QStackedWidget(self)
        self.root_layout.addWidget(self.content_stack, stretch=1)

        # [STEP-4] 裝載各業務功能之視圖分頁
        self._setup_view_pages()

        # [STEP-5] 導覽欄與多視圖分頁訊號對接 (按鈕 ID 對應所屬分頁索引)
        self.nav_button_group.idClicked.connect(self.content_stack.setCurrentIndex)

        # [STEP-6] 預設初始化面 (系統總覽)
        self.btn_nav_dashboard.setChecked(True)
        self.content_stack.setCurrentIndex(0)

    # =============================================
    def _create_sidebar(self):
        """
        [名稱] Func.C 左側導覽欄配置函式
        [功能] 構建左側固定寬度導覽側邊欄，包含系統標題、副標題、分割線、導覽按鈕群組與版本宣告標籤。
        [參數] None    : 無需外部傳入參數
        [輸出] QWidget : 完成左側導覽欄佈局
        """
        sidebar_container = QWidget()
        sidebar_container.setFixedWidth(220)
        sidebar_container.setStyleSheet("""
            QWidget {
                background-color: #0F172A;         /* 主題底色: 深邃藍黑底色 */
                border-right: 1px solid #1E293B;   /* 右側邊框: 1 像素深藍灰分界線 */
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(16, 20, 16, 16)
        sidebar_layout.setSpacing(8)

        # [STEP-1] 系統主標題配置
        # 1.標籤欄位 (Lable)
        main_title_lbl = QLabel("MARS SYSTEM")
        main_title_lbl.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* 字體大小: 18 像素 */
                font-weight: 900;      /* 字體粗細: 極粗黑體 */
                color: #38BDF8;        /* 文字顏色: 天空藍高亮 */
                letter-spacing: 1px;   /* 字元間距: 1 像素寬度 */
            }
        """)
        sidebar_layout.addWidget(main_title_lbl)

        # [STEP-2] 系統副標題配置
        # 1.標籤欄位 (Lable)
        sub_title_lbl = QLabel("目標自動辨識與監控")
        sub_title_lbl.setStyleSheet("""
            QLabel {
                font-size: 11px;       /* 字體大小: 11 像素 */
                font-weight: normal;   /* 字體粗細: 常規字型 */
                color: #64748B;        /* 文字顏色: 暗板岩灰 */
                margin-bottom: 12px;   /* 底部邊距: 向下留白 12 像素 */
            }
        """)
        sidebar_layout.addWidget(sub_title_lbl)

        # [STEP-3] 水平分割線條配置
        # 1.矩形框架 (Frame)
        h_split_line = QFrame()
        h_split_line.setFrameShape(QFrame.Shape.HLine)
        h_split_line.setStyleSheet("""
            QFrame {
                background-color: #1E293B;   /* 線條顏色: 深藍石墨灰 */
                height: 1px;                 /* 線條高度: 1 像素細線 */
                border: none;                /* 邊框樣式: 移除預設立體邊框 */
                margin-bottom: 10px;         /* 底部邊距: 向下留白 10 像素 */
            }
        """)
        sidebar_layout.addWidget(h_split_line)

        # [STEP-4] 導覽功能欄配置
        # 1.系統總覽按鈕 (Button)
        self.btn_nav_dashboard = self._create_nav_button("🗂️ 系統總覽", 0)
        sidebar_layout.addWidget(self.btn_nav_dashboard)
        # 2.訓練模式按鈕 (Button)
        self.btn_nav_train = self._create_nav_button("⚙️ 訓練模式", 1)
        sidebar_layout.addWidget(self.btn_nav_train)
        # 3.辨識模式按鈕 (Button)
        self.btn_nav_infer = self._create_nav_button("🔎 辨識模式", 2)
        sidebar_layout.addWidget(self.btn_nav_infer)
        # 4.審計模式按鈕 (Button)
        self.btn_nav_database = self._create_nav_button("📊 審計模式", 3)
        sidebar_layout.addWidget(self.btn_nav_database)

        # 彈性間距 (版本資訊置於底端)
        sidebar_layout.addStretch()

        # [STEP-5] 版本資訊配置
        # 1.標籤欄位 (Lable)
        ver_lbl = QLabel("v1.0.0-PROD (PyQt6)")
        ver_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver_lbl.setStyleSheet("""
            QLabel {
                font-size: 11px;       /* 字體大小: 11 像素 */
                font-weight: normal;   /* 字體粗細: 常規字型 */
                color: #475569;        /* 文字顏色: 暗鐵灰 */
            }
        """)
        sidebar_layout.addWidget(ver_lbl)

        return sidebar_container

    # =============================================
    def _create_nav_button(self, text, page_index):
        """
        [名稱] Func.D 導覽按鈕構建函式
        [功能] 產出具備單選互斥屬性之導覽按鈕，套用主題樣式並與目標分頁索引綁定。
        [參數] 共計 2 組參數，以下說明:
               - text       : [str] 按鈕顯示之字串
               - page_index : [int] 對應之 QStackedWidget 分頁索引編號
        [輸出] QPushButton  : 完成組態配置與事件對接之導覽按鈕實例
        """
        # [STEP-1] 導覽按鈕
        # 1. 按鈕 (Button)
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(40)
        btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;   /* 預設底色: 完全透明 */
                color: #94A3B8;                  /* 文字顏色: 暗石墨灰 */
                font-size: 13px;                 /* 字體大小: 13 像素 */
                font-weight: bold;               /* 字體粗細: 粗體字型 */
                text-align: left;                /* 文字對齊: 靠左對齊 */
                padding-left: 14px;              /* 左側內距: 內縮 14 像素以對齊圖示 */
                border: none;                    /* 邊框樣式: 移除預設邊框 */
                border-radius: 6px;              /* 邊框圓角: 6 像素圓角邊緣 */
            }
            QPushButton:hover {
                background-color: #1E293B;       /* 懸停狀態: 滑鼠移入時變為深藍石墨灰 */
                color: #F8FAFC;                  /* 懸停文字: 轉換為高對比白亮色 */
            }
            QPushButton:checked {
                background-color: #2563EB;       /* 選中狀態: 當前啟動分頁呈現科技寶藍色 */
                color: #FFFFFF;                  /* 選中文字: 純白高亮 */
            }
        """)

        # [STEP-2] 註冊至導覽按鈕群組
        self.nav_button_group.addButton(btn, page_index)

        return btn

    # =============================================
    def _setup_view_pages(self):
        """
        [名稱] Func.F 分頁視圖建構函式
        [功能] 實例化各業務子視圖 (DashboardView、TrainView 與開發中佔位元件)，並依序掛載至 QStackedWidget 分頁堆疊。
        [參數] None: 無需外部傳入參數
        [輸出] None: 完成分頁堆疊設置
        """

        # [PAGE-1] 系統總覽視圖
        self.dashboard_view = DashboardView(self)
        self.content_stack.addWidget(self.dashboard_view)

        # [PAGE-2] 訓練模式視圖
        self.train_view = TrainView(self)
        self.content_stack.addWidget(self.train_view)

        # [PAGE-3] 辨識模式視圖
        self.infer_view = InferView(self)
        self.content_stack.addWidget(self.infer_view)

        # [PAGE-3] 審計模式視圖
        # self.database_view = DatabaseView(self)
        # self.content_stack.addWidget(self.database_view)

        # [PAGE-X] 開發中視圖
        # self.dashboard_palceholder = self._create_placeholder_view("系統總覽 (系統開發中)")
        # self.train_palceholder     = self._create_placeholder_view("訓練模式 (系統開發中)")
        # self.infer_palceholder     = self._create_placeholder_view("辨識模式 (系統開發中)")
        self.database_placeholder  = self._create_placeholder_view("審計模式 (系統開發中)")
        # self.content_stack.addWidget(self.dashboard_palceholder)
        # self.content_stack.addWidget(self.train_palceholder)
        # self.content_stack.addWidget(self.infer_palceholder)
        self.content_stack.addWidget(self.database_placeholder)

    # =============================================
    def _create_placeholder_view(self, title_text):
        """
        [名稱] Func.G 
        [功能] 構建開發中功能分頁之過渡佔位元件，提供置中之狀態提示文字與一致性暗色背景。
        [參數] title_text : [str] 佔位提示字串
        [輸出] QWidget    : 完成樣式設定之佔位容器元件
        """
        dev_widget = QWidget()
        dev_widget.setStyleSheet("""
            QWidget {
                background-color: #09090B;   /* 背景底色: 極致黑 */
            }
        """)
        dev_layout = QVBoxLayout(dev_widget)
        dev_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # [STEP-1] 開發提示配置
        # 1.標籤欄位 (Lable)
        dev_lbl = QLabel(title_text)
        dev_lbl.setStyleSheet("""
            QLabel {
                font-size: 18px;     /* 字體大小: 18 像素 */
                font-weight: bold;   /* 字體粗細: 粗體字型 */
                color: #52525B;      /* 文字顏色: 石墨灰提示文字 */
            }
        """)
        dev_layout.addWidget(dev_lbl)

        return dev_widget

    # =============================================
    def _connect_bridge_signals(self):
        """
        [名稱] Func.G 橋接信號對接事件函式
        [功能] 將訓練視圖與辨識視圖之操作事件，無縫串接至背景管線工作執行緒 (PipelineWorker)。
        [參數] None: 無需外部參數
        [輸出] None: 完成各視圖操作事件之對接
        """
        # [STEP-1] 訓練視圖 (TrainView) 訊號連接
        self.train_view.sig_start_train_requested.connect(self._handle_start_train)
        self.train_view.sig_stop_train_requested.connect(self._handle_stop_train)
        self.train_view.sig_upload_videos_requested.connect(self._handle_upload_train)

        # [STEP-2] 辨識視圖 (InferView) 訊號連接
        self.infer_view.sig_start_infer_requested.connect(self._handle_start_infer)
        self.infer_view.sig_stop_infer_requested.connect(self._handle_stop_infer)
        self.infer_view.sig_upload_videos_requested.connect(self._handle_upload_infer)

    # =============================================
    def _handle_start_train(self, payload):
        """ 接收訓練啟動請求 """
        self._start_worker("train", payload, self.train_view)
    def _handle_stop_train(self):
        """ 接收訓練中斷請求 """
        self._terminate_work(self.train_view)
    def _handle_upload_train(self, paths):
        """ 接收訓練影片上傳請求 """
        self._start_worker("upload_train", paths, self.train_view)

    # =============================================
    def _handle_start_infer(self, payload):
        """ 接收辨識啟動請求 """
        self._start_worker("infer", payload, self.infer_view)
    def _handle_stop_infer(self):
        """ 接收辨識中斷請求 """
        self._treminate_work(self.infer_view)
    def _handle_upload_infer(self, paths):
        """ 接收辨識影片上傳請求 """
        self._start_worker("upload_infer", paths, self.infer_view)

    # =============================================
    def _start_worker(self, task_type, payload, source_view):
        """
        [名稱] Func.H 背景工作排程啟動函式
        [功能] 建立並啟動 PipelineWorker 執行緒，鎖定來源視圖狀態，避免重複發起任務。
        [參數] 共計 3 組參數，以下說明:
               - task_type   : [str] 任務類型 ('train', 'infer', 'upload_train', 'upload_infer')
               - payload     : [dict/list] 任務參數封包或檔案清單
               - source_view : [QWidget] 觸發此任務之前台視圖實例
        [輸出] None: 啟動後台工作執行緒
        """
        if self.active_worker and self.active_worker.isRunning():
            QMessageBox.warning(self, "任務執行中", "後端已有任務正在運作，請等待完成或手動中斷")
            # 防止重複觸發機制，通知來源視圖發出警告標示
            if hasattr(source_view, "on_pipeline_finished"):
                source_view.on_pipeline_finished(False, "目前尚有任務運作中")
            return

        # 鎖定發起任務之來源視圖
        self.current_source_view = source_view

        # 實例化後端管線橋接工具
        self.active_worker = PipelineWorker(
            task_type = task_type,
            payload   = payload,
            parent    = self
        )
        # 後端任務完成信號回傳機制
        self.active_worker.sig_finished.connect(self._on_worker_finished)
        # 啟動非同步後端執行緒
        self.active_worker.start()

    # =============================================
    def _on_worker_finished(self, is_success, summary_msg):
        """
        [名稱] Func.I 後端工作成果回傳函式
        [功能] 接收 PipelineWorker 完工訊號，通知來源視圖更新狀態並解鎖按鈕，最後彈窗提示執行成果。
        [參數] 共計 2 組參數，以下說明:
               - is_success   : [bool] 任務是否成功
               - summary_msg  : [str] 後端管線回傳之摘要字串
        [輸出] None: 釋放執行緒並恢復前端初始狀態
        """
        # 通知來源視圖「恢復原始按鈕」與「更新狀態標籤」
        if self.current_source_view and hasattr(self.current_source_view, "on_pipeline_finished"):
            self.current_source_view.on_pipeline_finished(is_success, summary_msg)
        
        # 彈窗提示使用者執行成果
        if is_success:
            QMessageBox.information(self, "任務圓滿完成", summary_msg)
        else:
            QMessageBox.critical(self, "任務中止執行", summary_msg)

        # 清除當前工作狀態
        self.active_worker = None
        self.current_source_view = None

    # =============================================
    def _terminate_worker(self, source_view):
        """
        [名稱] Func.J 強制中止工作函式
        [功能] 手動終止當前運作中的背景管線工作，通知來源視圖更新為中斷狀態，並恢復介面控制權。
        [參數] source_view : [QWidget] 發起中斷請求之前端視圖實例
        [輸出] None: 終止後端執行緒
        """
        if self.active_worker and self.active_worker.isRunning():
            self.active_worker.terminate()
            self.active_worker.wait()
            self.active_worker = None

            # 通知來源視圖恢復按鈕並更新標籤文字為中斷紅字
            if hasattr(source_view, "on_pipeline_finished"):
                source_view.on_pipeline_finished(False, "任務強制中止")

        self.current_source_view = None
        QMessageBox.warning(self, "任務強制中止", "後端執行程序已被強制中止")

    # =============================================
    def closeEvent(self, event):
        """
        [名稱] Func.K 視窗關閉攔截函式
        [功能] 攔截主視窗關閉動作，確保背景執行緒安全回收後再行退出，避免殭屍執行緒或非正常閃退。
        [參數] event : [QCloseEvent] Qt 視窗關閉事件
        [輸出] None: 退出應用程式
        """
        if self.active_worker and self.active_worker.isRunning():
            self.active_worker.terminate()
            self.active_worker.wait()
        event.accept()

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. MainWindow : [類別] 主視窗核心框架 >>> 負責整合全域側邊欄、多分頁切換與後端管線工作調度。

__all__ = ["MainWindow"]

# =============================================
# 🧩｜單元測試｜單一介面獨立測試
# =============================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

        



        