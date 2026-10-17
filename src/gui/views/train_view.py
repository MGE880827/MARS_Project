# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 模型訓練視圖工具 (Model Training View Tool)
# 維護日期: 2026-10-12
# 檔案路徑: MARS_Project/src/gui/views/train_view.py
# ##########################################################################################

# 掛載外部依賴
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QComboBox, QProgressBar, QFileDialog
)

# 掛載內部依賴
from ..widgets import TerminalConsoleWidget

# ==========================================================================================
class TrainView(QWidget):
    """
    [名稱] 模型訓練視圖 (Model Training View)
    [作用] 提供 YOLO 模型參數設定、原始訓練影片上傳匯入、介面狀態切換、任務進度監控與日誌終端輸出之配置介面。
    """
    # 宣告對外業務觸發信號: 提供主視窗或控制器對接背景工作執行緒 (Worker)
    sig_start_train_requested   = pyqtSignal(dict)   # 傳送參數訊號
    sig_stop_train_requested    = pyqtSignal()       # 中斷請求訊號
    sig_upload_videos_requested = pyqtSignal(list)   # 影片上傳訊號
    
    # =============================================
    def __init__(self, parent=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「模型訓練視圖」，構建左右分流之控制與監控版面，並配置專屬之終端主控元件。
        [參數] parent : [QWidget/None] Qt 父容器物件 (選填)
        [輸出] None   : 完成視圖與事件組態配置
        """
        super().__init__(parent)

        # [STEP-1] 建立主要水平佈局
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(16)

        # [STEP-2] 構建左側控制面板
        self._init_left_control_panel()

        # [STEP-3] 構建右側任務進度與日誌面板
        self._init_right_monitor_panel()

        # [STEP-4] 觸發初始模式聯動渲染 (預設: all 全流程模式)
        self._on_stage_changed()

    # =============================================
    def _init_left_control_panel(self):
        """
        [名稱] Func.B 左側控制面板配置函式
        [功能] 構建任務群組、資料群組、參數群組以及操作控制工具列等純參數輸入控制項。
        [參數] None: 無需外部參數
        [輸出] None: 完成左側面板佈局
        """
        left_container = QWidget()
        left_container.setFixedWidth(400)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        """ [STAGE-1] 任務群組配置 """
        task_group  = QGroupBox("任務群組")
        task_group.setStyleSheet("""
            QGroupBox {
                background-color: #18181B;   /* 容器底色: 石墨深灰 */
                border: 1px solid #27272A;   /* 邊框樣式: 1 像素暗灰線條 */
                border-radius: 6px;          /* 邊框圓角: 6 像素 */
                margin-top: 12px;            /* 頂部外距: 預留標題空間 */
                font-size: 13px;             /* 字體大小: 13 像素 */
                font-weight: bold;           /* 字體粗細: 粗體字型 */
                color: #38BDF8;              /* 標題顏色: 天空藍 */
                padding: 12px;               /* 內部留白: 全向 12 像素 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;   /* 標題定位: 依外距定位 */
                subcontrol-position: top left; /* 標題位置: 靠左上方 */
                left: 10px;                  /* 偏移位置: 左移 10 像素 */
                padding: 0 4px;              /* 標題內距: 左右留白 4 像素 */
            }
        """)
        task_layout = QVBoxLayout(task_group)
        task_layout.setSpacing(8)
        
        # [STEP-1] 執行階段參數 (Lable, ComboBox)
        stage_layout = QHBoxLayout()
        stage_layout.setSpacing(8)
        stage_lbl = QLabel("執行階段 (--stage):")
        stage_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;       /* 字體大小: 12 像素 */
                font-weight: normal;   /* 字體粗細: 常規字型 */
                color: #D4D4D8;        /* 文字顏色: 淡灰白 */
            }
        """)
        stage_layout.addWidget(stage_lbl)
        self.stage_cmb = QComboBox()
        self.stage_cmb.addItem("完整流程 (all)", "all")
        self.stage_cmb.addItem("資料整備 (prepare)", "prepare")
        self.stage_cmb.addItem("模型優化 (train)", "train")
        self.stage_cmb.setStyleSheet("""
            QComboBox {
                background-color: #09090B;   /* 下拉框底色: 極致黑 */
                color: #FFFFFF;              /* 文字顏色: 純白 */
                border: 1px solid #27272A;   /* 邊框樣式: 1 像素暗灰線 */
                border-radius: 4px;          /* 邊框圓角: 4 像素 */
                padding: 6px 10px;           /* 內距留白: 上下 6 像素、左右 10 像素 */
                font-size: 12px;             /* 字體大小: 12 像素 */
            }
            QComboBox::drop-down {
                border: none;                /* 移除箭頭區塊預設邊框 */
            }
        """)
        self.stage_cmb.currentIndexChanged.connect(self._on_stage_changed)
        stage_layout.addWidget(self.stage_cmb)
        task_layout.addLayout(stage_layout)

        # [STEP-2] 模型版本參數 (Lable, LineEdit)
        ver_id_layout = QHBoxLayout()
        ver_id_layout.setSpacing(8)
        ver_id_lbl = QLabel("模型版本 (--ver_id):")
        ver_id_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;       /* 字體大小: 12 像素 */
                font-weight: normal;   /* 字體粗細: 常規字型 */
                color: #D4D4D8;        /* 文字顏色: 淡灰白 */
            }
        """)
        ver_id_layout.addWidget(ver_id_lbl)
        self.ver_id_text = QLineEdit("VEH:v1.0.0")
        self.ver_id_text.setPlaceholderText("e.g., VEH:v1.0.0 MARA:v1.0.0")
        self.ver_id_text.setStyleSheet("""
            QLineEdit {
                background-color: #09090B;   /* 輸入框底色: 極致黑 */
                color: #38BDF8;              /* 文字顏色: 天空藍 */
                border: 1px solid #27272A;   /* 邊框樣式: 1 像素暗灰線 */
                border-radius: 4px;          /* 邊框圓角: 4 像素 */
                padding: 4px 8px;            /* 內距留白: 上下 4 像素、左右 8 像素 */
                font-size: 12px;             /* 字體大小: 12 像素 */
                font-weight: bold;           /* 字體粗細: 粗體字型 */
            }
        """)
        ver_id_layout.addWidget(self.ver_id_text)
        task_layout.addLayout(ver_id_layout)

        # [STEP-3] 群組配置併入控制面板
        left_layout.addWidget(task_group)

        """ [STAGE-2] 資料群組配置 """
        self.data_group  = QGroupBox("資料群組")
        self.data_group.setStyleSheet("""
            QGroupBox {
                background-color: #18181B;       /* 容器底色: 石墨深灰 */
                border: 1px solid #27272A;       /* 邊框樣式: 1 像素暗灰線條 */
                border-radius: 6px;              /* 邊框圓角: 6 像素 */
                margin-top: 12px;                /* 頂部外距: 預留標題空間 */
                font-size: 13px;                 /* 字體大小: 13 像素 */
                font-weight: bold;               /* 字體粗細: 粗體字型 */
                color: #38BDF8;                  /* 標題顏色: 天空藍 */
                padding: 12px;                   /* 內部留白: 全向 12 像素 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;       /* 標題定位: 依外距定位 */
                subcontrol-position: top left;   /* 標題位置: 靠左上方 */
                left: 10px;                      /* 偏移位置: 左移 10 像素 */
                padding: 0 4px;                  /* 標題內距: 左右留白 4 像素 */
            }
        """)
        data_layout = QVBoxLayout(self.data_group)
        data_layout.setSpacing(10)

        # [STEP-1] 原始影片上傳 (Button)
        self.btn_upload_videos = QPushButton("原始影片上傳 (raw_vids)")
        self.btn_upload_videos.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_upload_videos.setFixedHeight(36)
        self.btn_upload_videos.setStyleSheet("""
            QPushButton {
                background-color: #047857;   /* 預設底色: 翡翠深綠 */
                color: #FFFFFF;              /* 文字顏色: 純白 */
                border: none;                /* 邊框樣式: 移除邊框 */
                border-radius: 4px;          /* 邊框圓角: 4 像素 */
                padding: 4px 12px;           /* 內距留白: 上下 4 像素、左右 12 像素 */
                font-weight: bold;           /* 字體粗細: 粗體字型 */
                font-size: 12px;             /* 字體大小: 12 像素 */
            }
            QPushButton:hover {
                background-color: #059669;   /* 懸停狀態: 明亮翡翠綠 */
            }
            QPushButton:pressed {
                background-color: #065F46;   /* 點擊狀態: 深墨綠 */
            }
        """)
        self.btn_upload_videos.clicked.connect(self._on_upload_videos_clicked)
        data_layout.addWidget(self.btn_upload_videos)

        # [STEP-2] 群組配置併入控制面板
        left_layout.addWidget(self.data_group)

        """ [STAGE-3] 參數群組配置 """
        self.param_group = QGroupBox("參數群組")
        self.param_group.setStyleSheet("""
            QGroupBox {
                background-color: #18181B;       /* 容器底色: 石墨深灰 */
                border: 1px solid #27272A;       /* 邊框樣式: 1 像素暗灰線條 */
                border-radius: 6px;              /* 邊框圓角: 6 像素 */
                margin-top: 12px;                /* 頂部外距: 預留標題空間 */
                font-size: 13px;                 /* 字體大小: 13 像素 */
                font-weight: bold;               /* 字體粗細: 粗體字型 */
                color: #38BDF8;                  /* 標題顏色: 天空藍 */
                padding: 12px;                   /* 內部留白: 全向 12 像素 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;       /* 標題定位: 依外距定位 */
                subcontrol-position: top left;   /* 標題位置: 靠左上方 */
                left: 10px;                      /* 偏移位置: 左移 10 像素 */
                padding: 0 4px;                  /* 標題內距: 左右留白 4 像素 */
            }
        """)
        param_layout = QVBoxLayout(self.param_group)
        param_layout.setSpacing(8)

        # [STEP-1] 訓練輪次參數 (Label, LineEdit)
        epochs_layout = QHBoxLayout()
        epochs_layout.setSpacing(8)
        epochs_lbl = QLabel("訓練輪次 (--epochs):")
        epochs_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;       /* 字體大小: 12 像素 */
                font-weight: normal;   /* 字體粗細: 常規字型 */
                color: #D4D4D8;        /* 文字顏色: 淡灰白 */
            }
        """)
        epochs_layout.addWidget(epochs_lbl)
        self.epochs_text = QLineEdit("VEH:100")
        self.epochs_text.setPlaceholderText("e.g., VEH:100 MARA:100")
        self.epochs_text.setStyleSheet("""
            QLineEdit {
                background-color: #09090B;   /* 輸入框底色: 極致黑 */
                color: #F8FAFC;              /* 文字顏色: 白亮色 */
                border: 1px solid #27272A;   /* 邊框樣式: 1 像素暗灰線 */
                border-radius: 4px;          /* 邊框圓角: 4 像素 */
                padding: 4px 8px;            /* 內距留白: 上下 4 像素、左右 8 像素 */
                font-size: 12px;             /* 字體大小: 12 像素 */
            }
        """)
        epochs_layout.addWidget(self.epochs_text)
        param_layout.addLayout(epochs_layout)

        # [STEP-2] 學習速率參數 (Label, LineEdit)
        lr_layout = QHBoxLayout()
        lr_layout.setSpacing(8)
        lr_lbl = QLabel("學習速率 (--lr):")
        lr_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;       /* 字體大小: 12 像素 */
                font-weight: normal;   /* 字體粗細: 常規字型 */
                color: #D4D4D8;        /* 文字顏色: 淡灰白 */
            }
        """)
        lr_layout.addWidget(lr_lbl)
        self.lr_text = QLineEdit("VEH:0.01")
        self.lr_text.setPlaceholderText("e.g., VEH:0.01 MARA:0.005")
        self.lr_text.setStyleSheet("""
            QLineEdit {
                background-color: #09090B;   /* 輸入框底色: 極致黑 */
                color: #F8FAFC;              /* 文字顏色: 白亮色 */
                border: 1px solid #27272A;   /* 邊框樣式: 1 像素暗灰線 */
                border-radius: 4px;          /* 邊框圓角: 4 像素 */
                padding: 4px 8px;            /* 內距留白: 上下 4 像素、左右 8 像素 */
                font-size: 12px;             /* 字體大小: 12 像素 */
            }
        """)
        lr_layout.addWidget(self.lr_text)
        param_layout.addLayout(lr_layout)

        # [STEP-3] 群組配置併入控制面板
        left_layout.addWidget(self.param_group)

        """ [STAGE-4] 操作控制配置 """
        # [STEP-1] 啟動訓練管線 (Button)
        self.btn_start_train = QPushButton("▶ 啟動訓練")
        self.btn_start_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_train.setFixedHeight(40)
        self.btn_start_train.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;   /* 預設底色: 科技寶藍色 */
                color: #FFFFFF;              /* 文字顏色: 純白 */
                border: none;                /* 邊框樣式: 移除邊框 */
                border-radius: 4px;          /* 邊框圓角: 4 像素圓角邊緣 */
                font-size: 13px;             /* 字體大小: 13 像素 */
                font-weight: bold;           /* 字體粗細: 粗體字型 */
            }
            QPushButton:hover {
                background-color: #1D4ED8;   /* 懸停狀態: 深藍色 */
            }
            QPushButton:pressed {
                background-color: #1E40AF;   /* 點擊狀態: 深海藍色 */
            }
            QPushButton:disabled {
                background-color: #1E293B;   /* 禁用狀態: 暗灰底色 */
                color: #64748B;              /* 禁用文字: 暗鐵灰 */
            }
        """)
        self.btn_start_train.clicked.connect(self._on_start_pipeline_clicked)
        left_layout.addWidget(self.btn_start_train)

        # [STEP-2] 終止訓練管線 (Button)
        self.btn_stop_train = QPushButton("■ 終止訓練")
        self.btn_stop_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop_train.setFixedHeight(36)
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.setStyleSheet("""
            QPushButton {
                background-color: #3F3F46;   /* 預設底色: 石墨灰 */
                color: #FFFFFF;              /* 文字顏色: 純白 */
                border: none;                /* 邊框樣式: 移除邊框 */
                border-radius: 4px;          /* 邊框圓角: 4 像素圓角邊緣 */
                font-size: 12px;             /* 字體大小: 12 像素 */
                font-weight: bold;           /* 字體粗細: 粗體字型 */
            }
            QPushButton:hover {
                background-color: #EF4444;   /* 懸停狀態: 警示珊瑚紅 */
            }
            QPushButton:pressed {
                background-color: #B91C1C;   /* 點擊狀態: 深紅 */
            }
            QPushButton:disabled {
                background-color: #18181B;   /* 禁用狀態: 深沉黑 */
                color: #52525B;              /* 禁用文字: 灰暗色 */
            }
        """)
        self.btn_stop_train.clicked.connect(self._on_stop_pipeline_clicked)
        left_layout.addWidget(self.btn_stop_train)

        # [STEP-3] 彈性間距 (撐滿上方)
        left_layout.addStretch()

        # [STEP-4] 群組配置併入控制面板
        self.main_layout.addWidget(left_container)

    # =============================================
    def _init_right_monitor_panel(self):
        """
        [名稱] Func.C 右側監控面板配置
        [功能] 構建任務執行進度條、狀態指示文字標籤，以及專屬之虛擬終端主控元件。
        [參數] None: 無需外部參數
        [輸出] None: 完成右側面板佈局
        """
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        """ [STAGE-1] 任務進度配置 """
        progress_group  = QGroupBox("任務進度")
        progress_group.setStyleSheet("""
            QGroupBox {
                background-color: #18181B;   /* 容器底色: 石墨深灰 */
                border: 1px solid #27272A;   /* 邊框樣式: 1 像素暗灰線條 */
                border-radius: 6px;          /* 邊框圓角: 6 像素 */
                margin-top: 12px;            /* 頂部外距: 預留標題空間 */
                font-size: 13px;             /* 字體大小: 13 像素 */
                font-weight: bold;           /* 字體粗細: 粗體字型 */
                color: #38BDF8;              /* 標題顏色: 天空藍 */
                padding: 12px;               /* 內部留白: 全向 12 像素 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;   /* 標題定位: 依外距定位 */
                subcontrol-position: top left; /* 標題位置: 靠左上方 */
                left: 10px;                  /* 偏移位置: 左移 10 像素 */
                padding: 0 4px;              /* 標題內距: 左右留白 4 像素 */
            }
        """)
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(8)

        # [STEP-1] 訓練過程指標 (運行狀態, 執行進度)
        # 1.運行狀態 (Label)
        self.op_status_lbl = QLabel("[STATUS] 訓練管線待命中")
        self.op_status_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;     /* 字體大小: 12 像素 */
                font-weight: bold;   /* 字體粗細: 粗體字型 */
                color: #10B981;      /* 文字顏色: 翡翠綠 (正常) */
            }
        """)
        progress_layout.addWidget(self.op_status_lbl)
        # 2.執行進度 (ProgressBar)
        self.execute_bar = QProgressBar()
        self.execute_bar.setRange(0, 100)
        self.execute_bar.setValue(0)
        self.execute_bar.setTextVisible(True)
        self.execute_bar.setStyleSheet("""
            QProgressBar {
                background-color: #09090B;   /* 進度條槽底色: 極致黑 */
                color: #FFFFFF;              /* 百分比文字顏色: 白色 */
                border: 1px solid #27272A;   /* 邊框樣式: 1 像素暗灰線 */
                border-radius: 4px;          /* 邊框圓角: 4 像素 */
                text-align: center;          /* 數值文字置中 */
                height: 18px;                /* 高度: 18 像素 */
                font-size: 11px;             /* 字體大小: 11 像素 */
                font-weight: bold;           /* 字體粗細: 粗體 */
            }
            QProgressBar::chunk {
                background-color: #2563EB;   /* 填充塊底色: 科技寶藍色 */
                border-radius: 3px;          /* 圓角處理 */
            }
        """)
        progress_layout.addWidget(self.execute_bar)

        # [STEP-2] 群組配置併入控制面板
        right_layout.addWidget(progress_group)

        """ [STAGE-2] 終端日誌配置 """
        console_group = QGroupBox("終端日誌")
        console_group.setStyleSheet("""
            QGroupBox {
                background-color: #18181B;       /* 容器底色: 石墨深灰 */
                border: 1px solid #27272A;       /* 邊框樣式: 1 像素暗灰線條 */
                border-radius: 6px;              /* 邊框圓角: 6 像素 */
                margin-top: 12px;                /* 頂部外距: 預留標題空間 */
                font-size: 13px;                 /* 字體大小: 13 像素 */
                font-weight: bold;               /* 字體粗細: 粗體字型 */
                color: #38BDF8;                  /* 標題顏色: 天空藍 */
                padding: 12px;                   /* 內部留白: 全向 12 像素 */
            }
            QGroupBox::title {
                subcontrol-origin: margin;       /* 標題定位: 依外距定位 */
                subcontrol-position: top left;   /* 標題位置: 靠左上方 */
                left: 10px;                      /* 偏移位置: 左移 10 像素 */
                padding: 0 4px;                  /* 標題內距: 左右留白 4 像素 */
            }
        """)
        console_layout = QVBoxLayout(console_group)
        console_layout.setContentsMargins(10, 14, 10, 10)

        # [STEP-1] 虛擬終端機 (Widget)
        self.console_widget = TerminalConsoleWidget(self)
        console_layout.addWidget(self.console_widget)

        # [STEP-2] 群組配置併入控制面板
        right_layout.addWidget(console_group, stretch=1)

        self.main_layout.addWidget(right_container, stretch=1)

    # =============================================
    def _on_stage_changed(self):
        """
        [名稱] Func.D 執行階段欄位鎖定函式
        [功能] 當使用者切換 stage (all / prepare / train) 時，關閉特定輸入控制項與群組。
        [參數] None: 無需外部參數
        [輸出] None: 完成表單動態啟閉
        """
        stage_mode = self.stage_cmb.currentData()

        # [MODE-1] 資料整備階段
        if stage_mode == "prepare":
            self.data_group.setEnabled(True)
            self.param_group.setEnabled(False)

        # [MODE-2] 模型優化階段
        elif stage_mode == "train":
            self.data_group.setEnabled(False)
            self.param_group.setEnabled(True)

        # [MODE-3] 完整流程階段
        else:
            self.data_group.setEnabled(True)
            self.param_group.setEnabled(True)

    # =============================================
    def _on_upload_videos_clicked(self):
        """
        [名稱] Func.E 訓練影片選取與上傳函式
        [功能] 彈出 Windows 檔案選取對話框，支援批次選取影片檔 (*.mp4, *.avi, *.mov, *.mkv)，打包路徑清單並發射 sig_upload_videos_requested 信號。
        [參數] None: 無需外部參數
        [輸出] None: 發射影片路徑清單信號
        """
        vid_filters = "影片檔案 (*.mp4 *.avi *.mov *.mkv);;所有檔案 (*.*)"
        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "欲上傳之訓練影片檔案",
            "",
            vid_filters
        )
        if selected_files:
            self.sig_upload_videos_requested.emit(selected_files)

    # =============================================
    def _on_start_pipeline_clicked(self):
        """
        [名稱] Func.F 啟動訓練管線函式
        [功能] 切換 UI 鎖定狀態，將當前參數精準對齊 CLI 指令格式打包，發送 sig_start_train_requested 信號。
        [參數] None: 無需外部參數
        [輸出] None: 發射 sig_start_train_requested 信號 (封包: 指令參數)
        """
        self.btn_upload_videos.setEnabled(False)
        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.op_status_lbl.setText("[STATUS] 訓練指令已送出，等待後台工作執行啟動...")
        self.op_status_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;     /* 字體大小: 12 像素 */
                font-weight: bold;   /* 字體粗細: 粗體字型 */
                color: #F59E0B;      /* 文字顏色: 琥珀黃 (運作中) */
            }
        """)

        # 指令封包傳至後端 __call__(exec_stage, ver_id_dict, epoch_qty_dict, learn_rate_dict)
        exec_payload = {
            "mode"   : "train",
            "stage"  : self.stage_cmb.currentData(),
            "ver_id" : self.ver_id_text.text().strip(),
            "epochs" : self.epochs_text.text().strip(),
            "lr"     : self.lr_text.text().strip()
        }
        self.sig_start_train_requested.emit(exec_payload)

    # =============================================
    def _on_stop_pipeline_clicked(self):
        """
        [名稱] Func.G 終止訓練管線函式
        [功能] 切換 UI 狀態，發送中斷請求信號通知後台 Worker 安全退出。
        [參數] None: 無需外部參數
        [輸出] None: 發射 sig_stop_train_requested 信號
        """
        self.btn_upload_videos.setEnabled(False)
        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(False)
        self.op_status_lbl.setText("[STATUS] 中斷指令已送出，等待管線安全退出...")
        self.op_status_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;     /* 字體大小: 12 像素 */
                font-weight: bold;   /* 字體粗細: 粗體字型 */
                color: #EF4444;      /* 文字顏色: 珊瑚紅 (終止) */
            }
        """)
        self.sig_stop_train_requested.emit()

    # =============================================
    def on_pipeline_finished(self, success, summary_msg=""):
        """
        [名稱] Func.H 管線結束初始函式 (供外部使用)
        [功能] 接收 Worker 執行完成或中斷/失敗信號，自動復原操作按鈕與狀態標籤。
        [參數] 共計 2 組參數，以下說明:
               - success     : [bool] 任務是否成功執行完畢
               - summary_msg : [str] 狀態摘要字串 (選填)
        [輸出] None: 重置按鈕狀態與文字
        """
        self.btn_upload_videos.setEnabled(True)
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)

        if success:
            rslt_msg = summary_msg if summary_msg else "任務圓滿完成"
            self.op_status_lbl.setText(f"[STATUS] {rslt_msg} (Success)")
            self.op_status_lbl.setStyleSheet("""
                QLabel {
                    font-size: 12px;     /* 字體大小: 12 像素 */
                    font-weight: bold;   /* 字體粗細: 粗體字型 */
                    color: #10B981;      /* 文字顏色: 翡翠綠 (成功) */
                }
            """)
        else:
            rslt_msg = summary_msg if summary_msg else "任務中止執行"
            self.op_status_lbl.setText(f"[STATUS] {rslt_msg} (Fail)")
            self.op_status_lbl.setStyleSheet("""
                QLabel {
                    font-size: 12px;     /* 字體大小: 12 像素 */
                    font-weight: bold;   /* 字體粗細: 粗體字型 */
                    color: #EF4444;      /* 文字顏色: 珊瑚紅 (異常/失敗) */
                }
            """)

    # =============================================
    def update_progress(self, current_val, total_val, status_msg=""):
        """
        [名稱] Func.I 任務進度更新函式 (供外部使用)
        [功能] 供外部 Worker 跨執行緒回傳數值刷新進度條。
        [參數] 共計 3 組參數，以下說明:
               - current_val : [int] 當前進度數值
               - total_val   : [int] 總數值上限
               - status_msg  : [str] 即時狀態描述字串 (選填)
        [輸出] None: 刷新介面進度條
        """
        if total_val > 0:
            percentage = int((current_val / total_val) * 100)
            self.execute_bar.setValue(percentage)
            if status_msg:
                self.op_status_lbl.setText(f"[STATUS] {status_msg} ({percentage}%)")

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. TrainView : [類別] 模型訓練視圖 >>> 負責提供多階段訓練管線控制、訓練影片上傳匯入、參數配置與進度監控。

__all__ = ["TrainView"]

# =============================================
# 🧩｜單元測試｜單一介面獨立測試
# =============================================
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = TrainView()
    window.resize(1100, 720)
    window.setWindowTitle("MARS 多重目標自動辨識系統 - 模型訓練視圖測試")
    window.show()
    sys.exit(app.exec())
