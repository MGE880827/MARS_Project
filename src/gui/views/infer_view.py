# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 模型辨識視圖工具 (Model Inference View Tool)
# 維護日期: 2026-10-13
# 檔案路徑: MARS_Project/src/gui/views/infer_view.py
# ##########################################################################################

# 掛載外部依賴
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox,
    QFileDialog, QSizePolicy
)

# 掛載內部依賴
from ..widgets import TerminalConsoleWidget

# ==========================================================================================
class InferView(QWidget):
    """
    [名稱] 模型辨識視圖 (Model Inference View)
    [作用] 提供辨識目標模型指定、待辨識影片上傳匯入、加速引擎組態配置、即時畫面監控與日誌終端輸出之配置介面。
    """
    # 宣告對外業務觸發信號: 提供主視窗或控制器對接背景工作執行緒 (Worker)
    sig_start_infer_requested   = pyqtSignal(dict)   # 傳送參數訊號
    sig_stop_infer_requested    = pyqtSignal()       # 中斷請求訊號
    sig_upload_videos_requested = pyqtSignal(list)   # 影片上傳訊號

    # =============================================
    def __init__(self, parent=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「模型辨識視圖」，構建左右分流之控制與影像監控版面，並配置專屬之終端主控元件。
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

        # [STEP-3] 構建右側影像監控與日誌面板
        self._init_right_monitor_panel()

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
        task_group = QGroupBox("任務群組")
        task_group.setStyleSheet("""
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
        self.stage_cmb.addItem("即時辨識 (infer)", "infer")
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
        self.ver_id_text.setPlaceholderText("e.g., 空白自動探測 or VEH:v1.0.0 MARA:v1.0.0")
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
        self.data_group = QGroupBox("資料群組")
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

        # [STEP-1] 辨識影片上傳 (Button)
        self.btn_upload_videos = QPushButton("辨識影片上傳 (vids_todo)")
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

        # [STEP-1] TensorRT 加速參數 (CheckBox, Label)
        self.no_trt_chk = QCheckBox("停用 TensorRT 加速 (--no_trt)")
        self.no_trt_chk.setCursor(Qt.CursorShape.PointingHandCursor)
        self.no_trt_chk.setStyleSheet("""
            QCheckBox {
                color: #D4D4D8;              /* 文字顏色: 淡灰白 */
                font-size: 12px;             /* 字體大小: 12 像素 */
                spacing: 8px;                /* 方塊與文字間距: 8 像素 */
            }
            QCheckBox::indicator {
                width: 16px;                 /* 方塊寬度: 16 像素 */
                height: 16px;                /* 方塊高度: 16 像素 */
                border: 1px solid #3F3F46;   /* 邊框樣式: 暗灰細線 */
                border-radius: 3px;          /* 方塊圓角: 3 像素 */
                background-color: #09090B;   /* 方塊底色: 極致黑 */
            }
            QCheckBox::indicator:checked {
                background-color: #EF4444;   /* 勾選底色: 珊瑚紅 (代表降級) */
                border: 1px solid #DC2626;   /* 勾選邊框: 深紅色 */
            }
        """)
        param_layout.addWidget(self.no_trt_chk)
        no_trt_lbl = QLabel("預設優先使用 TensorRT (.engine) 進行 FP16 效能辨識")
        no_trt_lbl.setStyleSheet("""
            QLabel {
                font-size: 11px;       /* 字體大小: 11 像素 */
                font-weight: normal;   /* 字體粗細: 常規字型 */
                color: #64748B;        /* 文字顏色: 暗鐵灰 */
                padding-left: 2px;     /* 左側內距: 2 像素 */
            }
        """)
        param_layout.addWidget(no_trt_lbl)

        # [STEP-2] 群組配置併入控制面板
        left_layout.addWidget(self.param_group)

        """ [STAGE-4] 操作控制配置 """
        # [STEP-1] 啟動辨識管線 (Button)
        self.btn_start_infer = QPushButton("▶ 啟動辨識")
        self.btn_start_infer.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_infer.setFixedHeight(40)
        self.btn_start_infer.setStyleSheet("""
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
        self.btn_start_infer.clicked.connect(self._on_start_pipeline_clicked)
        left_layout.addWidget(self.btn_start_infer)

        # [STEP-2] 終止辨識管線 (Button)
        self.btn_stop_infer = QPushButton("■ 終止辨識")
        self.btn_stop_infer.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop_infer.setFixedHeight(36)
        self.btn_stop_infer.setEnabled(False)
        self.btn_stop_infer.setStyleSheet("""
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
        self.btn_stop_infer.clicked.connect(self._on_stop_pipeline_clicked)
        left_layout.addWidget(self.btn_stop_infer)

        # [STEP-3] 彈性間距 (撐滿上方)
        left_layout.addStretch()

        # [STEP-4] 群組配置併入控制面板
        self.main_layout.addWidget(left_container)

    # =============================================
    def _init_right_monitor_panel(self):
        """
        [名稱] Func.C 右側監控面板配置
        [功能] 構建即時視訊預覽畫布、辨識運作數據指標列，以及專屬之虛擬終端主控元件。
        [參數] None: 無需外部參數
        [輸出] None: 完成右側面板佈局
        """
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        """ [STAGE-1] 目標監控配置 """
        obj_moni_group  = QGroupBox("目標監控")
        obj_moni_group.setStyleSheet("""
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
        obj_moni_layout = QVBoxLayout(obj_moni_group)
        obj_moni_layout.setContentsMargins(8, 8, 8, 8)

        # [STEP-1] 影像渲染畫布 (Lable)
        self.vid_screen = QLabel("[STATUS] 啟動辨識管線後，此處呈現目標檢測標註框")
        self.vid_screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vid_screen.setMinimumSize(640, 360)
        self.vid_screen.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.vid_screen.setStyleSheet("""
            QLabel {
                background-color: #09090B;    /* 螢幕底色: 極致黑 */
                color: #52525B;               /* 文字顏色: 石墨灰待命提示 */
                font-size: 13px;              /* 字體大小: 13 像素 */
                font-weight: bold;            /* 字體粗細: 粗體字型 */
                border: 1px dashed #27272A;   /* 虛線外框: 科技虛擬框架 */
                border-radius: 4px;           /* 邊框圓角: 4 像素 */
            }
        """)
        obj_moni_layout.addWidget(self.vid_screen)

        # [STEP-2] 辨識過程指標 (即時幀率, 目標數量, 延遲耗時, 運行狀態)
        telemetry_layout = QHBoxLayout()
        telemetry_layout.setSpacing(16)
        # 1.即時幀率 (Lable)
        self.fps_lbl = QLabel("即時幀率(FPS): 0.0")
        self.fps_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;     /* 字體大小: 12 像素 */
                font-weight: bold;   /* 字體粗細: 粗體字型 */
                color: #10B981;      /* 文字顏色: 翡翠綠高亮 */
            }
        """)
        telemetry_layout.addWidget(self.fps_lbl)
        # 2.目標數量 (Label)
        self.obj_cnt_lbl = QLabel("目標數量: 0")
        self.obj_cnt_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;     /* 字體大小: 12 像素 */
                font-weight: bold;   /* 字體粗細: 粗體字型 */
                color: #38BDF8;      /* 文字顏色: 天空藍高亮 */
            }
        """)
        telemetry_layout.addWidget(self.obj_cnt_lbl)
        # 3.延遲耗時 (Label)
        self.speed_lbl = QLabel("延遲耗時: 0.0 ms")
        self.speed_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;     /* 字體大小: 12 像素 */
                font-weight: bold;   /* 字體粗細: 粗體字型 */
                color: #F59E0B;      /* 文字顏色: 琥珀黃高亮 */
            }
        """)
        telemetry_layout.addWidget(self.speed_lbl)
        telemetry_layout.addStretch()
        # 4.運行狀態 (Label)
        self.op_status_lbl = QLabel("[STATUS] 辨識管線待命中")
        self.op_status_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;     /* 字體大小: 12 像素 */
                font-weight: bold;   /* 字體粗細: 粗體字型 */
                color: #10B981;      /* 文字顏色: 翡翠綠 (正常) */
            }
        """)
        telemetry_layout.addWidget(self.op_status_lbl)

        obj_moni_layout.addLayout(telemetry_layout)

        # [STEP-3] 群組配置併入控制面板
        right_layout.addWidget(obj_moni_group)

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
    def _on_upload_videos_clicked(self):
        """
        [名稱] Func.D 辨識影片選取與上傳函式
        [功能] 彈出 Windows 檔案選取對話框，支援批次選取待辨識影片檔 (*.mp4, *.avi, *.mov, *.mkv)，打包路徑清單並發射 sig_upload_videos_requested 信號。
        [參數] None: 無需外部參數
        [輸出] None: 發射影片路徑清單信號
        """
        vid_filters = "影片檔案 (*.mp4 *.avi *.mov *.mkv);;所有檔案 (*.*)"
        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "欲上傳之辨識影片檔案",
            "",
            vid_filters
        )
        if selected_files:
            self.sig_upload_videos_requested.emit(selected_files)

    # =============================================
    def _on_start_pipeline_clicked(self):
        """
        [名稱] Func.E 啟動辨識管線函式
        [功能] 切換 UI 鎖定狀態，將當前參數精準對齊 CLI 指令格式打包，發送 sig_start_infer_requested 信號。
        [參數] None: 無需外部參數
        [輸出] None: 發射 sig_start_infer_requested 信號 (封包: 指令參數)
        """
        self.btn_upload_videos.setEnabled(False)
        self.btn_start_infer.setEnabled(False)
        self.btn_stop_infer.setEnabled(True)
        self.op_status_lbl.setText("[STATUS] 辨識指令已送出，等待後台工作執行啟動...")
        self.op_status_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;     /* 字體大小: 12 像素 */
                font-weight: bold;   /* 字體粗細: 粗體字型 */
                color: #F59E0B;      /* 文字顏色: 琥珀黃 (運作中) */
            }
        """)

        # 指令封包傳至後端 __call__(exec_stage, ver_id_dict, no_trt)
        exec_payload = {
            "mode"   : "infer",
            "stage"  : self.stage_cmb.currentData(),
            "ver_id" : self.ver_id_text.text().strip() or None,
            "no_trt" : self.no_trt_chk.isChecked()
        }
        self.sig_start_infer_requested.emit(exec_payload)

    # =============================================
    def _on_stop_pipeline_clicked(self):
        """
        [名稱] Func.F 終止辨識管線函式
        [功能] 切換 UI 狀態，發送中斷請求信號通知後台 Worker 安全退出。
        [參數] None: 無需外部參數
        [輸出] None: 發射 sig_stop_infer_requested 信號
        """
        self.btn_upload_videos.setEnabled(False)
        self.btn_start_infer.setEnabled(False)
        self.btn_stop_infer.setEnabled(False)
        self.op_status_lbl.setText("[STATUS] 中斷指令已送出，等待管線安全退出...")
        self.op_status_lbl.setStyleSheet("""
            QLabel {
                font-size: 12px;     /* 字體大小: 12 像素 */
                font-weight: bold;   /* 字體粗細: 粗體字型 */
                color: #EF4444;      /* 文字顏色: 珊瑚紅 (終止) */
            }
        """)
        self.sig_stop_infer_requested.emit()

    # =============================================
    def on_pipeline_finished(self, success, summary_msg=""):
        """
        [名稱] Func.G 管線結束初始函式 (供外部使用)
        [功能] 接收 Worker 執行完成或中斷/失敗信號，自動復原操作按鈕與狀態標籤。
        [參數] 共計 2 組參數，以下說明:
               - success     : [bool] 任務是否成功執行完畢
               - summary_msg : [str] 狀態摘要字串 (選填)
        [輸出] None          : 重置按鈕狀態與文字
        """
        self.btn_upload_videos.setEnabled(True)
        self.btn_start_infer.setEnabled(True)
        self.btn_stop_infer.setEnabled(False)

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
    def update_frame(self, pixmap, fps=0.0, obj_cnt=0, speed=0.0):
        """
        [名稱] Func.H 視訊畫面與數據更新函式 (供外部使用)
        [功能] 接收背景解碼並繪製完成之 QPixmap 幀與即時辨識數據，刷新畫面與統計指標。
        [參數] 共計 4 組參數，以下說明:
               - pixmap  : [QPixmap] 繪製完畢之影像畫布
               - fps     : [float] 即時幀率
               - obj_cnt : [int] 當前幀偵測目標總數
               - speed   : [float] 單幀辨識耗時毫秒
        [輸出] None: 刷新畫面與指標數據
        """
        if pixmap:
            self.vid_screen.setPixmap(pixmap)

        self.fps_lbl.setText(f"即時幀率(FPS): {fps:.1f}")
        self.obj_cnt_lbl.setText(f"目標數量: {obj_cnt}")
        self.speed_lbl.setText(f"延遲耗時: {speed:.1f} ms")

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. InferView : [類別] 模型辨識視圖 >>> 負責提供多目標辨識管線控制、影片上傳匯入、視訊監控螢幕與日誌監控。

__all__ = ["InferView"]

# =============================================
# 🧩｜單元測試｜單一介面獨立測試
# =============================================
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = InferView()
    window.resize(1100, 720)
    window.setWindowTitle("MARS 多重目標自動辨識系統 - 模型辨識視圖測試")
    window.show()
    sys.exit(app.exec())