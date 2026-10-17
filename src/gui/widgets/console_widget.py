# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 虛擬終端主控元件工具 (Terminal Console Widget Tool)
# 維護日期: 2026-10-14
# 檔案路徑: MARS_Project/src/gui/widgets/console_widget.py
# ##########################################################################################

# 掛載外部依賴
from PyQt6.QtGui import QFont, QTextCursor, QColor
from PyQt6.QtWidgets import QTextEdit

# 掛載內部依賴
from ..bridge.log_handler import log_emitter

# ==========================================================================================
class TerminalConsoleWidget(QTextEdit):
    """
    [名稱] 虛擬終端主控元件 (Terminal Console Widget)
    [作用] 封裝文字渲染視圖，對接全域日誌發射引擎，負責將後台管線日誌以終端機風格配色並自動捲動置底。
    """
    # =============================================
    def __init__(self, parent=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「虛擬終端主控元件」，配置深色終端樣式表，並自動掛載日誌信號監聽。
        [參數] parent : [QWidget/None] Qt 父容器物件 (選填)
        [輸出] None   : 完成元件外觀與事件組態設置
        """
        super().__init__(parent)

        # [STEP-1] 設置文字控制台核心屬性
        self.setReadOnly(True)
        font = QFont()
        # 英文優先採用等寬字體，並加入微軟正黑體 UI 解決 Windows 中文字體問題
        font.setFamilies([
            "Cascadia Mono", 
            "Consolas", 
            "Microsoft JhengHei UI", 
            "Microsoft JhengHei", 
            "monospace"
        ])
        font.setPointSize(10)
        # 啟用平滑邊緣渲染，消除鋸齒
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)

        # [STEP-2] 終端機深色風格樣式表配置
        self.setStyleSheet("""
            QTextEdit {
                font-family: "Cascadia Mono", "Consolas", "Microsoft JhengHei UI", "Microsoft JhengHei", monospace;
                                             /* 字型家族: 中英雙軌等寬排版 */
                background-color: #18181B;   /* 主題底色: 深黑灰底色 */
                color: #D4D4D8;              /* 文字顏色: 預設淡灰白 */
                font-size: 13px;             /* 字型大小: 13 像素高解析文字 */
                border: 1px solid #27272A;   /* 元件邊框: 1 像素暗灰色細邊框 */
                border-radius: 6px;          /* 邊框圓角: 6 像素圓角邊緣 */
                padding: 8px;                /* 內距留白: 全向內縮 8 像素 */
                line-height: 1.3;            /* 行高微調: 設定 1.3 倍行距，避免中英文上下擠壓 */
            }
        """)

        # [STEP-3] 自動綁定全域日誌發射信號
        self.bind_log_emitter(log_emitter)

    # =============================================
    def write_log(self, level, text):
        """
        [名稱] Func.B 日誌格式渲染函式
        [功能] 接收日誌字串，依據層級標籤設定色彩並顯示於主控台。
        [參數] 共計 2 組參數，以下說明:
               - level : [str] 系統日誌層級標籤 (e.g., "INFO", "WARN", "FAIL", "SUCC")
               - text  : [str] 待顯示之純文字字串
        [輸出] None: 即時刷新畫面輸出
        """
        if not text:
            return

        # 狀態層級配色映射
        color = "#D4D4D8"   # 預設淺灰

        clean_line = text.strip()
        # 1.主管線橫幅 (Banner / Footer) 與外框雙等號線
        if ">>> [主管線啟動]" in clean_line or "<<< [主管線結束]" in clean_line or clean_line.startswith("==="):
            color = "#38BDF8"   # 天空藍 (主管線邊界)
            
        # 2.次任務橫幅 (Banner 啟動 / Footer 結束)
        elif "● [次任務啟動]" in clean_line:
            color = "#F59E0B"   # 琥珀黃 (次任務啟動)

        elif "○ [次任務結束]" in clean_line:
            color = "#10B981"   # 翡翠綠 (次任務結束)

        # 3.內文數據列 (檢查行首的四種狀態標籤)
        elif clean_line.startswith("SUCC"):
            color = "#10B981"   # 翡翠綠 (成功)

        elif clean_line.startswith("WARN"):
            color = "#F59E0B"   # 琥珀黃 (警示)

        elif clean_line.startswith("FAIL") or level == "ERROR":
            color = "#EF4444"   # 珊瑚紅 (失敗/例外)

        # 4.次任務分隔線
        elif clean_line.startswith("---"):
            color = "#71717A"   # 霧面灰 (分隔線)

        # 5. 其餘一般訊息 (INFO)
        else:
            color = "#D4D4D8"   # 淺灰 (預設文字)

        # 將色彩應用至當前游標並插入文字
        self.setTextColor(QColor(color))
        self.append(str(text))

        # 自動將游標移至文字末端以實現自動置底捲動
        self.moveCursor(QTextCursor.MoveOperation.End)

    # =============================================
    def clear_console(self):
        """
        [名稱] Func.C 主控台緩衝重置函式
        [功能] 清空當前虛擬終端主控台上已累積的所有日誌文字緩衝區。
        [參數] None: 無需外部傳入參數
        [輸出] None: 完成文字清空
        """
        self.clear()

    # =============================================
    def bind_log_emitter(self, emitter):
        """
        [名稱] Func.D 日誌信號觸發連接函式
        [功能] 將指定的日誌事件發射引擎之信號連結至 write_log 函式。
        [參數] emitter : [QtLogEmitter] 負責發射 sig_log_emitted 之信號實例
        [輸出] None: 完成信號槽對接
        """
        if hasattr(emitter, "sig_log_emitted"):
            emitter.sig_log_emitted.connect(self.write_log)

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. TerminalConsoleWidget : [類別] 虛擬終端主控元件 >>> 負責呈現系統日誌流並自動配色捲動。

__all__ = ["TerminalConsoleWidget"]