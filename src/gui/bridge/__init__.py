# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 視圖橋接工具套件入口 (View Bridge Toolkit Entry)
# 維護日期: 2026-10-14
# 檔案路徑: MARS_Project/src/gui/bridge/__init__.py
# ##########################################################################################

# 掛載內部依賴
from .log_handler import log_emitter, attach_gui_log_handler, detach_gui_log_handler
from .pipeline_worker import PipelineWorker

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. log_emitter            : [實例] 系統日誌信號發射引擎 >>> 提供各視圖與元件綁定日誌事件監聽通道。
# 2. attach_gui_log_handler : [函式] 系統日誌轉發掛載函式 >>> 將系統日誌紀錄核心無縫接入 Qt 事件迴圈。
# 3. detach_gui_log_handler : [函式] 系統日誌轉發卸載函式 >>> 於視窗關閉時解除監聽，避免物件銷毀引發異常。
# 4. PipelineWorker         : [類別] 後端管線橋接引擎 >>> 負責解耦 GUI 介面與後端耗時管線，避免主視窗凍結。

__all__ = [
    "log_emitter",
    "attach_gui_log_handler",
    "detach_gui_log_handler",
    "PipelineWorker"
]