# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 後端管線橋接工具 (Backend Pipeline Bridge Tool)
# 維護日期: 2026-10-13
# 檔案路徑: MARS_Project/src/gui/bridge/pipeline_worker.py
# ##########################################################################################

# 掛載外部依賴
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

# 掛載內部依賴
from config import path_config
from utils.env import file_manager, input_resolver
from train_pipe import TrainPipeline
from infer_pipe import InferPipeline

# ==========================================================================================
class PipelineWorker(QThread):
    """
    [名稱] 後端管線橋接引擎 (Backend Pipeline Bridge Engine)
    [作用] 擔任 GUI 前端視圖與後端演算法管線之非同步橋接通道，於獨立工作執行緒調度訓練、辨識與影片匯入任務，避免主介面凍結並回傳執行成果。
    """
    # 宣告向 GUI 視圖反向通知之狀態訊號
    sig_finished = pyqtSignal(bool, str)   # (是否成功, 摘要說明)

    # =============================================
    def __init__(self, task_type, payload=None, parent=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「後端管線橋接引擎」，鎖定任務類型並裝載參數封包。
        [參數] 共計 3 組參數，以下說明:
               - task_type : [str] 任務類型 ('train', 'infer', 'upload_train', 'upload_infer')
               - payload   : [dict/list/None] 任務所需之封包資訊 (指令參數或檔案路徑清單)
               - parent    : [QObject/None] Qt 父容器物件
        [輸出] None: 完成執行緒組態配置
        """
        super().__init__(parent)
        self.task_type = task_type
        self.payload   = payload

    # =============================================
    def run(self):
        """
        [名稱] Func.B 核心工作執行主體
        [功能] 於獨立後端執行緒內執行目標管線，捕獲例外並發射 sig_finished 信號。
        [參數] None: 無需外部參數
        [輸出] None: 透過信號通知任務成果
        """
        try:
            # [TASK-1] 模型訓練任務調度
            if self.task_type == "train":
                self._run_train_pipeline()

            # [TASK-2] 模型辨識任務調度
            elif self.task_type == "infer":
                self._run_infer_pipeline()

            # [TASK-3] 訓練影片上傳
            elif self.task_type == "upload_train":
                self._run_upload_train()

            # [TASK-4] 辨識影片上傳
            elif self.task_type == "upload_infer":
                self._run_upload_infer()

            # 未知類型任務
            else:
                self.sig_finished.emit(False, f"未知任務類型: {self.task_type}")

        except Exception as e:
            err_msg = f"後端管線執行崩潰: {str(e)}"
            self.sig_finished.emit(False, err_msg)

    # =============================================
    def _run_train_pipeline(self):
        """
        [名稱] Func.C 執行模型訓練管線
        [功能] 解析前端傳入之參數並呼叫 TrainPipeline 調度器，執行模型訓練任務並發射成果信號。
        [參數] None: 自實例屬性 self.payload 提取參數
        [輸出] None: 透過 sig_finished 信號發送執行結果與摘要字串
        """
        payload = self.payload if isinstance(self.payload, dict) else {}

        # [STEP-1] 提取封包內任務參數
        stage      = str(payload.get("stage") or "all").strip()
        ver_id_str = str(payload.get("ver_id") or "").strip()
        epochs_str = str(payload.get("epochs") or "").strip()
        lr_str     = str(payload.get("lr") or "").strip()

        # [STEP-2] 重組清單並解析 CLI 參數字典 (e.g., "VEH:v1.0.0" -> {"VEH": "v1.0.0"})
        ver_id_dict     = input_resolver.parse_kv_pair(ver_id_str.split(), val_type=str)
        epoch_qty_dict  = input_resolver.parse_kv_pair(epochs_str.split(), val_type=int)
        learn_rate_dict = input_resolver.parse_kv_pair(lr_str.split(), val_type=float)

        # [STEP-3] 啟動自動化模型訓練管線
        pipeline = TrainPipeline()
        report = pipeline(
            exec_stage      = stage,
            ver_id_dict     = ver_id_dict,
            epoch_qty_dict  = epoch_qty_dict,
            learn_rate_dict = learn_rate_dict
        )

        stat_code   = report.get("stat_code", 500)
        is_success  = stat_code in [200, 206]
        summary_msg = report.get("rslt_msge", "訓練管線執行完畢")
        self.sig_finished.emit(is_success, summary_msg)

    # =============================================
    def _run_infer_pipeline(self):
        """
        [名稱] Func.D 執行模型辨識管線
        [功能] 解析前端傳入之參數並呼叫 InferPipeline 調度器，執行影像辨識任務並發射成果信號。
        [參數] None: 自實例屬性 self.payload 提取參數
        [輸出] None: 透過 sig_finished 信號發送執行結果與摘要字串
        """
        payload = self.payload if isinstance(self.payload, dict) else {}

        # [STEP-1] 提取封包內任務參數
        stage      = str(payload.get("stage") or "infer").strip()
        ver_id_str = str(payload.get("ver_id") or "").strip()
        no_trt     = bool(payload.get("no_trt", False))

        # [STEP-2] 重組清單並解析 CLI 參數字典 (e.g., "VEH:v1.0.0" -> {"VEH": "v1.0.0"})
        ver_id_dict = input_resolver.parse_kv_pair(ver_id_str.split(), val_type=str) if ver_id_str else None
        use_trt = not no_trt

        # [STEP-3] 啟動自動化影像辨識管線
        pipeline = InferPipeline()
        report = pipeline(
            exec_stage  = stage,
            ver_id_dict = ver_id_dict,
            use_trt     = use_trt 
        )

        stat_code   = report.get("stat_code", 500)
        is_success  = stat_code in [200, 206]
        summary_msg = report.get("rslt_msge", "辨識管線執行完畢")
        self.sig_finished.emit(is_success, summary_msg)

    # =============================================
    def _run_upload_train(self):
        """
        [名稱] Func.E 訓練影片上傳與任務佇列路由
        [功能] 調用檔案管理工具將原始訓練影片批量匯入待辦目錄或任務佇列，並回傳匯入成果信號。
        [參數] None: 自實例屬性 self.payload 提取選取之檔案路徑清單
        [輸出] None: 透過 sig_finished 信號發送執行結果與摘要字串
        """
        # [STEP-1] 提取封包內影片路徑清單
        vids_path = self.payload if isinstance(self.payload, list) else []

        # [STEP-2] 啟動訓練影片上傳流程
        succ_cnt  = file_manager.upload_train_videos(
            src_vid_paths   = vids_path,
            active_vids_dir = path_config.train.raw_vids,
            task_queue_dir  = path_config.train.task_queue
        )

        is_success  = succ_cnt > 0
        summary_msg = f"成功匯入 {succ_cnt} 部訓練影片" if is_success else "影片上傳失敗或格式不符"
        self.sig_finished.emit(is_success, summary_msg)

    # =============================================
    def _run_upload_infer(self):
        """
        [名稱] Func.F 辨識影片上傳
        [功能] 調用檔案管理工具將待辨識影片批量匯入辨識待辦目錄，並回傳匯入成果信號。
        [參數] None: 自實例屬性 self.payload 提取選取之檔案路徑清單
        [輸出] None: 透過 sig_finished 信號發送執行結果與摘要字串
        """
        # [STEP-1] 提取封包內影片路徑清單
        vids_path = self.payload if isinstance(self.payload, list) else []

        # [STEP-2] 啟動辨識影片上傳流程
        succ_cnt  = file_manager.upload_infer_videos(
            src_vid_paths   = vids_path,
            target_vids_dir = path_config.infer.vids_todo,
        )

        is_success  = succ_cnt > 0
        summary_msg = f"成功匯入 {succ_cnt} 部辨識影片" if is_success else "影片上傳失敗或格式不符"
        self.sig_finished.emit(is_success, summary_msg)

# =============================================
# ⭐｜模組導出｜對外公開接口匯出
# =============================================
# 1. PipelineWorker : [類別] 後端管線橋接引擎 >>> 負責解耦 GUI 介面與後端耗時管線，避免主視窗凍結。

__all__ = ["PipelineWorker"]