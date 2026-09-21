# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 領域管線配發工具 (Domain Pipeline Dispatcher)
# 維護日期: 2026-09-19
# 檔案路徑: MARS_Project/src/vis_recg/dispatcher.py
# ##########################################################################################

# 掛載內部依賴
from utils import log
from .pipelines import PIPELINE_TOOLS

# ==========================================================================================
class PipelineDispatcher:
    """
    [名稱] 領域管線配發引擎 (Pipeline Dispatcher Engine)
    [作用] 依據系統指定之領域識別代碼 (domain)，自 PIPELINE_TOOLS 自動檢索並實例化對應管線。
    """
    # =============================================
    def __init__(self, sys_cfg):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「領域管線配發引擎」，掛載系統組態及 pipelines 動態聚合之管線模組總表。
        [參數] sys_cfg : [dict] 全域組態檢索
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg   = sys_cfg
        self._registry = dict(PIPELINE_TOOLS)

    # =============================================
    def __call__(self, domain, **kwargs):
        """
        [名稱] Func.B 領域管線配發函式
        [功能] 驗證傳入之 domain 有效性，並從註冊表 (PIPELINE_TOOLS) 中檢索對應類別並完成實例化；支援透過 **kwargs 將各管線所需之專屬參數直接傳遞。
        [參數] 共計 2 組參數，以下說明:
               - domain : [str] 目標領域識別代碼 (e.g., "VEH")
               - kwargs : [dict] 目標領域專屬管線之擴充注入實例 (e.g., ocr_inst 等)
        [輸出] 領域專屬管線實例；若未註冊或為空則回傳 None (Fail Fast)
        """
        if not domain:
            log.CONTENT(
                type = "PIPE",
                targ = "DISP-ROUTER",
                idnt = "MISSING_DOMAIN",
                stat = "FAIL",
                msge = "[參數: domain] 驗證失敗 >>> 未明確指定 domain 領域代碼，拒絕分派管線 (Fail Fast)"
            )
            return None

        domain = str(domain).strip().upper()
        if domain not in self._registry:
            log.CONTENT(
                type = "PIPE",
                targ = "DISP-ROUTER",
                idnt = "UNREGISTERED_DOMAIN",
                stat = "FAIL",
                msge = f"[映射表: PIPELINE_TOOLS] 配置失敗 >>> 領域代碼 ({domain}) 未在註冊表中定義，拒絕處理 (Fail Fast)"
            )
            return None

        pipeline_cls = self._registry[domain]
        return pipeline_cls(sys_cfg=self.sys_cfg, domain=domain, **kwargs)