# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 最佳權重靜態編譯工具 (Best Weight Static Compilation Tool)
# 維護日期: 2026-10-16
# 檔案路徑: MARS_Project/src/modl_opts/trt_export.py
# ##########################################################################################

# 掛載外部依賴
import traceback
import torch
from pathlib import Path
from ultralytics import YOLO

# 掛載內部依賴
from utils import log
from utils.env import input_resolver

# ==========================================================================================
class TensorRTExporter:
    """
    [名稱] 最佳權重靜態編譯引擎 (Best Weight Static Compilation Engine)
    [作用] 對接模型訓練產出成果，專職統籌單/雙階層架構下所有最佳權重之靜態編譯管線：
           1. 承接 auto_train 產出之最佳權重清單 (wght_paths)
           2. 驅動 TensorRT 加速編譯核心，執行 FP16 半精度靜態量化以極大化推論吞吐量
           3. 導出靜態二進位結構 (.engine)，回傳標準化產出路徑清單
    """
    # =============================================
    def __init__(self, sys_cfg, device_idx=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「最佳權重靜態編譯引擎」，自動探測或鎖定目標 GPU 硬體定址。
        [參數] 共計 2 組參數，以下說明:
               - sys_cfg   : [dict] 全域組態檢索，內含 stage_inst 階層策略組態
               - device_idx: [int/None] 指定執行加速編譯之 GPU 識別碼；若未指定則自動探測當前可用 CUDA 裝置
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg   = sys_cfg
        self.stage_cfg = self.sys_cfg["stage_inst"]

        if device_idx is not None:
            self.gpu_idx = int(device_idx)
        else:
            if torch.cuda.is_available():
                self.gpu_idx = torch.cuda.current_device()
            else:
                self.gpu_idx = 0
                
    # =============================================
    def _resolve_stage_tag(self, domain, src_wght_path):
        """
        [名稱] Func.B 階層任務標籤解析函式
        [功能] 依據階層策略配置對齊路徑結構，動態產出標準階層標籤 (e.g., TRT-S1-VEH, TRT-S2-LIC)。
        [參數] 共計 2 組參數，以下說明:
               - domain        : [str] 目標領域識別代碼 (e.g., "VEH")
               - src_wght_path : [Path] 原生最佳權重實體路徑
        [輸出] str: 標準化任務識別名稱 (TRT-S1-{DOMAIN} 或 TRT-S2-{SUB_DOMAIN})
        """
        stage_rule = self.stage_cfg.get(domain, {})
        sub_task   = stage_rule.get("stage_2", {}).get("sub_tasks", {})
        path_parts = set(Path(src_wght_path).parts)

        # 階層策略配置中定義的次領域
        for sub_domain in sub_task.keys():
            if sub_domain in path_parts:
                return f"TRT-S2-{sub_domain}"
        # 預設為該領域主階層 S1
        return f"TRT-S1-{domain}"

    # =============================================
    def _compile_single_engine(self, domain, ver_id, src_wght_path, half_precision=True):
        """
        [名稱] Func.C 單一權重靜態編譯函式
        [功能] 驗證原生 best.pt 實體檔案，並透過 YOLO export 核心導出單一 TensorRT (.engine) 檔案。
        [參數] 共計 4 組參數，以下說明:
               - domain         : [str] 目標領域識別代碼 (e.g., "VEH")
               - ver_id         : [str] 模型版本識別代碼 (e.g., "v1.0.0")
               - src_wght_path  : [Path] 原生 YOLO 最佳權重實體路徑 (best.pt)
               - half_precision : [bool] 是否啟用 FP16 量化；預設為 True
        [輸出] tuple: (engine_path: Path/None, err_ret: dict/None)
               - engine_path : [Path/None] 成功時回傳 TensorRT 引擎實體路徑，失敗為 None
               - err_ret     : [dict/None] 失敗時回傳結構化錯誤字典 (包含 404 缺檔或 500 異常)，成功為 None
        """
        src_wght_path = Path(src_wght_path)
        src_wght_name = src_wght_path.name

        # 內部自動解析階層標籤 (TRT-S1-VEH, TRT-S2-LIC...)
        stage_name = self._resolve_stage_tag(domain, src_wght_path)

        """ [STAGE-1] 最佳權重整備階段 """
        if not src_wght_path.exists():
            # 驗證最佳權重 best.pt 實體狀態，攔截檔案遺失異常
            rel_path = input_resolver.to_root_relative(src_wght_path)
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "VALIDATE_BEST_WEIGHT",
                stat = "WARN",
                msge = f"[檔案: {src_wght_name}] 找不到指定的最佳權重，請檢查檔案路徑 ({rel_path})"
            )
            log.FOOTER(acnt="TRT-HALT", rslt=f"[{domain}] 原生最佳權重不存在，任務安全中止")

            err_ret = {
                "stat_code" : 404,
                "stat_msge" : "MISSING_WEIGHT",
                "rslt_msge" : f"[{domain}] 原生最佳權重不存在，請檢查檔案路徑 ({rel_path})",
                "details"   : {
                    "domain"       : domain,
                    "ver_id"       : ver_id,
                    "dst_engine_paths" : []
                }
            }
            return None, err_ret
        
        """ [STAGE-2] 驅動 TensorRT 模型靜態編譯 """
        log.CONTENT(
            type = "MODEL",
            targ = stage_name,
            idnt = "EXECUTE_TRT_EXPORT",
            stat = "INFO",
            msge = f"[任務: 靜態編譯轉譯] 啟動編譯作業，來源權重 ({src_wght_name}), 硬體指派 (GPU-{self.gpu_idx}), 半精度 (FP16={half_precision})"
        )

        model = YOLO(str(src_wght_path))
        exported_path_str = model.export(
            format  = "engine",         # 導出為 TensorRT 靜態加速引擎 (.engine)
            half    = half_precision,   # FP16 半精度量化
            device  = self.gpu_idx,     # 指定執行 CUDA 編譯之硬體編號
            dynamic = False             # 鎖定模型輸入尺寸獲取極致吞吐效能
        )
        dst_engine_path = Path(exported_path_str)
        
        log.CONTENT(
            type = "MODEL",
            targ = stage_name,
            idnt = "GENERATE_ENGINE",
            stat = "SUCC",
            msge = f"[檔案: {dst_engine_path.name}] 已完成 TensorRT 靜態加速引擎導出"
        )
        return dst_engine_path, None
    
    # =============================================
    def __call__(self, domain, ver_id, src_wght_paths, half_precision=True):
        """
        [名稱] Func.D 最佳權重靜態編譯函式
        [功能] 讀取原生最佳權重，驅動加速轉譯核心，執行 FP16 半精度最佳化優化，並導出高吞吐最佳引擎。
        [參數] 共計 4 組核心參數，以下說明:
               - domain         : [str] 目標領域識別代碼 (e.g., "VEH")
               - ver_id         : [str] 目標領域版本代碼 (e.g., "v1.0.0")
               - src_wght_paths : [list] 待編譯之原生 YOLO 最佳權重實體路徑清單 (來自 auto_train 回傳之 details.wght_paths)
               - half_precision : [bool] 是否啟用 FP16 量化開關；預設為啟用(True)
        [輸出] dict: 內含執行結果之結構化字典，共計 4 組鍵值，以下說明:
               - stat_code : [int] 執行狀態代碼 (200-成功, 400-參數缺失, 404-檔案遺失, 500-系統異常)
               - stat_msge : [str] 狀態識別標籤 (COMPLETED, EMPTY_WEIGHT_LIST, MISSING_WEIGHT, SYSTEM_ERROR)             
               - rslt_msge : [str] 執行結果之詳細文字說明
               - details   : [dict] 共計 3 組任務指標明細，以下說明:
                 - domain           : [str] 目標領域識別代碼 (e.g., "VEH")
                 - ver_id           : [str] 模型版本識別代碼 (e.g., "v1.0.0")
                 - dst_engine_paths : [list] 產出之靜態權重檔案相對路徑清單 (.engine)
        """
        domain = str(domain).strip().upper()
        stage_name = f"TRT-{domain}"
        log.BANNER(acnt="TRT-TASK", msin=f"最佳權重靜態編譯任務 (載入 {domain} 權重集 - 版本 {ver_id})")
        try:
            """ [STAGE-1] 權重清單驗證階段 """
            if not src_wght_paths:
                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "VALIDATE_WEIGHT_FILES",
                    stat = "WARN",
                    msge = f"[領域: {domain}] 未接收到任何待編譯之權重清單"
                )
                log.FOOTER(acnt="TRT-HALT", rslt=f"[{domain}] 待編譯權重清單為空，任務安全中止")

                trt_mngr_info = {
                    "stat_code" : 400,
                    "stat_msge" : "EMPTY_WEIGHT_LIST",
                    "rslt_msge" : f"[{domain}] 待編譯之權重清單為空，無法執行靜態編譯",
                    "details"   : {
                        "domain"           : domain,
                        "ver_id"           : ver_id,
                        "dst_engine_paths" : []
                    }
                }
                return trt_mngr_info

            """ [STAGE-2] 權重靜態編譯階段 """
            dst_engine_paths = []
            for src_wght_path in src_wght_paths:
                sgl_engine_path, err_ret = self._compile_single_engine(
                    domain         = domain,
                    ver_id         = ver_id,
                    src_wght_path  = src_wght_path,
                    half_precision = half_precision
                )
                if err_ret:
                    return err_ret

                rel_path = input_resolver.to_root_relative(sgl_engine_path)
                dst_engine_paths.append(str(rel_path))

            """ [STAGE-3] 編譯成果彙整階段 """
            log.FOOTER(acnt="TRT-DONE", rslt=f"[{domain}] 全階層最佳權重靜態編譯完成 (共計生成 {len(dst_engine_paths)} 組引擎)")
            trt_mngr_info = {
                "stat_code" : 200,
                "stat_msge" : "COMPLETED",
                "rslt_msge" : f"[{domain}] 最佳權重靜態編譯作業圓滿完成",
                "details"   : {
                    "domain"           : domain,
                    "ver_id"           : ver_id,
                    "dst_engine_paths" : dst_engine_paths
                }
            }
            return trt_mngr_info
        
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "TRT_EXPORT_EXCEPTION",
                stat = "FAIL",
                msge = f"[任務: 靜態編譯任務] 遭遇非預期例外中斷，堆疊資訊如下: \n{sys_err}"
            )
            log.FOOTER(acnt="TRT-FAIL", rslt=f"[{domain}] 靜態編譯非預期中斷，請調閱堆疊排查")

            trt_mngr_info = {
                "stat_code" : 500,
                "stat_msge" : "SYSTEM_ERROR",
                "rslt_msge" : f"[{domain}] 編譯過程遭遇非預期系統異常",
                "details"   : {
                    "domain"           : domain,
                    "ver_id"           : ver_id,
                    "dst_engine_paths" : []
                }
            }
            return trt_mngr_info