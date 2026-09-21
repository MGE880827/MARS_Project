# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 視覺辨識模型加載工具 (Vision Recognition Model Preload Tool)
# 維護日期: 2026-09-19
# 檔案路徑: MARS_Project/src/vis_recg/modl_load.py
# ##########################################################################################

# 掛載外部依賴
import gc
import traceback
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR

# 掛載內部依賴
from utils import log
from utils.env import file_manager, input_resolver

# ==========================================================================================
class VisionModelPreloader:
    """
    [名稱] 視覺辨識模型加載引擎 (Vision Recognition Model Preload Engine)
    [作用] 對接「全域階層策略-INFER」與「演算法路徑組態」，專職統籌辨識階段所需之模型預載管線:
           1. 依主領域自動查閱 INFER 階層規則並載入 Stage-1 主階層 YOLO 模型 (e.g., VEH)
           2. 依階層策略逐一載入 Stage-2 次階層微觀 YOLO 模型 (e.g., LIC, BRAND)
           3. 動態探測各階層所需重型影像工具 (img_tools)，自動預載對應之推論引擎 (e.g., char_ocr)
           4. 委由 file_manager 集中探測最新權重檔案路徑，統一封裝回傳並提供載入模型使用
    """
    # =============================================
    def __init__(self, sys_cfg):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「視覺辨識模型加載引擎」，並掛載演算法組態 (algo)、路徑組態 (path) 及階層策略字典 (stage)。
        [參數] sys_cfg: [dict] 全域組態檢索，內含 algo_inst, path_inst 與 stage_inst 組態
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg   = sys_cfg
        self.algo_cfg  = self.sys_cfg["algo_inst"]
        self.path_cfg  = self.sys_cfg["path_inst"]
        self.stage_cfg = self.sys_cfg["stage_inst"]

        # YOLO 辨識模型實例映射表 {"VEH": yolo_inst, "LIC": yolo_inst, "BRAND": yolo_inst }
        self.yolo_pool = {}     
        # IMG-PROC 重型影像處理工具實例映射表 {"char_ocr": ocr_inst}
        self.tool_pool = {}
        # 預載工具調度註冊表: 定義需實例化之重型推論引擎與其專屬加載函式之對照關係
        self.tool_loaders = {
            "char_ocr": self._load_ocr_model
        }

    # =============================================
    def get_yolo_model(self, domain):
        """
        [名稱] Func.A YOLO 模型實例索取函式
        [功能] 提供外部依領域代碼 (e.g., "VEH", "LIC", "BRAND") 自 yolo_pool 索取已預載之 YOLO 實例。
        [參數] domain: [str] 目標領域或次領域識別代碼 (e.g., "VEH", "LIC", "BRAND")
        [輸出] object: YOLO 預載模型實例，查無時回傳 None
        """
        if not domain:
            return None
        return self.yolo_pool.get(str(domain).strip().upper())
    
    # =============================================
    def get_img_tool_model(self, tool_name, lang="chinese_cht"):
        """
        [名稱] Func.B 工具實例索取函式
        [功能] 提供外部依「工具名稱」與「語系代碼」，自 tool_pool 索取已預載之實例。
        [參數] 共計 2 組參數，以下說明:
               - tool_name : [str] 工具識別代碼 (e.g., "char_ocr")
               - lang      : [str, optional] 語系代碼；預設為 "chinese_cht"
        [輸出] object: 預載工具實例，查無時回傳 None
        """
        if not tool_name:
            return None

        tool_key = f"{tool_name}@{lang}" if lang else tool_name
        return self.tool_pool.get(tool_key)

    # =============================================
    def clear_cache(self):
        """
        [名稱] Func.C 模型預載釋放函式
        [功能] 釋放當前已載入之「影像辨識模型」與「影像處理模型」，顯式調用垃圾回收以安全釋出 GPU 顯存與記憶體資源。
        [參數] None: 無需外部傳入參數
        [輸出] None: 完成「影像辨識模型」與「影像處理模型」實例重設
        """
        self.yolo_pool.clear()
        self.tool_pool.clear()
        gc.collect()
        log.CONTENT(
            type = "MODEL",
            targ = "PRELOADER",
            idnt = "CLEAR_CACHE_SUCCESS",
            stat = "SUCC",
            msge = "[任務: 模型預載任務] 清理成功 >>> 已釋放所有已載入之「影像辨識模型」與「影像處理模型」，重置記憶體資源"
        )

    # =============================================
    def _load_yolo_model(self, domain, sub_domain=None, ver_id=None, use_trt=True):
        """
        [名稱] Func.D-1 YOLO 模型掛載函式
        [功能] 透過 file_manager 探測權重路徑，載入指定主領域或次領域之 YOLO 模型並快取於 self.yolo_pool。
        [參數] 共計 4 組參數，以下說明:
               - domain     : [str] 目標領域識別代碼 (e.g., "VEH")
               - sub_domain : [str, optional] 次領域識別代碼 (e.g., "LIC", "BRAND")；預設 None
               - ver_id     : [str, optional] 模型版本代碼 (e.g., "v1.0.0")；若為 None 則自動載入最新版本
               - use_trt    : [bool] 是否優先使用 TensorRT 引擎加速；預設 True
        [輸出] dict: 內含執行結果之結構化字典，共計 5 組鍵值，以下說明:
               - stat_code  : [int] 執行狀態代碼 (200-成功, 404-缺檔, 500-系統異常)
               - stat_msge  : [str] 狀態識別標籤 (COMPLETED, MISSING_WEIGHT_FILE, LOAD_MODEL_EXCEPTION)
               - rslt_msge  : [str] 執行結果之詳細文字說明
               - yolo_model : [str] 成功加載之 YOLO 模型名稱，若失敗則為 None
               - ver_id     : [str] 實際載入之模型版本代碼；若失敗則為 None
        """
        main_domain = str(domain).strip().upper()
        sub_domain  = str(sub_domain).strip().upper() if sub_domain else None

        load_domain = sub_domain if sub_domain else main_domain
        stage_name  = f"YOLO-{main_domain}-{sub_domain}" if sub_domain else f"YOLO-{main_domain}"
        scope_name  = f"[次領域: {main_domain}/{sub_domain}]" if sub_domain else f"[主領域: {main_domain}]"

        """ [STAGE-1] 記憶體檢索與權重路徑解析階段 """
        wght_root_dir = Path(self.path_cfg.weights)

        # [STEP-1] 模型版本定位，若未指定 ver_id，則委由 file_manager 探測最新版本
        use_ver_id = str(ver_id).strip() if ver_id else file_manager.find_latest_weight(
            domain        = main_domain,
            wght_root_dir = wght_root_dir,
            sub_domain    = sub_domain
        )

        if load_domain in self.yolo_pool:
            load_rslt = {
                "stat_code"  : 200,
                "stat_msge"  : "COMPLETED",
                "rslt_msge"  : f"{scope_name} 已存在記憶體中",
                "yolo_model" : load_domain,
                "ver_id"     : use_ver_id
            }
            return load_rslt


        if not use_ver_id:
            wght_domain_dir = wght_root_dir / main_domain
            rel_path = input_resolver.to_root_relative(wght_domain_dir)
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "WEIGHT_NOT_FOUND",
                stat = "WARN",
                msge = f"{scope_name} 掛載警示 >>> 查無任何有效之模型版本資料夾 (目錄: {rel_path})"
            )
            load_rslt = {
                "stat_code"  : 404,
                "stat_msge"  : "MISSING_WEIGHT_FILE",
                "rslt_msge"  : f"{scope_name} 查無任何有效之模型版本",
                "yolo_model" : None,
                "ver_id"     : None
            }
            return load_rslt

        # [STEP-2] 最佳權重定位
        # 主領域權重路徑: weights/<DOMAIN>/main/<VER_ID>/best.engine
        # 次領域權重路徑: weights/<DOMAIN>/sub/<SUB_DOMAIN>/<VER_ID>/best.engine
        if sub_domain:
            ver_id_dir = wght_root_dir / main_domain / "sub" / sub_domain / use_ver_id
        else:
            ver_id_dir = wght_root_dir / main_domain / "main" / use_ver_id

        wght_trt_path = ver_id_dir / "best.engine"
        wght_pt_path  = ver_id_dir / "best.pt"

        dst_wght_path = None
        dst_wght_type = "UNK"
        if use_trt and wght_trt_path.exists():
            dst_wght_type = "TRTensor (.engine)"
            dst_wght_path = wght_trt_path

        elif wght_pt_path.exists():
            dst_wght_type = "PyTorch (.pt)"
            dst_wght_path = wght_pt_path
            if use_trt:
                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "FALLBACK_PYTORCH",
                    stat = "WARN",
                    msge = f"{scope_name} 掛載警示 >>> 版本標記 ({use_ver_id}) 找不到 TensorRT 靜態引擎，自動退回使用原生 PyTorch 權重"
                )
        else:
            # 未發現適合之權重檔案 (完全沒有 or 不使用 TRT 且缺少 PT)
            rel_path = input_resolver.to_root_relative(ver_id_dir)
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "WEIGHT_NOT_FOUND",
                stat = "WARN",
                msge = f"{scope_name} 掛載警示 >>> 指定版本目錄下缺少有效之權重檔 ({rel_path})"
            )
            load_rslt = {
                "stat_code"  : 404,
                "stat_msge"  : "MISSING_WEIGHT_FILE",
                "rslt_msge"  : f"{scope_name} 指定版本目錄下缺少權重檔案 ({use_ver_id})",
                "yolo_model" : None,
                "ver_id"     : None
            }
            return load_rslt

        """ [STAGE-2] 驅動 YOLO 模型加載記憶體階段 """
        try:
            yolo_inst = YOLO(str(dst_wght_path))
            self.yolo_pool[load_domain] = yolo_inst
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "LOAD_MODEL_SUCCESS",
                stat = "SUCC",
                msge = f"{scope_name} 掛載成功 >>> 已載入 {dst_wght_type} 權重 (版本: {use_ver_id})"
            )
            load_rslt = {
                "stat_code"  : 200,
                "stat_msge"  : "COMPLETED",
                "rslt_msge"  : f"{scope_name} 模型加載成功",
                "yolo_model" : load_domain,
                "ver_id"     : use_ver_id
            }
            return load_rslt

        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "LOAD_MODEL_EXCEPTION",
                stat = "FAIL",
                msge = f"{scope_name} 掛載失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            load_rslt = {
                "stat_code"  : 500,
                "stat_msge"  : "LOAD_MODEL_EXCEPTION",
                "rslt_msge"  : f"{scope_name} 模型載入過程遭遇非預期系統異常",
                "yolo_model" : None,
                "ver_id"     : None
            }
            return load_rslt

    # =============================================
    def _load_ocr_model(self, lang="chinese_cht"):
        """
        [名稱] Func.D-2 OCR 模型掛載函式
        [功能] 預載入 PaddleOCR 字元辨識模型至 self.tool_pool 映射表，供各領域管線查表索取，避免辨識期重複加載。
        [參數] lang: [str] 欲載入之辨識語系代碼；預設為繁體中文(含英數) "chinese_cht"
        [輸出] dict: 內含執行結果之結構化字典，共計 4 組鍵值，以下說明:
               - stat_code : [int] 執行狀態代碼 (200-成功, 500-系統異常)
               - stat_msge : [str] 狀態識別標籤 (COMPLETED, LOAD_OCR_EXCEPTION)
               - rslt_msge : [str] 執行結果之詳細文字說明
               - img_tool  : [str] 成功加載之影像處理工具名稱，若失敗則為 None
        """
        tool_key = f"char_ocr@{lang}"   # 模型識別金鑰

        # [STEP-1] 若 OCR 模型實例已存在於記憶體中，則直接回傳
        if tool_key in self.tool_pool:
            load_rslt = {
                "stat_code" : 200,
                "stat_msge" : "COMPLETED",
                "rslt_msge" : "OCR 引擎已存在記憶體中",
                "img_tool"  : tool_key
            }
            return load_rslt

        # [STEP-2] 驅動 PaddleOCR 載入記憶體
        try:
            # 關閉「方向分類器」以提升辨識速度，並使用「繁中」與「英數」模型來進行字元辨識
            ocr_inst = PaddleOCR(use_angle_cls=False, lang=lang)
            # 將此模型實例存放至重型影像處理工具實例映射表，供下層領域管線引用
            self.tool_pool[tool_key] = ocr_inst
            log.CONTENT(
                type = "MODEL",
                targ = "OCR-Paddle",
                idnt = "LOAD_OCR_SUCCESS",
                stat = "SUCC",
                msge = f"[模型: PaddleOCR] 掛載成功 >>> 字元辨識引擎已預載入記憶體，使用語系({lang})"
            )
            load_rslt = {
                "stat_code" : 200,
                "stat_msge" : "COMPLETED",
                "rslt_msge" : "PaddleOCR 字元辨識引擎加載成功",
                "img_tool"  : tool_key
            }
            return load_rslt
        
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "MODEL",
                targ = "OCR-Paddle",
                idnt = "LOAD_OCR_EXCEPTION",
                stat = "FAIL",
                msge = f"[模型: PaddleOCR] 掛載失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            load_rslt = {
                "stat_code" : 500,
                "stat_msge" : "LOAD_OCR_EXCEPTION",
                "rslt_msge" : "PaddleOCR 字元辨識引擎加載遭遇非預期系統異常",
                "img_tool"  : None
            }
            return load_rslt

    # =============================================
    def __call__(self, domain, ver_id=None, use_trt=True):
        """
        [名稱] Func.D 視覺辨識模型加載函式
        [功能] 統籌單/雙階層架構之端到端 YOLO 模型預載 (主/次階層)，並動態預載階層策略指定之重型影像推論工具 (e.g., char_ocr)。
        [參數] 共計 3 組參數，以下說明:
               - domain  : [str] 目標領域識別代碼 (e.g., "VEH", "MARA")
               - ver_id  : [str] 目標領域版本代碼 (e.g., "v1.0.0")；若未提供則自動使用模型之最新版本
               - use_trt : [bool] 是否優先載入 TensorRT (.engine) 模型；預設為 True
        [輸出] dict: 內含執行結果之結構化字典，共計 4 組鍵值，以下說明:
               - stat_code : [int] 執行狀態代碼 (200-成功, 206-部分就緒, 400-規則缺失, 404-缺檔, 500-系統異常)
               - stat_msge : [str] 狀態識別標籤 (COMPLETED, PARTIAL_SUCCESS, MISSING_RULE, SYSTEM_ERROR)
               - rslt_msge : [str] 執行結果之詳細文字說明
               - details   : [dict] 共計 6 組任務指標明細，以下說明:
                 - domain      : [str] 目標領域識別代碼 (e.g., "VEH")
                 - ver_id      : [str] 目標領域版本代碼 (e.g., v1.0.0)
                 - stage_qty   : [int] 模型階層數量 (1: 單階層全圖辨識, 2: 二階層串聯微觀辨識)
                 - sub_domains : [list] 次領域識別代碼清單，若為單階層則為 []
                 - yolo_models : [list] 成功加載之 YOLO 模型名稱清單
                 - img_tools   : [list] 成功加載之影像處理工具清單
        """
        domain = str(domain).strip().upper()
        actual_main_ver = ver_id
        stage_name = f"PRELOAD-{domain}"
        log.BANNER(acnt="PRELOAD-TASK", msin=f"模型預載任務，載入 [{domain}] 領域管線")

        try:
            """ [STAGE-1] 階層辨識策略驗證階段 """
            stage_rule = self.stage_cfg.get(domain, {})

            if not stage_rule:
                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "VALIDATE_STAGE_RULE",
                    stat = "WARN",
                    msge = f"[領域: {domain}] 驗證警示 >>> 未定義階層策略規則，無法進行模型預載"
                )
                log.FOOTER(acnt="PRELOAD-HALT", rslt="模型預載暫停流程")      

                preload_mngr_info = {
                    "stat_code" : 400,
                    "stat_msge" : "MISSING_STAGE_RULE",
                    "rslt_msge" : f"[{domain}] 未定義階層策略規則，請檢查 stage_maps",
                    "details"   : {
                        "domain"      : domain,
                        "ver_id"      : ver_id,
                        "stage_qty"   : 0,
                        "sub_domains" : [],
                        "yolo_models" : [],
                        "img_tools"   : []
                    }
                }
                return preload_mngr_info

            stage_qty = stage_rule["stage_qty"]

            """ [STAGE-2] 任務指標初始化階段 """
            has_partial = False
            sub_domains = []
            yolo_models = []
            img_tools   = []
            required_tools = []   # 儲存工具名稱與參數 [("char_ocr", {"lang": "en"}), ...]

            """ [STAGE-3] 主階層 (Stage-1) 模型預載階段 """
            stage_1_cfg = stage_rule["stage_1"]
            main_domain = str(stage_1_cfg["domain"]).upper().strip()
            stage_1_load = self._load_yolo_model(
                domain     = main_domain,
                sub_domain = None,
                ver_id     = ver_id,
                use_trt    = use_trt
            )
            actual_main_ver = stage_1_load.get("ver_id") or ver_id

            if stage_1_load["yolo_model"] is None:
                log.CONTENT(
                    type = "MODEL",
                    targ = stage_name,
                    idnt = "MAIN_MODEL_LOAD_FAILED",
                    stat = "FAIL",
                    msge = f"[領域: {main_domain}] 預載失敗 >>> 無法成功載入主階層辨識核心模型"
                )
                log.FOOTER(acnt="PRELOAD-FAIL", rslt="主階層核心模型載入失敗，終止預載管線")

                preload_mngr_info = {
                    "stat_code"   : stage_1_load["stat_code"],
                    "stat_msge"   : stage_1_load["stat_msge"],
                    "rslt_msge"   : f"[{main_domain}] 主階層核心模型載入失敗，無法執行視覺辨識",
                    "details"     : {
                        "domain"      : domain,
                        "ver_id"      : actual_main_ver,
                        "stage_qty"   : stage_qty,
                        "sub_domains" : [],
                        "yolo_models" : [],
                        "img_tools"   : []
                    }
                }
                return preload_mngr_info
            
            yolo_models.append(stage_1_load["yolo_model"])
            for tool in stage_1_cfg.get("img_tools", []):
                if isinstance(tool, dict):
                    for tool_name, tool_params in tool.items():
                        if tool_name in self.tool_loaders: 
                            required_tools.append((tool_name, tool_params))   # 每筆格式: (工具名稱, 函式參數字典)

            """ [STAGE-4] 次階層 (Stage-2) 模型預載階段 """
            if stage_qty >= 2:
                stage_2_cfg = stage_rule.get("stage_2", {})
                sub_tasks = stage_2_cfg.get("sub_tasks", {})

                for sub_domain, sub_info in sub_tasks.items():
                    stage_2_load = self._load_yolo_model(
                        domain     = main_domain,
                        sub_domain = sub_domain,
                        ver_id     = ver_id,
                        use_trt    = use_trt
                    )

                    if stage_2_load["yolo_model"] is None:
                        has_partial = True
                        log.CONTENT(
                            type = "MODEL",
                            targ = stage_name,
                            idnt = "SUB_MODEL_LOAD_FAILED",
                            stat = "FAIL",
                            msge = f"[領域: {main_domain}/{sub_domain}] 預載失敗 >>> 無法成功載入次階層辨識核心模型"
                        )
                        log.FOOTER(acnt="PRELOAD-FAIL", rslt="次階層核心模型載入失敗，終止預載管線")

                        preload_mngr_info = {
                            "stat_code"   : stage_2_load["stat_code"],
                            "stat_msge"   : stage_2_load["stat_msge"],
                            "rslt_msge"   : f"[{main_domain}/{sub_domain}] 次階層核心模型載入失敗，無法執行視覺辨識",
                            "details"     : {
                                "domain"      : domain,
                                "ver_id"      : actual_main_ver,
                                "stage_qty"   : stage_qty,
                                "sub_domains" : sub_domains,
                                "yolo_models" : yolo_models,
                                "img_tools"   : img_tools
                            }
                        }
                        return preload_mngr_info

                    sub_domains.append(sub_domain)
                    yolo_models.append(stage_2_load["yolo_model"])
                    for tool in sub_info.get("img_tools", []):
                        if isinstance(tool, dict):
                            for tool_name, tool_params in tool.items():
                                if tool_name in self.tool_loaders:
                                    required_tools.append((tool_name, tool_params))   # 每筆格式: (工具名稱, 函式參數字典)

            """ [STAGE-5] 影像處理工具引擎預載階段 """
            for tool_name, tool_params in required_tools:
                load_func = self.tool_loaders.get(tool_name)
                if load_func is None:
                    # 無須預載工具 (e.g., color_extr)，安全略過
                    continue

                tool_load_rslt = load_func(**tool_params) if tool_params else load_func()
                if tool_load_rslt["img_tool"] is None:
                    has_partial = True
                    log.CONTENT(
                        type = "MODEL",
                        targ = stage_name,
                        idnt = "TOOL_PRELOAD_WARN",
                        stat = "FAIL",
                        msge = f"[工具: {tool_name}] 預載失敗 >>> 無法成功載入影像處理工具"
                    )
                else:
                    loaded_tool = tool_load_rslt["img_tool"]
                    if loaded_tool not in img_tools:
                        img_tools.append(loaded_tool)

            """ [STAGE-6] 預載成果彙整階段 """
            yolo_models_str = ", ".join(yolo_models)
            img_tools_str   = ", ".join(img_tools) if img_tools else "無"
            log.FOOTER(acnt="PRELOAD-DONE", rslt=f"[{domain}] 領域預載完成，辨識引擎({yolo_models_str})、引擎工具({img_tools_str})")

            preload_mngr_info = {
                "stat_code"   : 200 if not has_partial else 206,
                "stat_msge"   : "COMPLETED" if not has_partial else "PARTIAL_SUCCESS",
                "rslt_msge"   : f"[{domain}] 視覺辨識模型預載全數完成" if not has_partial else f"[{domain}] 視覺辨識模型預載部分完成",
                "details"     : {
                    "domain"      : domain,
                    "ver_id"      : actual_main_ver,
                    "stage_qty"   : stage_qty,
                    "sub_domains" : sub_domains,
                    "yolo_models" : yolo_models,
                    "img_tools"   : img_tools
                }
            }
            return preload_mngr_info
            
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "MODEL",
                targ = stage_name,
                idnt = "PRELOAD_EXCEPTION",
                stat = "FAIL",
                msge = f"[任務: 模型預載任務({domain})] 運行失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            log.FOOTER(acnt="PRELOAD-FAIL", rslt="模型預載發生異常，請調閱堆疊日誌排查問題")
            preload_mngr_info = {
                "stat_code"   : 500,
                "stat_msge"   : "SYSTEM_ERROR",
                "rslt_msge"   : f"[{stage_name}] 預載過程遭遇非預期系統異常",
                "details"     : {
                    "domain"      : domain,
                    "ver_id"      : actual_main_ver,
                    "stage_qty"   : 0,
                    "sub_domains" : [],
                    "yolo_models" : [],
                    "img_tools"   : []
                }
            }
            return preload_mngr_info

