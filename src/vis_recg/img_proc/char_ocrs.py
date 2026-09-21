# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 光學字元辨識工具 (Optical Character Recognition Tool)
# 維護日期: 2026-09-22
# 檔案路徑: MARS_Project/src/vis_recg/img_proc/char_ocrs.py
# ##########################################################################################

# 掛載外部依賴
import re
import traceback
import numpy as np

# 掛載內部依賴
from utils import log
from .char_rules import CHAR_RULES

# ==========================================================================================
class CharReader:
    """
    [名稱] 光學字元辨識引擎 (Optical Character Recognition Engine)
    [作用] 接收拉直後之矩形 ROI 影像，調用預先載入或內部降級之 OCR 模型提取英數字串，並自動執行格式正規化與雜訊過濾。
    """
    # =============================================
    def __init__(self, sys_cfg=None, ocr_inst=None):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「光學字元辨識引擎」，優先注入由 modl_load 統一載入之 OCR 實例；若未傳入則啟用延遲載入策略進行內部降級加載。
        [參數] 共計 2 組參數，以下說明:
               - sys_cfg  : [dict] 全域組態檢索 (選填，保留未來擴充彈性)
               - ocr_inst : [object] 預先載入之 OCR 引擎模型實例 (選填)
        [輸出] None: 依定義之屬性完成初始化
        """
        self.sys_cfg  = sys_cfg
        self.ocr_inst = ocr_inst
        self.backend_type = "INJECTED" if ocr_inst is not None else "NONE"

        # 若外部 (modl_load) 未注入模型實例，則啟用內部容錯降級機制
        if ocr_inst is None:
            self._init_backend()

    # =============================================
    def _init_backend(self):
        """
        [名稱] Func.B 內部備援辨識引擎初始化
        [功能] 嘗試載入內部 OCR 辨識引擎實例 (優先 PaddleOCR、降級 EasyOCR)，確保單元測試與獨立呼叫時之系統可用性。
        [參數] None: 無需外部傳入參數，依據本機安裝之辨識套件動態偵測載入
        [輸出] None: 依載入狀態直接指派 self.ocr_inst 與 self.backend_type 屬性
        """
        try:
            from paddleocr import PaddleOCR
            # 關閉「方向分類器」以提升辨識速度，並使用「繁中」與「英數」模型來進行字元辨識
            self.ocr_inst = PaddleOCR(use_angle_cls=False, lang="chinese_cht")
            self.backend_type = "PADDLE"
        except ImportError:
            try:
                import easyocr
                # 啟用 GPU 硬體加速並關閉 verbose 防止載入期間印出底層資訊
                self.ocr_inst = easyocr.Reader(lang_list=["en", "ch_tra"], gpu=True, verbose=False)
                self.backend_type = "EASY"
            except ImportError:
                self.ocr_inst = None
                self.backend_type = "NONE"
                log.CONTENT(
                    type = "MODEL",
                    targ = "CHAR-OCR",
                    idnt = "LOAD_MODEL",
                    stat = "WARN",
                    msge = f"[套件: PaddleOCR/easyocr] 載入警示 >>> 未偵測到預載入實例或本機 OCR 套件，字元辨識功能將受限"
                )

    # =============================================
    def _clean_text(self, raw_text, domain):
        """
        [名稱] Func.C 文字正規篩選與特徵屬性解構函式
        [功能] 調用 CHAR_RULES 規則表解構文字特徵：分離出「主號碼」與「特徵標籤」，並過濾雜訊字詞。
        [參數] 共計 2 組參數，以下說明:
               - raw_text : [str] OCR 辨識產出之原始字串
               - domain   : [str] 目標領域識別代碼 (e.g., "VEH", "LIC")
        [輸出] tuple: (clean_text, char_attr)
               - clean_text : [str] 清洗後之標準字串
               - char_attr  : [str] 提取出之目標特徵標籤，若無則為 default_attr
        """
        domain = str(domain).strip().upper()
        rule   = CHAR_RULES.get(domain, {})
        default_attr = rule.get("default_attr", "STANDARD")

        if raw_text is None:
            return "", default_attr
        
        raw_str = str(raw_text).strip()
        if not raw_str:
            return "", default_attr
        
        # 依領域規則動態決定是否強制轉為大寫
        scrubbed_str = raw_str.upper() if rule.get("force_upper", False) else raw_str 
        char_attr    = default_attr

        # [STEP-1] 依特徵標籤規則進行特徵識別提取
        for attr_name, triggers in rule.get("attr_patterns", []):
            if any(token in scrubbed_str for token in triggers):
                char_attr = attr_name
                break

        # [STEP-2] 移除干擾字詞與裝飾性標記
        for token in rule.get("noise_tokens", []):
            scrubbed_str = scrubbed_str.replace(token, "")

        # [STEP-3] 套用白名單正則過濾，若規則未定義 regex 則縮減多餘空白
        whitelist_regex = rule.get("whitelist_regex")
        if whitelist_regex is not None:
            filtered_str = whitelist_regex.sub("", scrubbed_str).strip()

            # 依設定檔動態讀取並套用前綴抑制
            prefix_supp_regex = rule.get("prefix_supp_regex")
            if prefix_supp_regex is not None:
                clean_text = prefix_supp_regex.sub(r"\1", filtered_str)
            else:
                clean_text = filtered_str
        else:
            clean_text = re.sub(r"\s+", " ", scrubbed_str)
        
        return clean_text, char_attr
        
    # =============================================
    def __call__(self, roi_crop, domain):
        """
        [名稱] Func.D 光學字元辨識函式
        [功能] 接收平整 ROI 影像，呼叫 ocr_inst 執行文字辨識，回傳依領域清洗後之通用字串與信心數值。
        [參數] 共計 2 組參數，以下說明:
               - roi_crop : [numpy.ndarray] 已校正之矩形 BGR 影像陣列 (H, W, C)
               - domain   : [str] 目標領域識別代碼 (e.g., "VEH", "LIC")
        [輸出] dict: 內含執行結果之結構化字典，共計 3 組鍵值，以下說明:
               - char_text : [str] 辨識並純化後之核心字串 (e.g., "EAB1234"，失敗為 "UNK")
               - char_attr : [str] 目標特徵屬性標籤 (e.g., "電動車"、"STANDARD")
               - char_conf : [float] 辨識信心度 (0.0 ~ 1.0)
        """
        # 驗證輸入參數有效性
        domain = str(domain).strip().upper() if domain is not None else ""
        if roi_crop is None or not domain or domain not in CHAR_RULES:
            if roi_crop is None:
                err_args, err_stat = "roi_crop", "輸入影像為 None"
            elif not domain:
                err_args, err_stat = "domain", "輸入領域為空字串或 None"
            else:
                err_args, err_stat = "domain", f"輸入領域識別代碼 ({domain}) 未定義規則"
            log.CONTENT(
                type = "PROCESS",
                targ = "CHAR-READ",
                idnt = "VALIDATE_ARGUMENTS",
                stat = "WARN",
                msge = f"[參數: {err_args}] 驗證警示 >>> {err_stat}，跳過光學字元辨識程序"
            )
            char_rslt = {
                "char_text" : "UNK",
                "char_attr" : "UNK",
                "char_conf" : 0.0
            }
            return char_rslt

        # 驗證底層 OCR 辨識引擎狀態
        if self.ocr_inst is None:
            log.CONTENT(
                type = "PROCESS",
                targ = "CHAR-READ",
                idnt = "CHECK_BACKEND_STATUS",
                stat = "WARN",
                msge = f"[引擎: CHAR-OCR] 驗證警示 >>> OCR 引擎未實例化，跳過光學字元辨識程序"
            )
            char_rslt = {
                "char_text" : "UNK",
                "char_attr" : "UNK",
                "char_conf" : 0.0
            }
            return char_rslt
        
        # 執行文字檢測與多行空間資訊提取
        try:
            text_lines = []   # 支援多行與多文字塊空間串接，避免折行或多欄位文字遺失
            # [MODL-1] PaddleOCR 光學字元辨識模組
            if hasattr(self.ocr_inst, "ocr"):
                results = self.ocr_inst.ocr(roi_crop, cls=False)
                if results and results[0]:
                    for line in results[0]:
                        if line and len(line) >= 2:
                            # 提取幾何座標左上角 Y 座標，作為垂直排序依據
                            bbox_pts   = line[0]
                            top_y_pts  = min(pt[1] for pt in bbox_pts) if bbox_pts and len(bbox_pts) == 4 else 0.0
                            text_lines.append({
                                "top_y_pts" : top_y_pts,          # 文字框頂部 Y 座標，供空間垂直排序使用
                                "raw_text"  : str(line[1][0]),    # 該文字區塊之原始辨識字串
                                "char_conf" : float(line[1][1])   # 該文字區塊之辨識信心數值
                            })
            # [MODL-2] EasyOCR 光學字元辨識模組
            elif hasattr(self.ocr_inst, "readtext"):
                results = self.ocr_inst.readtext(roi_crop)
                if results and len(results) > 0:
                    for line in results:
                        if line and len(line) >= 3:
                            bbox_pts   = line[0]
                            top_y_pts  = min(pt[1] for pt in bbox_pts) if bbox_pts and len(bbox_pts) == 4 else 0.0
                            text_lines.append({
                                "top_y_pts" : top_y_pts,          # 文字框頂部 Y 座標，供空間垂直排序使用
                                "raw_text"  : str(line[1]),       # 該文字區塊之原始辨識字串
                                "char_conf" : float(line[2])      # 該文字區塊之辨識信心數值
                            })

            # [STEP-1] 依垂直空間座標 (由上而下) 排序，確保多行讀取時序一致
            text_lines.sort(key=lambda item: item["top_y_pts"])
            if not text_lines:
                # 檢驗辨識結果是否包含文字區塊，若全無文字則快速熔斷返回
                return {
                    "char_text" : "UNK",
                    "char_attr" : "UNK",
                    "char_conf" : 0.0
                }

            # [STEP-2] 提取領域規格與單行格式比對正則
            rule = CHAR_RULES.get(domain, {})
            target_regex = rule.get("target_regex")
            matched_line = None
            if target_regex is not None:
                # 檢查是否有獨立單行已高度吻合目標領域標準格式
                for line in text_lines:
                    clean_text, _ = self._clean_text(raw_text=line["raw_text"], domain=domain)
                    if target_regex.match(clean_text):
                        matched_line = (clean_text, line["char_conf"])
                        break

            # [STEP-3] 聚合多片段文字全文，解構全域特徵屬性 (如 "電動車"、"軍車" 等)
            raw_full_text = "".join(line["raw_text"] for line in text_lines)
            _, char_attr = self._clean_text(raw_text=raw_full_text, domain=domain)

            if matched_line is not None:
                # [SITU-A] 行級單行命中判別: 若單行即符合目標規格，直接採納該行作為主字串與信心值，避免與標籤字詞無腦硬拼
                char_text = matched_line[0]
                char_conf = float(matched_line[1])
            else:
                # [SITU-B] 跨行拼接回退判別: 無單行命中時，啟用跨行拼接回退通道 (適用於折行排版如上行"軍"、下行"C1234")
                char_text, _ = self._clean_text(raw_text=raw_full_text, domain=domain)
                conf_scores  = [line["char_conf"] for line in text_lines]
                char_conf    = float(np.mean(conf_scores)) if conf_scores else 0.0

            if not char_text:
                char_text = "UNK"
                char_attr = "UNK"
                char_conf = 0.0

            char_rslt = {
                "char_text" : char_text,
                "char_attr" : char_attr,
                "char_conf" : round(char_conf, 4)
            }
            return char_rslt
        
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "PROCESS",
                targ = "CHAR-READ",
                idnt = "HANDLE_CHAR_READ_EXCEPTION",
                stat = "FAIL",
                msge = f"[任務: 光學字元辨識] 運行失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            char_rslt = {
                "char_text" : "UNK",
                "char_attr" : "UNK",
                "char_conf" : 0.0
            }
            return char_rslt