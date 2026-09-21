# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 資料庫全域組態配置 (Global Database Configuration)
# 維護日期: 2026-09-11
# 檔案路徑: MARS_Project/config/db_cfg.py
# ##########################################################################################

# 掛載外部依賴
import os
import traceback
from dotenv import load_dotenv

# 掛載內部依賴
from utils import log
from .maps import schm_maps

load_dotenv()   # 載入環境變數

# ==========================================================================================
class DBConfig:
    """
    [名稱] 資料庫組態管理引擎 (Database Configuration Management Engine)
    [作用] 執行全域環境變數解析與安全驗證，集中管理底層資料庫驅動程序所需之連線與映射結構定義。
    """
    # =============================================
    def __init__(self):
        """
        [名稱] Func.A 類別組態配置
        [功能] 啟動資料庫核心參數掛載、資料表屬性檢索與身分授權防呆驗證。
        [參數] None: 無需使用外部參數。
        [輸出] None: 依定義完成屬性封裝，提供全域連線與查詢調用。
        """
        self._validate_db_params()   # 啟動環境驗證

        """ [ATTR-1] 資料庫連線參數 """
        self.params = {
            "host"     : os.getenv("DB_HOST"),           # 伺服器名稱
            "port"     : os.getenv("DB_PORT", "5432"),   # 連線埠
            "database" : os.getenv("DB_NAME"),           # 資料庫名稱
            "user"     : os.getenv("DB_USERNAME"),       # 使用者名稱
            "password" : os.getenv("DB_PASSWORD")        # 使用者密碼
        }

        """ [ATTR-2] 資料表屬性檢索 """
        self.tables = {
            "train" : {**schm_maps.TRAIN_SCHEMA_INFO},   # [訓練階段] 資料表屬性定義
            "infer" : {**schm_maps.INFER_SCHEMA_INFO}    # [辨識階段] 資料表屬性定義
        }

    # =============================================
    def _validate_db_params(self):
        """
        [名稱] Func.B 環境變數驗證函式
        [功能] 執行嚴格邊界檢查，確保環境變數 (.env) 具備建立資料庫連線所需之核心參數。
        [參數] None: 無需使用外部參數。
        [輸出] None: 若驗證通過則無回傳；若核心參數遺失則拋出例外並中斷系統。
        """
        err_ctx = {}   # 異常診斷標籤

        try:
            # [STEP-1] 驗證核心定址組態狀態，攔截主機或資料庫名稱遺失者
            if not os.getenv("DB_HOST") or not os.getenv("DB_NAME"):
                err_ctx["log_idnt"] = "VALIDATE_DATABASE_CORE"
                err_ctx["log_msge"] = "[環境變數: .env] 驗證失敗 >>> 缺少 DB_HOST 或 DB_NAME 核心參數"
                err_ctx["rse_idnt"] = "CORE-參數遺失"
                raise RuntimeError()
            
            # [STEP-2] 驗證身分授權組態狀態，攔截使用者名稱或密碼遺失者
            if not os.getenv("DB_USERNAME") or not os.getenv("DB_PASSWORD"):
                err_ctx["log_idnt"] = "VALIDATE_DATABASE_AUTH"
                err_ctx["log_msge"] = "[環境變數: .env] 驗證失敗 >>> 缺少 DB_USERNAME 或 DB_PASSWORD 授權參數"
                err_ctx["rse_idnt"] = "AUTH-參數遺失"
                raise RuntimeError()

        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            err_ctx["log_idnt"] = err_ctx.get("log_idnt", "VALIDATE_DATABASE_EXCEPTION")
            err_ctx["log_msge"] = err_ctx.get("log_msge", f"[環境變數: .env] 驗證失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}")
            err_ctx['rse_idnt'] = err_ctx.get("rse_idnt", "VALIDATE-系統異常")

            log.CONTENT(
                type = "FILE",
                targ = ".env",
                idnt = err_ctx["log_idnt"],
                stat = "FAIL",
                msge = err_ctx["log_msge"]
            )
            raise RuntimeError(f"[FAIL] 錯誤來源 >>> [DIR] config >> [FILE] db_cfg.py >> [Func.B] _validate_db_params ({err_ctx['rse_idnt']})") from None

# =============================================
# ⭐｜模組導出｜實例化管理對象
# =============================================
""" 使用底線標記為內部私有實例，防止外部 import * 時污染命名空間 """
_db_config = DBConfig()

""" 對外開放的正式單例接口 (公開名稱) """
# 1. db_config : [實例] 資料庫組態管理引擎 >>> 負責全域資料庫環境參數之集中管理與嚴格驗證。
# 2. DB_PARAMS : [封裝] 資料庫連線組態字典 >>> 儲存底層驅動程序所需之核心定址與授權參數。
# 3. TABLES    : [封裝] 資料表屬性檢索字典 >>> 儲存全域資料表之架構歸屬、實體名稱與鍵值等核心配置。

db_config = _db_config
DB_PARAMS = _db_config.params
TABLES    = _db_config.tables