# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 資料庫調度管理工具 (Database Orchestration Manager Tool)
# 維護日期: 2026-08-25
# 檔案路徑: MARS_Project/src/database/db_mngr.py
# ##########################################################################################

# 掛載外部依賴
import traceback
import psycopg2
from psycopg2 import extras

# 掛載內部依賴
from config import DB_PARAMS
from utils import log

# ==========================================================================================
class DBManager:
    """
    [名稱] 資料庫調度管理引擎 (Database Orchestration Manager Engine)
    [作用] 負責 PostgreSQL「高效批次寫入」與「通用數據查詢」之邏輯調度，並動態生成衝突策略，確保資料精確落地。
    """
    # =============================================
    def __init__(self):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「資料庫調度管理引擎」，載入資料庫連線參數，建立底層 SQL 執行環境與交易調度基礎。
        [參數] None: 無需使用外部參數
        [輸出] None: 依定義之屬性完成初始化，提供後續 SQL 指令執行環境
        """
        self.params = DB_PARAMS   # 資料庫連線參數

    # =============================================
    def _build_conflict_fragments(self, upsert_df, table_path, cflt_keys, w_mode):
        """
        [名稱] Func.B 衝突處理語法建構函式
        [功能] 動態構建 PostgreSQL 衝突處理語句 (ON CONFLICT)，支援多維度寫入策略。
        [參數] 共計 4 組參數，以下說明:
               - upsert_df  : [pd.DataFrame] 待寫入之數據集，用於提取目標欄位
               - table_path : [str] 完整資料表路徑 (schema.table_name)，用於 SQL 語句之別名定位
               - cflt_keys  : [list] 資料表主鍵清單，作為觸發衝突之判定基準
               - w_mode     : [dict] 欄位寫入策略配置；預設為 overwrite (覆寫模式)
        [輸出] str: 封裝完成之 DO UPDATE SET 衝突處理語句
        """
        upd_frags = []   # 數據更新 SQL 語法片段 (Update)
        cnd_frags = []   # 變異偵測 SQL 語法片段 (Condition)

        # [STEP-1] 構建「數據更新」及「變異偵測」語法片段
        update_cols = [col for col in upsert_df.columns if col not in cflt_keys]
        for col in update_cols:
            # 數據更新模式判斷，預設為覆蓋模式
            mode = w_mode.get(col, "overwrite")
            if mode == "coalesce":
                # 保留模式: 優先保留現值，僅在現有值為 NULL 時更新
                upd_frags.append(f'"{col}" = COALESCE({table_path}."{col}", EXCLUDED."{col}")')
            else:
                # 覆蓋模式: 強制以新數據覆蓋現有值
                upd_frags.append(f'"{col}" = EXCLUDED."{col}"')
            
            # 每個欄位增加變異偵測條件
            cnd_frags.append(f'{table_path}."{col}" IS DISTINCT FROM EXCLUDED."{col}"')

        # [STEP-2] 整合完整 DO UPDATE SET 子句, 並附加「系統更新時間戳」與「變異過濾器」
        if upd_frags:
            # 只要有任一「實體欄位」發生變異 (True)，就執行「數值全盤覆蓋」與「update_ts 時間戳更新」
            upd_clse = ", ".join(upd_frags) + ", update_ts = CURRENT_TIMESTAMP"
            cnd_clse = " OR ".join(cnd_frags)
            cflt_act = f"DO UPDATE SET {upd_clse} WHERE {cnd_clse}"
        else:
            cflt_act = "DO NOTHING"

        return cflt_act

    # =============================================
    def batch_upsert(self, upsert_df, table_info):
        """
        [名稱] Func.C 數據批次寫入函式
        [功能] 動態解析資料表配置，執行 PostgreSQL 數據高速批次寫入。
        [參數] 共計 2 組參數，以下說明:
               - upsert_df  : [pd.DataFrame] 待寫入之標準化數據集
               - table_info : [dict] 目標資料表配置資訊字典
        [輸出] int: 成功寫入之有效數據筆數；若數據集為空則回傳 0
        """
        try:
            """ [STAGE-1] 資料表資訊解析階段 """
            schema     = table_info["schm"]
            table_name = table_info["tabl"]
            ukey       = table_info["ukey"]
            mode       = table_info.get("mode")
            
            # 驗證更新模式組態格式，攔截配置遺失或格式異常者
            if not isinstance(mode, dict):
                log.CONTENT(
                    type = "CONFIG",
                    targ = f"{schema}_schm.py",
                    idnt = "VLDT_MODE",
                    stat = "WARN",
                    msge = f"[資料表: {table_name}] 寫入警示 >>> 資料更新模式 (mode) 配置遺失或格式錯誤"
                )
                return 0
            
            # 驗證待寫入數據集狀態，攔截資料為空或未定義者
            if upsert_df is None or upsert_df.empty:
                log.CONTENT(
                    type = "DB",
                    targ = "MARS_DB", 
                    idnt = "UPST_DATA",
                    stat = "WARN", 
                    msge = f"[資料表: {table_name}] 寫入警示 >>> 待寫入數據為空，無法執行批次寫入"
                )
                return 0

            """ [STAGE-2] 建構批次寫入語法階段 """
            table_path    = f'"{schema}"."{table_name}"'
            cols_list     = upsert_df.columns.tolist()
            cols_str      = ", ".join([f'"{col}"' for col in cols_list])
            placeholders  = ", ".join([f"%({col})s" for col in cols_list])
            cflt_keys_str = ", ".join([f'"{key}"' for key in ukey])

            # 整合完整 SQL 批次寫入語法 (INSERT ... ON CONFLICT DO UPDATE)
            cflt_acnts = self._build_conflict_fragments(
                upsert_df  = upsert_df,
                table_path = table_path,
                cflt_keys  = ukey,
                w_mode     = mode
            )
            upsert_sql = f"""
                INSERT INTO {table_path} ({cols_str}) VALUES ({placeholders})
                ON CONFLICT ({cflt_keys_str})
                {cflt_acnts};
            """

            """ [STAGE-3] 數據批次寫入階段 """
            upst_cnt = 0   # 成功寫入之數據筆數

            with psycopg2.connect(**self.params) as conn:
                with conn.cursor() as cur:
                    # 轉為 Record 格式 (字典串列: [{}, {}, ...{}])
                    records = upsert_df.to_dict("records")
                    extras.execute_batch(cur, upsert_sql, records)
            upst_cnt = len(upsert_df)
            
            log.CONTENT(
                type = "DB",
                targ = "MARS_DB",
                idnt = "UPST_DATA",
                stat = "SUCC",
                msge = f"[資料表: {table_name}] 寫入成功 >>> 共計更新 {upst_cnt} 筆數據"
            )
            return upst_cnt

        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            safe_tb_name = table_info.get("tabl", "UNK") if isinstance(table_info, dict) else "UNK"
            log.CONTENT(
                type = "DB",
                targ = "MARS_DB",
                idnt = "UPST_EXCP",
                stat = "FAIL",
                msge = f"[資料表: {safe_tb_name}] 寫入失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            return 0

    # =============================================
    def exec_query(self, sql_cmd, vars=None):
        """
        [名稱] Func.D 數據標準查詢函式
        [功能] 執行 PostgreSQL 通用查詢語句，並提取完整數據結果集。
        [參數] 共計 2 組參數，以下說明:
               - sql_cmd : [str] 待執行之 SQL 查詢語句，支援佔位符綁定
               - vars    : [tuple/dict] 選擇性綁定參數，安全傳遞動態過濾條件
               # Remark-1: 若 SQL 使用匿名佔位符--%s，則傳入格式必須為 tuple
               # Remark-2: 若 SQL 使用命名佔位符--%(key)s，則傳入格式必須為 dict
        [輸出] list: 封裝查詢結果之元組清單 (List of Tuples)；若查無數據則回傳空清單
        """
        try:
            with psycopg2.connect(**self.params) as conn:
                with conn.cursor() as cur:
                    # 使用參數化查詢, 有效防禦 SQL 注入攻擊
                    cur.execute(sql_cmd, vars)
                    return cur.fetchall()
        
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "DB",
                targ = "MARS_DB",
                idnt = "QUER_EXCP",
                stat = "FAIL",
                msge = f"[資料庫: MARS_DB] 查詢失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            return []
        
# =============================================
# ⭐｜模組導出｜實例化管理對象
# =============================================
""" 使用底線標記為內部私有實例，防止外部 import * 時污染命名空間 """
_dtbs_mngr = DBManager()

""" 對外開放的正式單例接口 (公開名稱) """
# 1. dtbs_mngr  : [實例] 資料庫調度管理引擎 >>> 負責管控全域資料庫之連線配置、交易調度與事務執行。
# 2. SQL_UPSERT : [封裝] 數據批次寫入函式 >>> 執行 PostgreSQL 高速批次寫入作業，並支援動態衝突策略。
# 3. SQL_QUERY  : [封裝] 數據標準查詢函式 >>> 執行 PostgreSQL 參數化查詢作業，兼備 SQL 注入防禦機制。

dtbs_mngr  = _dtbs_mngr
SQL_UPSERT = _dtbs_mngr.batch_upsert
SQL_QUERY  = _dtbs_mngr.exec_query