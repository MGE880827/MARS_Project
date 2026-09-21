# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 資料庫架構初始化工具 (Database Schema Initializer Tool)
# 維護日期: 2026-08-25
# 檔案路徑: MARS_Project/src/database/db_init.py
# ##########################################################################################

# 掛載外部依賴
import traceback
import psycopg2
from pathlib import Path

# 掛載內部依賴
from config import DB_PARAMS, TABLES
from utils import log

# ==========================================================================================
class DBInitializer:
    """
    [名稱] 資料庫架構初始化引擎 (Database Schema Initializer Engine)
    [作用] 負責管控全域資料庫生命週期，執行「架構清理」、「標準化建置」與「現況審計」，確保底層環境純淨。
    """
    # =============================================
    def __init__(self):
        """
        [名稱] Func.A 類別組態配置
        [功能] 初始化「資料庫架構初始化引擎」，載入資料庫連線參數，建立後續 SQL 執行環境。
        [參數] None: 無需使用外部參數
        [輸出] None: 依定義之屬性完成初始化，提供後續 SQL 指令執行環境
        """
        self.params = DB_PARAMS   # 資料庫連線參數

    # =============================================
    def _wipe_tables(self):
        """
        [名稱] Func.B 全域資料表移除函式
        [功能] 強制移除全域資料表與架構，徹底清除舊有實體與相依關係。
        [參數] None: 無需使用外部參數
        [輸出] None: 完成架構與資料表刪除作業，並透過結構化日誌工具 (SystemLog) 動態輸出
        """
        log.BANNER(acnt="WIPE-DB", msin="全域資料表移除任務")

        try:
            with psycopg2.connect(**self.params) as conn:
                with conn.cursor() as cur:
                    tabl_cnt = 0   # 成功移除之資料表數量

                    # 反轉 Schema 順序，避免相依性衝突
                    schm_keys = list(TABLES.keys())
                    schm_keys.reverse()

                    for schm_key in schm_keys:
                        schm_info = TABLES[schm_key]   # 取得所屬架構下資料表配置資訊
                        """ [STAGE-1] 資料表移除階段 (TABLE)"""
                        for tabl_key, tabl_info in schm_info.items():
                            schm_name = tabl_info['schm']
                            tabl_name = tabl_info['tabl']
                            cur.execute(f"DROP TABLE IF EXISTS {schm_name}.{tabl_name} CASCADE;")
                            tabl_cnt += 1
                            log.CONTENT(
                                type = "DB",
                                targ = "MARS_DB",
                                idnt = "WIPE_TABL",
                                stat = "SUCC",
                                msge = f"[資料表: {tabl_name}] 移除成功 >>> 目前累計清除 {tabl_cnt} 張資料表"
                            )
                            
                        """ [STAGE-2] 架構移除階段 (SCHEMA)"""
                        cur.execute(f"DROP SCHEMA IF EXISTS {schm_key} CASCADE;")
                        log.CONTENT(
                            type = "DB",
                            targ = "MARS_DB",
                            idnt = "WIPE_SCHM",
                            stat = "SUCC",
                            msge = f"[架構: {schm_key}] 移除成功 >>> 該架構及其相依實體已徹底清除"
                        )
                        
                    log.FOOTER(acnt="WIPE-DONE", rslt=f"全域資料庫清理完畢，共計移除 {tabl_cnt} 張資料表")
                
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "DB",
                targ = "MARS_DB",
                idnt = "WIPE_EXCP",
                stat = "FAIL",
                msge = f"[資料庫: MARS_DB] 移除失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            log.FOOTER(acnt="WIPE-FAIL", rslt="全域資料表移除發生異常，請調閱堆疊日誌排查問題")
            raise RuntimeError(f"[FAIL] 錯誤來源 >>> [DIR] database >> [FILE] db_init.py >> [Func.B] _wipe_tables (WIPE-系統異常)") from None

    # =============================================
    def _init_tables(self):
        """
        [名稱] Func.C 全域資料表部署函式
        [功能] 讀取標準 SQL 實體檔，自動部署全域架構與資料表。
        [參數] None: 無需使用外部參數
        [輸出] None: 完成架構與實體建置作業，並透過結構化日誌工具 (SystemLog) 動態輸出
        """
        err_ctx = {}   # 異常診斷標籤
        log.BANNER(acnt="INIT-DB", msin="全域資料表部署任務")

        try:
            # 建立 SQL 部署指令之根目錄路徑: MARS_Project/src/database/deploy_repo/
            base_sql_path = Path(__file__).parent / "deploy_repo" 

            with psycopg2.connect(**self.params) as conn:
                with conn.cursor() as cur:
                    tabl_cnt = 0   # 成功部署之資料表數量

                    for schm_key, schm_info in TABLES.items():
                        """ [STAGE-1] 架構部署階段 (SCHEMA)"""
                        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schm_key};")
                        log.CONTENT(
                            type = "DB",
                            targ = "MARS_DB",
                            idnt = "INIT_SCHM",
                            stat = "SUCC",
                            msge = f"[架構: {schm_key}] 部署成功 >>> 該架構及其執行環境已準備就緒"
                        )
                        
                        """ [STAGE-2] 資料表部署階段 (TABLE)"""
                        for tabl_key, tabl_info in schm_info.items():
                            schm_name = tabl_info['schm']
                            tabl_name = tabl_info['tabl']
                            
                            # 建立 SQL 部署指令之檔案路徑: MARS_Project/src/database/deploy_repo/<schm_key>/<tabl_key>.sql
                            sql_file = base_sql_path / schm_key / f"{tabl_key}.sql"
                            
                            # 驗證實體檔案存在性，攔截部署指令遺失或不支援者
                            if not sql_file.exists():
                                err_ctx["log_idnt"] = "VLDT_SQLC"
                                err_ctx["log_msge"] = f"[資料表: {tabl_name}] 部署失敗 >>> {tabl_key}.sql 指令檔遺失或不支持"
                                err_ctx["rse_idnt"] = "SQLC-指令遺失"
                                raise RuntimeError()

                            sql_content = sql_file.read_text(encoding="utf-8")
                            tabl_path = f"{schm_name}.{tabl_name}"
                            sql_cmd   = sql_content.replace("{{TARGET_PATH}}", tabl_path)   # 替換部署指令中「路徑標籤」{{TARGET_PATH}}
                            cur.execute(sql_cmd)
                            tabl_cnt += 1

                            # 統計資料表欄位數量
                            cur.execute(f"""
                                SELECT count(*) FROM information_schema.columns
                                WHERE table_schema = '{schm_name}' AND table_name = '{tabl_name}';                
                            """)
                            cols_cnt = cur.fetchone()[0]
                            log.CONTENT(
                                type = "DB",
                                targ = "MARS_DB",
                                idnt = "INIT_TABL",
                                stat = "SUCC",
                                msge = f"[資料表: {tabl_name}] 部署成功 >>> 結構建置完畢，欄位共計 {cols_cnt} 個"
                            )
                            
                    log.FOOTER(acnt="INIT-DONE", rslt=f"全域資料表部署完畢，共計部署 {tabl_cnt} 張資料表")

        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            err_ctx["log_idnt"] = err_ctx.get("log_idnt", "INIT_EXCP")
            err_ctx["log_msge"] = err_ctx.get("log_msge", f"[資料庫: MARS_DB] 部署失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}")
            err_ctx["rse_idnt"] = err_ctx.get("rse_idnt", "INIT-系統異常")

            log.CONTENT(
                type = "DB",
                targ = "MARS_DB",
                idnt = err_ctx["log_idnt"],
                stat = "FAIL",
                msge = err_ctx["log_msge"]
            )
            log.FOOTER(acnt="INIT-FAIL", rslt="全域資料表部署發生異常，請調閱堆疊日誌排查問題")
            raise RuntimeError(f"[FAIL] 錯誤來源 >>> [DIR] database >> [FILE] db_init.py >> [Func.C] _init_tables ({err_ctx['rse_idnt']})") from None

    # =============================================
    def _audit_tables(self):
        """
        [名稱] Func.D 全域資料表審計函式
        [功能] 掃描底層元數據 (pg_class)，盤點並列出當前系統中所有實體資料表。
        [參數] None: 無需使用外部參數
        [輸出] None: 完成現況盤點作業，並透過結構化日誌工具 (SystemLog) 動態輸出
        """
        log.BANNER(acnt="AUDT-DB", msin="全域資料表審計任務")

        try:
            schm_keys = list(TABLES.keys())

            # 構建底層元數據審計語法
            if len(schm_keys) == 1:
                schm_name = f"('{schm_keys[0]}')"
            else:
                schm_name = str(tuple(schm_keys))

            query = f"""
                SELECT
                    n.nspname AS schema_name,                   -- 元數據: schema (train, proc...)
                    c.relname AS table_name,                    -- 元數據: table_name (hist_modl_vers)
                    c.oid AS table_oid                          -- 元數據: table_oid (五碼數字)
                FROM pg_class c                                 -- pg_class 裡存放 relname (資料表名稱)
                JOIN pg_namespace n ON n.oid = c.relnamespace   -- pg_namespace 裡存放 nspname (架構)
                WHERE n.nspname IN {schm_name}                  -- 利用 pg_class 中的 relnamespace 與 pg_namespace 中的 oid 相互參照
                AND c.relkind = 'r'                             -- 指選取特定格式 (relkind): r (資料表)
                ORDER BY c.oid;                                 -- 指定排序 (oid)
            """

            with psycopg2.connect(**self.params) as conn:
                with conn.cursor() as cur:
                    info_cnt = 0   # 成功審計之資料表數量

                    # 資料表審計流程 (TABLE)
                    # 回傳格式: 位元組串列 (e.g., [('train', 'hist_modl_vers', 16405), ...])
                    cur.execute(query)
                    db_info = cur.fetchall()
                    
                    # 驗證元數據盤點結果，攔截資料表未建置之空白情境
                    if not db_info:
                        # [SITU-1] 空白情境分支，查無實體資料表，直接跳過處理
                        log.CONTENT(
                            type = "DB",
                            targ = "MARS_DB",
                            idnt = "AUDT_TABL",
                            stat = "WARN",
                            msge = "[資料庫: MARS_DB] 審計警示 >>> 系統內無任何資料表資訊"
                        )
                        log.FOOTER(acnt="AUDT-DONE", rslt="無相關資料，跳過處理")
                        return

                    else:
                        # [SITU-2] 正常情境分支，逐筆解析並紀錄資料表實體資訊
                        for index, (schema, table_name, table_oid) in enumerate(db_info, 1):
                            info_cnt += 1
                            log.CONTENT(
                                type = "DB",
                                targ = "MARS_DB",
                                idnt = "AUDT_TABL",
                                stat = "SUCC",
                                msge = f"[資料表: {table_name}] 審計成功 >>> 項次-{str(index).zfill(3)}，所屬架構({schema})、系統識別碼({table_oid})"
                            )

            log.FOOTER(acnt="AUDT-DONE", rslt=f"全域資料表審計完畢，共計審計 {info_cnt} 筆實體資料表")

        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "DB",
                targ = "MARS_DB",
                idnt = "AUDT_EXCP",
                stat = "FAIL",
                msge = f"[資料庫: MARS_DB] 審計失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            log.FOOTER(acnt="AUDT-FAIL", rslt="全域資料表審計發生異常，請調閱堆疊日誌排查問題")
            raise RuntimeError(f"[FAIL] 錯誤來源 >>> [DIR] database >> [FILE] db_init.py >> [Func.D] _audit_tables (AUDT-系統異常)") from None

# =============================================
# ⭐｜模組導出｜實例化管理對象
# =============================================
""" 使用底線標記為內部私有實例，防止外部 import * 時污染命名空間 """
_dtbs_init = DBInitializer()

""" 對外開放的正式單例接口 (公開名稱) """
# 1. dtbs_init : [實例] 資料庫架構初始化引擎 >>> 管控全域資料庫生命週期，統籌實體移除、部署與盤點作業。
# 2. WIPE_TBLS : [封裝] 全域資料表移除函式 >>> 強制清除全域資料表與底層架構，徹底淨空舊有資料實體。
# 3. INIT_TBLS : [封裝] 全域資料表部署函式 >>> 讀取標準 SQL 指令檔，自動建置全域架構與全新資料表。
# 4. AUDT_TBLS : [封裝] 全域資料表審計函式 >>> 掃描系統底層元數據，盤點並列出當前所有的實體資料表。

dtbs_init = _dtbs_init
WIPE_TBLS = _dtbs_init._wipe_tables
INIT_TBLS = _dtbs_init._init_tables
AUDT_TBLS = _dtbs_init._audit_tables