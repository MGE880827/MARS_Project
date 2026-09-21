# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 車輛領域資料庫寫入工具 (Vehicle Domain Data Write Tool)
# 維護日期: 2026-09-20
# 檔案路徑: MARS_Project/src/database/write_repo/infer/veh_domain.py
# ##########################################################################################

# 掛載外部依賴
import pandas as pd

# 掛載內部依賴
from config import TABLES
from src.database import db_mngr

# ==========================================================================================
def upsert_veh_recg_logs(records):
    """
    [名稱] 車輛辨識日誌寫入函式
    [功能] 接收「多筆車輛辨識特徵」之結構化字典清單，直接向量化封裝為 DataFrame 並委由 db_mngr 批次寫入資料庫 (infer.live_veh_recg)。
    [參數] records: [list] 記錄「多筆車輛辨識特徵」之結構化字典清單
    [輸出] bool: 資料庫寫入作業是否成功 (True/False)
    """
    if not records:
        return False
    
    # 所屬資料表名稱
    table_info = TABLES["infer"]["veh_recg"]
    # 高效批次寫入並回傳 DB 寫入結果
    upsert_df = pd.DataFrame(records)
    upst_cnts = db_mngr.SQL_UPSERT(upsert_df=upsert_df, table_info=table_info)

    return bool(upst_cnts > 0)

# ==========================================================================================
def upsert_veh_monitor_records(records):
    """
    [名稱] 車輛監控清單寫入函式
    [功能] 接收「多筆車輛監控資料」之結構化字典清單（或單筆字典），直接向量化封裝為 DataFrame 並委由 db_mngr 批次寫入資料庫 (infer.live_veh_moni)。
    [參數] records: [list/dict] 記錄「多筆車輛監控資料」之結構化字典清單或單筆字典
    [輸出] bool: 資料庫寫入作業是否成功 (True/False)
    """
    if not records:
        return False

    # 若傳入的是單一字典，自動轉為串列以便統一處理
    if isinstance(records, dict):
        records = [records]

    # 所屬資料表名稱
    table_info = TABLES["infer"]["veh_moni"]

    # 高效批次寫入並回傳 DB 寫入結果
    upsert_df = pd.DataFrame(records)
    upst_cnts = db_mngr.SQL_UPSERT(upsert_df=upsert_df, table_info=table_info)

    return bool(upst_cnts > 0)