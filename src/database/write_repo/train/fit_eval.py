# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 模型成效指標同步寫入工具 (Model Fit Evaluation Sync Write Tool)
# 維護日期: 2026-09-21
# 檔案路徑: MARS_Project/src/database/write_repo/train/fit_eval.py
# ##########################################################################################

# 掛載外部依賴
import pandas as pd

# 掛載內部依賴
from config import TABLES
from src.database import db_mngr

# ==========================================================================================
def upsert_fit_evaluation(records):
    """
    [名稱] 模型成效指標寫入函式
    [功能] 接收「各輪次效能評估指標」之結構化字典清單，直接向量化封裝為 DataFrame 並委由 db_mngr 批次寫入資料庫 (train.modl_fit_eval)。
    [參數] records: [list] 記錄「各輪次效能評估指標」之結構化字典清單
    [輸出] bool: 資料庫寫入作業是否成功 (True/False)
    """
    if not records:
        return False
    
    # 所屬資料表名稱
    table_info = TABLES["train"]["fit_eval"]
    # 高效批次寫入並回傳 DB 寫入結果
    upsert_df = pd.DataFrame(records)
    upst_cnts = db_mngr.SQL_UPSERT(upsert_df=upsert_df, table_info=table_info)

    return bool(upst_cnts > 0)