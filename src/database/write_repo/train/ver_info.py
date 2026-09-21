# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 模型版本資訊同步寫入工具 (Model Version Info Sync Write Tool)
# 維護日期: 2026-09-16
# 檔案路徑: MARS_Project/src/database/write_repo/train/ver_info.py
# ##########################################################################################

# 掛載外部依賴
import pandas as pd

# 掛載內部依賴
from config import TABLES
from src.database import db_mngr

# ==========================================================================================
def upsert_version_info(record):
    """
    [名稱] 模型版本資訊寫入函式
    [功能] 將本次訓練之「核心版本資訊」字典，直接向量化封裝為 DataFrame 並寫入資料庫 (train.modl_ver_info)。
    [參數] record: [dict] 包含核心識別、階層幾何、演算法組態與訓練參數之「核心版本資訊」字典
    [輸出] bool: 資料庫寫入作業是否成功 (True/False)
    """
    if not record:
        return False

    # 所屬資料表名稱
    tbls_info = TABLES["train"]["ver_info"]
    
    # 高效批次寫入並回傳 DB 寫入結果
    upsert_df = pd.DataFrame([record])
    upst_cnts = db_mngr.SQL_UPSERT(upsert_df=upsert_df, table_info=tbls_info)

    return bool(upst_cnts > 0)