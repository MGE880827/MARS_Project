# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 環境快取移除工具 (Environment Cache Removal Tool)
# 維護日期: 2026-08-24
# 檔案路徑: MARS_Project/utils/env/cache_remover.py
# ##########################################################################################

# 掛載外部依賴
import sys
import shutil
import traceback

# 掛載內部依賴
from config.path_cfg import PROJECT_ROOT
from .input_resolver import to_root_relative   # [NOTE] 同層資料夾，必須使用相對路徑
from utils import log

# ==========================================================================================
def remove_pycache():
    """
    [名稱] Python 編譯快取移除函式
    [功能] 強制遞迴檢索並移除專案下所有 __pycache__ 快取檔案。
    [參數] None: 無需使用外部參數
    [輸出] int: 成功刪除之快取總數量
    """
    # 禁止 Python 於執行期間生成新的位元組碼快取
    sys.dont_write_bytecode = True

    log.BANNER(acnt="REMV-PYC", msin="PyCache 快取檔案移除任務")
    project_name = PROJECT_ROOT.name
    rmv_cnt = 0   # 成功移除之檔案數量

    for sgl_pyc_path in PROJECT_ROOT.rglob("__pycache__"):
        # 驗證目標路徑結構，攔截隸屬虛擬環境 (venv) 之暫存目錄
        if "venv" in sgl_pyc_path.parts:
            continue
        try:
            sgl_pyc_name = sgl_pyc_path.name
            pyc_rel_path = to_root_relative(sgl_pyc_path)
            shutil.rmtree(sgl_pyc_path)
            log.CONTENT(
                type = "FILE",
                targ = "PyCache",
                idnt = "REMV_PYC",
                stat = "SUCC",
                msge = f"[檔案: {sgl_pyc_name}] 移除成功 >>> 該檔案路徑({pyc_rel_path})"
            )
            rmv_cnt += 1
                
        except Exception:
            # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
            sys_err = traceback.format_exc()
            log.CONTENT(
                type = "FILE",
                targ = "PyCache",
                idnt = "REMV_EXCP",
                stat = "FAIL",
                msge = f"[檔案: {sgl_pyc_name}] 移除失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
            )
            continue

    rmv_msge = f"共計移除 {rmv_cnt} 個快取檔案" if rmv_cnt > 0 else "環境中無 PYCACHE 快取檔案，無須移除"
    log.FOOTER(acnt="REMV-DONE", rslt=rmv_msge)
    return rmv_cnt