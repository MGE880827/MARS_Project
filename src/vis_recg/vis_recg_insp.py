# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 視覺辨識檢查與統計工具 (Visual Recognition Inspector Tool)
# 維護日期: 2026-09-19
# 檔案路徑: MARS_Project/src/vis_recg/vis_recg_insp.py
# ##########################################################################################

# 掛載外部依賴
from pathlib import Path

# 掛載內部依賴
from utils.env import input_resolver

# ==========================================================================================
def check_pending_videos_exist(vids_todo_dir):
    """
    [名稱] 辨識影片狀態檢驗函式
    [功能] 檢驗 vids_todo 目錄下，是否存有任何合法之待辦識影片檔案 (.mp4, .avi, .mkv)。
    [參數] vids_todo_dir: [str/Path] 存放待辨識影片之根目錄路徑 (e.g., vids_todo)
    [輸出] bool: 是否存在合法待辦識影片 (True/False)
    """
    vids_todo_dir = Path(vids_todo_dir)
    if not vids_todo_dir.exists():
        return False

    for file in vids_todo_dir.iterdir():
        # 檢驗是否為檔案格式並篩選特定副檔名
        if file.is_file() and not file.name.startswith(".") and file.suffix.lower() in [".mp4", ".avi", ".mkv"]:
            return True
    return False

# ==========================================================================================
def get_pending_video_summary(vids_todo_dir):
    """
    [名稱] 辨識影片數據統計函式
    [功能] 掃描 vids_todo 目錄，統計待辨識影片數量及各領域分佈比例。
    [參數] vids_todo_dir: [str/Path] 存放待辨識影片之根目錄路徑 (e.g., vids_todo)
    [輸出] dict: 內含執行結果之結構化統整字典，共計 2 組鍵值，以下說明:
           - totl_vids  : [int] 待辨識影片之總數量
           - domain_cnt : [dict] 各目標領域之影片數量統計字典 (e.g., {'VEH': 3, 'MARA': 2})
    """
    vids_todo_dir = Path(vids_todo_dir)
    summary = {
        "totl_vids"  : 0,
        "domain_cnt" : {}
    }
    if not vids_todo_dir.exists():
        return summary

    for vid_file in vids_todo_dir.iterdir():
        # 檢驗是否為檔案格式並篩選特定副檔名
        if vid_file.is_file() and not vid_file.name.startswith(".") and vid_file.suffix.lower() in [".mp4", ".avi", ".mkv"]:
            summary["totl_vids"] += 1
            meta = input_resolver.analyze_video_filename(vid_file)
            domain = meta.get("domain", "UNK")
            summary["domain_cnt"][domain] = summary["domain_cnt"].get(domain, 0) + 1

    return summary

# ==========================================================================================
def get_available_pending_domains(vids_todo_dir):
    """
    [名稱] 待辨識影片領域細節探測函式
    [功能] 掃描 vids_todo 目錄，解析所有合規之待辨識影片，提取並回傳所有可用的領域代碼清單。
    [參數] vids_todo_dir: [str/Path] 存放待辨識影片之根目錄路徑 (e.g., vids_todo)
    [輸出] list: 已存在且可供辨識之領域代碼清單 (e.g., ['VEH', 'MARA'])
    """
    vids_todo_dir = Path(vids_todo_dir)
    domains = set()
    if not vids_todo_dir.exists():
        return sorted(list(domains))

    for vid_path in vids_todo_dir.iterdir():
        # 檢驗是否為檔案格式並忽略隱藏檔
        if vid_path.is_file() and not vid_path.name.startswith("."):
            if vid_path.suffix.lower() in [".mp4", ".avi", ".mkv"]:
                meta = input_resolver.analyze_video_filename(vid_path)
                # 必須通過命名規範檢驗
                if meta.get("is_valid", False):
                    domains.add(meta["domain"].upper())

    return sorted(list(domains))