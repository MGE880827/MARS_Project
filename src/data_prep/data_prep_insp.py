# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 資料整備檢查與統計工具 (Data Preparation Inspector Tool)
# 維護日期: 2026-09-19
# 檔案路徑: MARS_Project/src/data_prep/data_prep_insp.py
# ##########################################################################################

# 掛載外部依賴
from pathlib import Path

# 掛載內部依賴
from utils.env import input_resolver

# ==========================================================================================
def check_raw_video_exist(raw_vids_dir):
    """
    [名稱] 訓練影片狀態檢驗函式
    [功能] 檢驗 raw_vids 目錄下，是否存在待抽幀之影片檔案 (.mp4, .avi, .mkv)。
    [參數] raw_vids_dir: [str/Path] 存放待抽幀模型訓練影片之根目錄路徑
    [輸出] bool: 是否存在待抽幀影片 (True/False)
    """
    raw_vids_dir = Path(raw_vids_dir)
    if not raw_vids_dir.exists():
        return False

    for file in raw_vids_dir.iterdir():
        # 檢驗是否為檔案格式並篩選特定副檔名
        if file.is_file() and file.suffix.lower() in [".mp4", ".avi", ".mkv"]:
            return True
    return False

# ==========================================================================================
def get_raw_video_summary(raw_vids_dir):
    """
    [名稱] 訓練影片數據統計函式
    [功能] 掃描 raw_vids 目錄，統計待抽幀訓練影片數量及各領域分佈比例。
    [參數] raw_vids_dir: [str/Path] 存放待抽幀模型訓練影片之根目錄路徑 (e.g., raw_vids)
    [輸出] dict: 內含執行結果之結構化統整字典，共計 2 組鍵值，以下說明:
           - totl_vids  : [int] 待抽幀之影片總數量
           - domain_cnt : [dict] 各目標領域之影片數量統計字典 (e.g., {'VEH': 3, 'MARA': 2})
    """
    raw_vids_dir = Path(raw_vids_dir)
    summary = {
        "totl_vids"  : 0,
        "domain_cnt" : {}
    }
    if not raw_vids_dir.exists():
        return summary
    
    for vid_file in raw_vids_dir.iterdir():
        # 檢驗是否為檔案格式並篩選特定副檔名
        if vid_file.is_file() and vid_file.suffix.lower() in [".mp4", ".avi", ".mkv"]:
            summary["totl_vids"] += 1
            meta = input_resolver.analyze_video_filename(vid_file)
            domain = meta["domain"]
            summary["domain_cnt"][domain] = summary["domain_cnt"].get(domain, 0) + 1

    return summary

# ==========================================================================================
def get_unarchived_frames_detail(frames_dir):
    """
    [名稱] 未歸檔抽幀細節探測函式
    [功能] 掃描 frames 目錄，檢驗各領域是否存在尚未進行訓練與歸檔之抽幀影像資料夾 (<VID_STEM>)。
    [參數] frames_dir: [str/Path] 存放待標註之時序抽幀全景影像根目錄路徑
    [輸出] dict: 內含執行結果之結構化字典，共計 N 組鍵值 (依照領域數量多寡)，以下說明:
           - <DOMAIN> : [list] 以領域識別代碼 (e.g., VEH, MARA) 為鍵，其值為該領域下未歸檔之「抽幀影像資料夾」(VID_STEM) 之字串清單
             (e.g., {'VEH': ['VID_STEM_01', 'VID_STEM_02']})
    """
    frames_dir = Path(frames_dir)
    unarchived_detail = {}
    if not frames_dir.exists():
        return unarchived_detail

    for domain_dir in frames_dir.iterdir():
        # 檢驗是否為資料夾格式並忽略隱藏檔
        if domain_dir.is_dir() and not domain_dir.name.startswith("."):
            domain_name = domain_dir.name.upper()
            pending_stems = []
            for vid_stem_dir in domain_dir.iterdir():
                # 檢驗是否為資料夾格式並忽略隱藏檔
                if vid_stem_dir.is_dir() and not vid_stem_dir.name.startswith("."):
                    # 驗證目錄內是否至少含有一張有效圖片格式 (.jpg, .jpeg, .png)
                    has_imgs = any(
                        img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
                        for img in vid_stem_dir.iterdir()
                    )
                    if has_imgs:
                        pending_stems.append(vid_stem_dir.name)
            if pending_stems:
                unarchived_detail[domain_name] = pending_stems

    return unarchived_detail

# ==========================================================================================
def check_unarchived_frames_exist(frames_dir):
    """
    [名稱] 未歸檔抽幀狀態檢驗函式
    [功能] 檢驗 frames 目錄下，是否存在任何「抽幀影像資料夾」(VID_STEM) 尚未歸檔。
    [參數] frames_dir: [str/Path] 存放待標註之時序抽幀全景影像根目錄路徑
    [輸出] bool: 是否存在未歸檔狀態 (True/False)
    """
    unarchived_detail = get_unarchived_frames_detail(frames_dir)
    return len(unarchived_detail) > 0