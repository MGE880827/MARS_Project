# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 外部輸入解析工具 (External Input Resolver Tool)
# 維護日期: 2026-09-17
# 檔案路徑: MARS_Project/utils/env/input_resolver.py
# ##########################################################################################

# 掛載外部依賴
from pathlib import Path

# 掛載內部依賴
from config.path_cfg import PROJECT_ROOT

# ==========================================================================================
def analyze_video_filename(src_vid_path):
    """
    [名稱] 單一影片檔名分析函式
    [功能] 解析 MARS 專案之標準影片檔案名稱，精確提取特徵中介資料 (Metadata)。
    [參數] src_vid_path: [str/Path] 單一來源影片檔案之路徑
    [輸出] dict: 內含階段、廠牌、領域、角度與時間戳等檔名特徵之結構化解析字典，共計 7 組鍵值，以下說明:
           - phase     : [str] 運行階段 (e.g., TRAIN/INFER)
           - cam_brand : [str] 相機廠牌 (e.g., ACEPRO)
           - domain    : [str] 目標領域 (e.g., VEH/MARA)
           - angle     : [str] 拍攝角度 (e.g., ROOF/SIDE)
           - cap_date  : [str] 拍攝日期 (e.g., YYYYMMDD)
           - cap_time  : [str] 拍攝時間 (e.g., HHMMSS)
           - is_valid  : [bool] 檢驗格式 (e.g., True/False)
    """
    # 擷取純檔案名稱並切分欄位特徵
    file_path = Path(src_vid_path)
    file_stem = file_path.stem
    parts = file_stem.split("_")

    # 建立預設中介資料字典，防範因非標準檔名導致系統中斷
    metadata = {
        "phase"     : "UNK",   # 運行階段
        "cam_brand" : "UNK",   # 相機廠牌
        "domain"    : "UNK",   # 目標領域
        "angle"     : "UNK",   # 拍攝角度
        "cap_date"  : "UNK",   # 拍攝日期
        "cap_time"  : "UNK",   # 拍攝時間
        "is_valid"  : False    # 檢驗格式
    }

    # 驗證是否符合標準的六段式命名結構
    if len(parts) != 6:
        return metadata
    
    # 驗證階段格式 (僅有 TRAIN/INFER)
    if parts[0].upper() not in ["TRAIN", "INFER"]:
        return metadata 

    # 驗證日期 (8碼數字) 與時間 (6碼數字) 格式
    if not (parts[4].isdigit() and len(parts[4]) == 8):
        return metadata
    if not (parts[5].isdigit() and len(parts[5]) == 6):
        return metadata
    
    # 提取出的各欄位特徵中介資料
    metadata["phase"]     = parts[0].upper()
    metadata["cam_brand"] = parts[1].upper()
    metadata["domain"]    = parts[2].upper()
    metadata["angle"]     = parts[3].upper()
    metadata["cap_date"]  = parts[4]
    metadata["cap_time"]  = parts[5]
    metadata["is_valid"]  = True
    
    return metadata

# ==========================================================================================
def to_root_relative(target_path, include_root=True):
    """
    [名稱] 專案相對路徑解析函式
    [功能] 將傳入之「實體絕對路徑」進行自動解析、防呆並裁切為優雅的專案相對路徑。
    [參數] 共計 2 組參數，以下說明:
           - target_path  : [str/Path] 待解析之實體絕對路徑
           - include_root : [bool] 是否在路徑最前面保留「專案根目錄」資料夾名稱 (e.g., MARS_Project)
    [輸出] str: 格式化後之路徑字串，統一將 Windows 反斜線 '\\' 轉為工業標準正斜線 '/' 以利閱讀
    """
    if not target_path:
        return ""
    
    try:
        proc_path = Path(target_path).resolve()

        try:
            # 嘗試計算相對路徑（相容所有 Python 3.x 版本）
            pure_path = proc_path.relative_to(PROJECT_ROOT)
            if include_root:
                display_path = Path(PROJECT_ROOT.name) / pure_path
            else:
                display_path = pure_path

            # 統一使用 Linux 路徑斜線規範方向
            return display_path.as_posix()

        except ValueError:
            # 備援防線: 當路徑在根目錄外，僅傳回檔名以防洩漏主機隱私
            return proc_path.name
    
    except Exception:
        # 當路徑解析過程中發生未知異常，僅傳回檔名以確保流程持續運作
        try:
            return Path(target_path).name
        except Exception:
            return str(target_path)

# ==========================================================================================
def parse_kv_pair(kv_list, val_type=None):
    """
    [名稱] CLI 鍵值對標準解析函式
    [功能] 將 CLI 傳入之 'KEY:VALUE' 清單格式 (e.g., ['VEH:v1.0.1', 'MARA:v1.0.0']) 解析為字典格式。
    [參數] 共計 2 組參數，以下說明:
           - kv_list  : [list/str] CLI 傳入之字串或清單
           - val_type : [type/None] 強制指定數值型態 (e.g., int, float, str)；若未指定則維持原字串
    [輸出] dict: 內含解析結果之結構化字典，共計 N 組鍵值 (依照傳入鍵值對數量多寡)，以下說明:
           - <KEY> : [val_type/str] 以領域類別 (e.g., "VEH", "MARA") 為鍵，其值為該領域之「訓練組態參數與版本代號數值」
             (e.g., {'VEH': 'v1.0.1', 'MARA': 'v1.0.0'} 或 {'VEH': 100, 'MARA': 50})
    """
    kv_rslv = {}
    if not kv_list:
        return kv_rslv

    # 自動轉換為清單，防止單一字串傳入
    if isinstance(kv_list, str):
        kv_list = [kv_list]
    
    for pair in kv_list:
        pair_str = str(pair).strip()
        if ":" in pair_str:
            # CLI 鍵值拆解流程
            key, value = pair_str.split(":", 1)
            key   = key.strip().upper()
            value = value.strip()
            if val_type is not None:
                try:
                    # 指定數值型態
                    value = val_type(value)
                except (ValueError, TypeError):
                    pass
        kv_rslv[key] = value
    return kv_rslv
    