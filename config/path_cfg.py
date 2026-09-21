# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 實體路徑全域組態配置 (Global Path Configuration)
# 維護日期: 2026-09-11
# 檔案路徑: MARS_Project/config/path_cfg.py
# ##########################################################################################

# 掛載外部依賴
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # 動態取得專案根目錄

# ==========================================================================================
@dataclass
class TrainPathConfig:
    """ 
    [名稱] 訓練階段路徑組態 (Training Phase Path Config) 
    [作用] 定義模型訓練階段之任務佇列、原始影片、抽幀影像、標註資料與歷史歸檔之實體根目錄。
    """
    base_dir   : Path = PROJECT_ROOT / "data" / "01_train"   # [綜整區] 模型訓練資料區
    task_queue : Path = base_dir / "task_queue"              # [待辦區] 待處理批次任務佇列
    raw_vids   : Path = base_dir / "raw_vids"                # [待辦區] 待抽幀之模型訓練影片
    frames     : Path = base_dir / "frames"                  # [待辦區] 待標註之時序抽幀全景影像
    roboflow   : Path = base_dir / "roboflow"                # [待辦區] 物件標註資料集
    archive    : Path = base_dir / "archive"                 # [歸檔區] 訓練資料歸檔集

# ==========================================================================================
@dataclass
class InferPathConfig:
    """ 
    [名稱] 辨識階段路徑組態 (Inference Phase Path Config) 
    [作用] 定義正式辨識環境中待辨識影片與歸檔歷史影片之實體根目錄。
    """
    base_dir  : Path = PROJECT_ROOT / "data" / "02_infer"    # [綜整區] 模型辨識資料區
    vids_todo : Path = base_dir / "vids_todo"                # [待辦區] 待處理之模型辨識影片
    vids_done : Path = base_dir / "vids_done"                # [歸檔區] 辨識資料歸檔集

# ==========================================================================================
@dataclass
class PathConfig:
    """
    [名稱] 實體路徑組態管理引擎 (Path Configuration Management Engine)
    [作用] 集中管理專案根路徑、模型權重、日誌目錄與訓練/辨識資料夾映射，利用 pathlib 確保跨平台相容性。
    """
    root    : Path = PROJECT_ROOT                                        # 專案實體絕對根目錄
    weights : Path = PROJECT_ROOT / "weights"                            # [權重區] 各領域專屬模型權重 (跨階段共用)
    logs    : Path = PROJECT_ROOT / "logs"                               # [日誌區] 系統日誌紀錄根目錄 (跨階段共用)
    train   : TrainPathConfig = field(default_factory=TrainPathConfig)   # 訓練階段資料路徑集合
    infer   : InferPathConfig = field(default_factory=InferPathConfig)   # 辨識階段資料路徑集合

# =============================================
# ⭐｜模組導出｜實例化管理對象
# =============================================
""" 使用底線標記為內部私有實例，防止外部 import * 時污染命名空間 """
_path_config = PathConfig()

""" 對外開放的正式單例接口 (公開名稱) """
# 1. path_config : [實例] 實體路徑組態管理引擎 >>> 負責全域資料夾與檔案實體路徑之集中管理與靜態調用。

path_config = _path_config