# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 模型優化檢查與統計工具 (Model Optimization Inspector Tool)
# 維護日期: 2026-09-13
# 檔案路徑: MARS_Project/src/modl_opts/modl_opts_insp.py
# ##########################################################################################

# 掛載外部依賴
import yaml
from pathlib import Path

# ==========================================================================================
def get_available_datasets(roboflow_dir):
    """
    [名稱] 標註資料集細節探測函式
    [功能] 掃描 roboflow 目錄，找出所有已具備 data.yaml 且內含訓練圖檔之領域資料集。
    [參數] roboflow_dir: [str/Path] 存放標註資料集根目錄路徑
    [輸出] list: 已存在且可供訓練之領域資料集清單 (e.g., ['VEH', 'MARA'])
    """
    roboflow_dir = Path(roboflow_dir)
    datasets = []
    if not roboflow_dir.exists():
        return datasets

    for domain_dir in roboflow_dir.iterdir():
        # 檢驗是否為資料夾格式並忽略隱藏檔
        if domain_dir.is_dir() and not domain_dir.name.startswith("."):
            data_yaml_path = domain_dir / "data.yaml"
            train_imgs_dir = domain_dir / "train" / "images"

            # 必須包含 data.yaml 且訓練集影像目錄內至少有 1 張合法圖檔
            if data_yaml_path.exists() and train_imgs_dir.exists():
                has_train_imgs = any(
                    img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
                    for img in train_imgs_dir.iterdir()
                )
                if has_train_imgs:
                    datasets.append(domain_dir.name.upper())

    return sorted(datasets)

# ==========================================================================================
def check_unprocessed_datasets_exist(roboflow_dir):
    """
    [名稱] 標註資料集狀態檢驗函式
    [功能] 檢驗 roboflow 目錄下，是否存有尚未進行模型訓練與歸檔之領域標註集。
    [參數] roboflow_dir: [str/Path] 存放標註資料集根目錄路徑
    [輸出] bool: 是否存在待訓練之標註資料集 (True/False)
    """
    datasets = get_available_datasets(roboflow_dir)
    return len(datasets) > 0

# ==========================================================================================
def get_dataset_summary(roboflow_dir):
    """
    [名稱] 標註資料集數據統計函式
    [功能] 掃描 roboflow 各領域目錄，從 data.yaml 中提取類別數量 (nc)，並統計 train/valid/test 各階段圖片數量。
    [參數] roboflow_dir: [str/Path] 存放標註資料集根目錄路徑
    [輸出] dict: 內含執行結果之結構化統整字典，共計 N 組鍵值 (依照領域數量多寡)，以下說明:
           - <DOMAIN> : [dict] 以領域類別 (e.g., VEH, MARA) 為鍵，其值為該領域之「資料集統計字典」，包含 5 組鍵值:
            (e.g., {'VEH': {'totl_imgs': 500, 'train_imgs': 400, 'valid_imgs': 50, 'test_imgs': 50, 'cls_qty': 2}})
            - totl_imgs  : [int] 該領域資料集之影像總張數
            - train_imgs : [int] 訓練集之影像張數
            - valid_imgs : [int] 驗證集之影像張數
            - test_imgs  : [int] 測試集之影像張數
            - cls_qty    : [int] 該領域支援辨識之物件類別總數
    """
    roboflow_dir = Path(roboflow_dir)
    summary = {}
    if not roboflow_dir.exists():
        return summary

    for domain_dir in roboflow_dir.iterdir():
        # 檢驗是否為資料夾格式並忽略隱藏檔
        if domain_dir.is_dir() and not domain_dir.name.startswith("."):
            domain_name = domain_dir.name.upper()
            data_yaml   = domain_dir / "data.yaml"
            if not data_yaml.exists():
                continue

            # [STEP-1] 解析 data.yaml 中介資料，從中提取 nc-類別數量
            cls_qty = 0
            try:
                with open(data_yaml, "r", encoding="utf-8") as f:
                    yaml_cfg = yaml.safe_load(f)
                    if isinstance(yaml_cfg, dict):
                        cls_qty = yaml_cfg.get("nc") or len(yaml_cfg.get("names", []))
            except Exception:
                cls_qty = 0
            
            # [STEP-2] 統計各領域標註資料集圖片數量
            counts = {}
            totl_imgs = 0
            for split in ["train", "valid", "test"]:
                img_dir = domain_dir / split / "images"
                if img_dir.exists():
                    img_cnt = sum(
                        1 for file in img_dir.iterdir()
                        if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
                    )
                else:
                    img_cnt = 0
                counts[f"{split}_imgs"] = img_cnt
                totl_imgs += img_cnt
                
            summary[domain_name] = {
                "totl_imgs"  : totl_imgs,
                "train_imgs" : counts["train_imgs"],
                "valid_imgs" : counts["valid_imgs"],
                "test_imgs"  : counts["test_imgs"],
                "cls_qty"    : cls_qty
            }
    return summary