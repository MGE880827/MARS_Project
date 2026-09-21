# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 演算法全域組態配置 (Global Algorithm Configuration)
# 維護日期: 2026-09-17
# 檔案路徑: MARS_Project/config/algo_cfg.py
# ##########################################################################################

# 掛載外部依賴
from dataclasses import dataclass, field

# ==========================================================================================
@dataclass
class StreamConfig:
    """ 
    [名稱] 影片串流與抽幀組態 (Video Stream & Extraction Config) 
    [作用] 定義全域階段 (訓練/辨識) 之視角抽幀間距與辨識環境之系統預期最高處理幀率。
    """
    # [全域階段] 領域與視角抽幀間距對照表 (單位: 幀)
    intervals: dict = field(default_factory=lambda: {
        "VEH"  : {"ROOF": 10, "SIDE": 25},   # [車輛領域]   高速行駛: 俯視(每10幀1張), 側視(每25幀1張)
        "MARA" : {"ROOF": 15, "SIDE": 30}    # [馬拉松領域] 移動稍慢: 俯視(每15幀1張), 側視(每30幀1張)
    })

    # [辨識階段] 系統運算規範表
    sys_max_fps : int = 60                 # 運算上限: 限制辨識環境之每秒最高處理幀率(fps)，防止硬體資源超載

# ==========================================================================================
@dataclass
class ClaheConfig:
    """
    [名稱] 光線正規化組態 (Light Normalization Config) 
    [作用] 定義影像前處理之對比度限制與網格分塊大小，精準抑制過曝並提亮暗部，強化夜間與強光環境下之目標輪廓特徵。
    """
    clip_limit : float = 2.0      # 對比限制門檻: 數值越高明暗對比越強烈，過高易產生影像噪點
    grid_size  : tuple = (8, 8)   # 局部切塊大小: 將影像分割為 8x8 網格，獨立執行局部光線最佳化

# ==========================================================================================
@dataclass
class YoloTrainConfig:
    """
    [名稱] YOLO 訓練階段組態 (YOLO Train Phase Config)
    [作用] 定義全域計算與訓練基準參數，各領域專屬策略由 train_stage.py 動態覆寫。
    """
    proj_name          : str = "MARS"             # 專案系統識別標籤: 用於組合模型演算法架構代碼
    arch_name          : str = "YOLO11"           # 神經網路底層架構: 指定特徵提取之模型家族
    default_hbb_bone   : str = "yolo11n.pt"       # 全域預設水平框 (HBB) 基礎預訓練骨幹權重
    default_obb_bone   : str = "yolo11n-obb.pt"   # 全域預設旋轉框 (OBB) 基礎預訓練骨幹權重

    default_main_imgsz : int = 640                # 主階層 (Stage-1) 預設全景輸入影像尺寸 (像素寬高)
    default_sub_imgsz  : int = 480                # 次階層 (Stage-2) 預設微觀裁切輸入影像尺寸 (像素寬高)
    
    batch_size         : int = 4                  # 訓練批次大小: 限制單次饋入 GPU 之影像張數
    nbs                : int = 16                 # 名目批次大小: 目標等效 Batch，配合 batch_size 自動折算梯度累積步數 (nbs / batch_size = 4 步)，在顯存限制下模擬 Batch=16 的收斂精度
    epoch_qty          : int = 100                # 全域預設總訓練輪次數 (各領域可獨立覆寫)
    learn_rate         : float = 0.01             # 最佳化初始學習率: 控制特徵權重修正力道大小

# ==========================================================================================
@dataclass
class YoloInferConfig:
    """ 
    [名稱] YOLO 辨識階段組態 (YOLO Inference Phase Config) 
    [作用] 定義物件偵測之全域基準信心度與 NMS 門檻，未指定領域自動回退此配置。
    """
    conf_thresh : float = 0.45   # 全域預設通用信心度閥值
    nms_thresh  : float = 0.45   # 非極大值抑制 IoU 閾值

    # 各領域與次任務專屬信心度閥值，支援主階層與次階層獨立設定
    confidences : dict = field(default_factory=lambda: {
        "VEH"   : 0.45,   # 主階層-車輛辨識
        "LIC"   : 0.40,   # 次階層-車牌辨識
        "BRAND" : 0.35,   # 次階層-廠牌辨識
    })

# ==========================================================================================
@dataclass
class AlgorithmConfig:
    """
    [名稱] 演算法組態管理引擎 (Algorithm Configuration Management Engine)
    [作用] 集中管控「時序抽幀」、「影像預處理」、「模型訓練底層基準」以及「物件偵測底層基準」之核心基石。
    """
    stream : StreamConfig    = field(default_factory=StreamConfig)
    clahe  : ClaheConfig     = field(default_factory=ClaheConfig)
    yolo_t : YoloTrainConfig = field(default_factory=YoloTrainConfig)
    yolo_i : YoloInferConfig = field(default_factory=YoloInferConfig)

# =============================================
# ⭐｜模組導出｜實例化管理對象
# =============================================
""" 使用底線標記為內部私有實例，防止外部 import * 時污染命名空間 """
_algo_config = AlgorithmConfig()

""" 對外開放的正式單例接口 (公開名稱) """
# 1. algo_config : [實例] 演算法組態管理引擎 >>> 負責全域演算法核心參數之集中管理與靜態調用。

algo_config = _algo_config