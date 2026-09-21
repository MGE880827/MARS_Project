# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 車輛領域階層策略配置 (Vehicle Domain Stage Configuration)
# 維護日期: 2026-09-22
# 檔案路徑: MARS_Project/config/maps/stage_maps/rules/veh_rule.py
# ##########################################################################################

"""
[名稱] 車輛領域階層策略字典 (Vehicle Domain Stage Dictionary)
[作用] 定義車輛領域 (VEH) 之策略字典：
       - TRAIN: 定義雙階層訓練策略，涵蓋主階車體檢測、邊界安全裁切、次階車牌/廠牌坐標投影與標籤平移重組，以及各階模型訓練規格。
       - INFER: 定義雙階層辨識策略，涵蓋主階車體檢測、邊界安全裁切、次階車牌/廠牌定位、透視幾何校正與 OCR 字元讀取。
"""

# ==========================================================================================
# < TRAIN 字典配置說明表 >
# 1. stage_qty      : [int]   模型階層數量 (1: 單階層全圖訓練, 2: 二階層串聯裁切投影與微觀訓練)
# 2. stage_desc     : [str]   該領域訓練架構任務描述
# 3. stage_1        : [dict]  第一階層 (主領域) 訓練規格配置
#    - task_name    : [str]   任務名稱識別碼
#    - domain       : [list]  主領域識別代碼清單 (e.g., ["VEH"])
#    - bbox_type    : [str]   邊界框類型 ("HBB": 水平框, "OBB": 旋轉框)
#    - gt_cls_ids   : [list]  全景標註檔中代表主階層之物件類別編號清單
#    - cls_names    : [list]  物件類別名稱清單
#    - imgsz        : [int]   訓練輸入影像尺度 (寬高像素)
#    - bone_name    : [str]   基礎預訓練骨幹權重 (e.g., yolo11n.pt)
# 4. transition     : [dict]  階層級聯銜接參數 (僅於 stage_qty >= 2 時生效；若為單階層則為 {})
#    - crop_margin  : [float] 邊界裁切外擴比例 (e.g., 0.05 代表外擴 5%，保障邊界次目標完整度)
# 5. stage_2        : [dict]  第二階層 (次領域) 訓練規格配置 (若為單階層則為 {})
#    - task_name    : [str]   任務名稱識別碼
#    - domain       : [list]  次領域識別代碼清單 (e.g., ["LIC", "BRAND"])
#    - bbox_type    : [str]   邊界框類型 ("HBB": 水平框, "OBB": 旋轉框)
#    - sub_tasks    : [dict]  次領域獨立任務配置 (以次領域代碼如 LIC, BRAND 為鍵)
#      - gt_cls_ids : [list]  全景標註檔中代表次階層之物件類別專屬編號分段清單 (LIC: 100~199, BRAND: 200~299)
#      - cls_names  : [list]  物件類別名稱清單 (轉出次階段 data.yaml 時將類別編號平移歸零)
#      - imgsz      : [int]   訓練輸入影像尺度 (寬高像素)
#      - bone_name  : [str]   基礎預訓練骨幹權重 (e.g., yolo11n-obb.pt)
# ==========================================================================================

TRAIN = {
    # =============================================
    # ｜TRAIN｜雙層架構｜VEH-車輛領域
    # =============================================
    "stage_qty"  : 2,
    "stage_desc" : "[MAIN] VEH-車輛, [SUB] LIC-車牌/BRAND-廠牌 (雙階層模型訓練架構)",
    "stage_1"    : {
        "task_name"  : "VEHICLE_DETECTION",
        "domain"     : "VEH",
        "bbox_type"  : "HBB",
        "gt_cls_ids" : [0, 1, 2, 3],
        "cls_names"  : [
            "car",                         # 0 汽車
            "truck",                       # 1 卡車
            "bus",                         # 2 公車
            "motorcycle"                   # 3 摩托車
        ],
        "imgsz"      : 640,
        "bone_name"  : "yolo11n.pt"
    },
    "transition" : {
        "crop_margin": 0.05
    },
    "stage_2"    : {
        "task_name"  : "SUB_OBJECT_DETECTION",
        "sub_tasks"  : {
            # [次領域-A] LIC-車牌 (單一類別檢測；區間-ID: 100~199)
            "LIC"   : {
                "domain"     : "LIC",
                "bbox_type"  : "OBB",
                "gt_cls_ids" : [100],
                "cls_names"  : ["license_plate"],
                "imgsz"      : 480,
                "bone_name"  : "yolo11n-obb.pt"
            },
            # [次領域-B] BRAND-廠牌 (多個類別檢測；區間-ID: 200~299)
            "BRAND" : {
                "domain"     : "BRAND",
                "bbox_type"  : "HBB",
                "gt_cls_ids" : [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212],
                "cls_names"  : [
                    "toyota",              # 200 豐田 (台灣市佔龍頭)
                    "honda",               # 201 本田 (國產休旅/房車主力)
                    "nissan",              # 202 日產 (國產跨界/轎車主力)
                    "mitsubishi",          # 203 三菱 (商用車/休旅車主力)
                    "ford",                # 204 福特 (歐系底盤國產主力)
                    "hyundai",             # 205 現代 (韓系熱銷休旅/商旅)
                    "mazda",               # 206 馬自達 (日系進口美型主力)
                    "volkswagen",          # 207 福斯 (歐系進口國民車)
                    "lexus",               # 208 凌志 (日系豪華進口常勝軍)
                    "benz",                # 209 賓士 (德系頂級豪華霸主)
                    "bmw",                 # 210 BMW (德系豪華性能霸主)
                    "tesla",               # 211 特斯拉 (純電動車能見度冠軍)
                    "luxgen"               # 212 納智捷 (台灣自主品牌休旅/純電車主力)
                ],
                "imgsz"      : 480,
                "bone_name"  : "yolo11n.pt"
            } 
        }
    }
}

# ==========================================================================================
# < INFER 字典配置說明表 >
# 1. stage_qty       : [int]   模型階層數量 (1: 單階層全圖辨識, 2: 二階層串聯辨識)
# 2. stage_desc      : [str]   該領域辨識架構任務描述
# 3. stage_1         : [dict]  第一階層 (主領域) 辨識規格配置
#    - task_name     : [str]   任務名稱識別碼
#    - domain        : [str]   主領域識別代碼 (e.g., "VEH")
#    - bbox_type     : [str]   邊界框類型 ("HBB": 水平框, "OBB": 旋轉框)
#    - conf_thresh   : [float] 主物件偵測信心度門檻 (range: 0.0 ~ 1.0)
#    - nms_thresh    : [float] 主物件非極大值抑制 IoU 閾值
#    - cls_mapping   : [dict]  類別索引與英文名稱對照表 (全景標註索引)
# 4. transition      : [dict]  階層級聯銜接參數 (僅於 stage_qty >= 2 時生效；若為單階層則為 {})
#    - crop_margin   : [float] 邊界裁切外擴比例 (e.g., 0.05 代表外擴 5%，保障邊界次目標完整度)
#    - min_crop_w    : [int]   最小有效裁切寬度 (像素)，低於此門檻視為噪點予以過濾
#    - min_crop_h    : [int]   最小有效裁切高度 (像素)
# 5. stage_2         : [dict]  第二階層 (次領域) 辨識規格配置 (若為單階層則為 {})
#    - task_name     : [str]   任務名稱識別碼
#    - sub_tasks     : [dict]  次領域獨立任務配置 (以次領域代碼如 LIC, BRAND 為鍵)
#      - domain      : [str]   次領域識別代碼 (e.g., "LIC", "BRAND")
#      - bbox_type   : [str]   邊界框類型 ("HBB": 水平框, "OBB": 旋轉框)
#      - conf_thresh : [float] 次物件偵測信心度門檻 (range: 0.0 ~ 1.0)
#      - nms_thresh  : [float] 次物件非極大值抑制 IoU 閾值
#      - img_tools   : [list]  該次領域啟用之後處理工具鏈清單 (內嵌專屬參數，如 geom_warp, char_ocr)
#      - cls_mapping : [dict]  類別索引與英文名稱對照表 (轉出平移歸零後之識別標籤)
# ==========================================================================================

INFER = {
    # =============================================
    # ｜INFER｜雙階架構｜VEH-車輛領域
    # =============================================
    "stage_qty"  : 2,
    "stage_desc" : "[MAIN] VEH-車輛, [SUB] LIC-車牌/BRAND-廠牌 (雙階層模型辨識架構)",
    "stage_1"    : {
        "task_name"   : "VEHICLE_DETECTION",
        "domain"      : "VEH",
        "bbox_type"   : "HBB",
        "conf_thresh" : 0.001,   # 暫時調低，讓低輪次模型順利觸發辨識 (0.45)
        "nms_thresh"  : 0.45,
        "img_tools"   : [
            {"color_extr"  : {}}
        ],
        "cls_mapping" : {
            0 : "car",                     # 0 汽車
            1 : "truck",                   # 1 卡車
            2 : "bus",                     # 2 公車
            3 : "motorcycle"               # 3 摩托車
        }
    },
    "transition" : {
        "crop_margin" : 0.05,              # 邊界裁切向外延伸 5%，確保緊貼邊緣之車牌完整無裁切損毀
        "min_crop_w"  : 32,                # 最小有效裁切寬度 (像素)，低於此門檻視為噪點予以過濾
        "min_crop_h"  : 32                 # 最小有效裁切高度 (像素)
    },
    "stage_2"    : {
        "task_name"   : "SUB_FEATURE_EXTRACTION",
        "sub_tasks"   : {
            # [次領域-A] LIC-車牌 (車牌微觀定位與文字辨識)
            "LIC"   : {
                "domain"      : "LIC",
                "bbox_type"   : "OBB",
                "conf_thresh" : 0.40,
                "nms_thresh"  : 0.45,
                "img_tools"   : [
                    {"geom_warp" : {"min_warp_w" : 20, "min_warp_h" : 10}},   # 車牌拉平後最小寬高度
                    {"char_ocr"  : {"lang": "chinese_cht"}}
                ],
                "cls_mapping" : {
                    0  : "license_plate"   # 100 車牌
                }
            },
            # [次領域-B] BRAND-廠牌 (汽車廠牌定位與分類辨識；可選擴充)
            "BRAND" : {
                "domain"      : "BRAND",
                "bbox_type"   : "HBB",
                "conf_thresh" : 0.35,
                "nms_thresh"  : 0.45,
                "img_tools"   : [],        # 純分類辨識，無須額外影像工具
                "cls_mapping" : {
                    0  : "toyota",         # 200 豐田 (台灣市佔龍頭)
                    1  : "honda",          # 201 本田 (國產休旅/房車主力)
                    2  : "nissan",         # 202 日產 (國產跨界/轎車主力)
                    3  : "mitsubishi",     # 203 三菱 (商用車/休旅車主力)
                    4  : "ford",           # 204 福特 (歐系底盤國產主力)
                    5  : "hyundai",        # 205 現代 (韓系熱銷休旅/商旅)
                    6  : "mazda",          # 206 馬自達 (日系進口美型主力)
                    7  : "volkswagen",     # 207 福斯 (歐系進口國民車)
                    8  : "lexus",          # 208 凌志 (日系豪華進口常勝軍)
                    9  : "benz",           # 209 賓士 (德系頂級豪華霸主)
                    10 : "bmw",            # 210 BMW (德系豪華性能霸主)
                    11 : "tesla",          # 211 特斯拉 (純電動車能見度冠軍)
                    12 : "luxgen"          # 212 納智捷 (台灣自主品牌休旅/純電車主力)
                }
            }
        }
    }
}