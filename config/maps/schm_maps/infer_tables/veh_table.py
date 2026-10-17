# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 車輛領域之資料表屬性定義 (Vehicle Domain Table Attributes Definition)
# 維護日期: 2026-09-20
# 檔案路徑: MARS_Project/config/maps/schm_maps/infer_tables/veh_table.py
# ##########################################################################################

"""
[名稱] 車輛領域之資料表屬性字典 (Vehicle Domain Table Attributes Dictionary)
[作用] 負責集中定義車輛領域 (VEH) 之即時辨識與監控日誌資料表實體名稱、寫入模式與核心欄位結構。
"""
# ==========================================================================================
# < 字典配置說明表 >
# 1. schm : [str]  資料表所屬綱要架構 (train, infer)
# 2. tabl : [str]  資料表實體名稱
# 3. mode : [dict] 欄位衝突更新模式 (如 ON CONFLICT DO UPDATE 之覆寫策略)
# 4. cols : [list] 資料表欄位順序 (嚴格對應 SQL 寫入與 DDL 欄位定義)
# 5. ukey : [list] 資料表唯一主鍵或複合唯一鍵清單
# ==========================================================================================

INFER_SCHEMA_INFO = {
    # =============================================
    # ｜INFER｜TABL-001｜車輛辨識日誌
    # =============================================
    "veh_recg" : {
        "schm" : "infer",
        "tabl" : "live_veh_recg",
        "mode" : {},
        "cols" : [
            "cam_id",      # 攝影機定址代碼 (<CAM_BRAND> e.g., "ACEPRO", "CAM_01")
            "vid_name",    # 來源影片檔案名稱 (e.g., "INFER_ACEPRO_VEH_ROOF_20260705_120000.mp4")
            "frame_idx",   # 影片當前影格序號 (curr_frms)
            "recg_ts",     # 辨識影格實體時間戳記 (精準對應影格時間)
            "veh_id",      # 該影格追蹤識別碼 (e.g., "VEH_00001")
            "veh_type",    # Stage-1 車輛種類，未辨識出為 "UNK" (e.g., "car", "truck", "bus", "motorcycle")
            "plate_num",   # Stage-2 車牌號碼，未辨識出為 "UNK" (e.g., "ABC1234", "軍C1234")
            "plate_attr",  # Stage-2 車牌類別，未辨識出為 "UNK" (e.g., "STANDARD", "電動車", "軍車", "試車牌")
            "veh_brand",   # Stage-2 車輛廠牌，未辨識出為 "UNK" (e.g., "toyota", "benz")
            "veh_color",   # 車輛外觀顏色，預設 "UNK"
            "conf_score",  # 演算法綜合辨識置信度 (0.0 ~ 1.0)
            "crop_path"    # 局部車牌拉直或車體特徵影像儲存路徑
        ],
        "ukey" : ["cam_id", "vid_name", "frame_idx", "recg_ts", "veh_id"]
    },
    # =============================================
    # ｜INFER｜TABL-002｜車輛監控日誌
    # =============================================
    "veh_moni" : {
        "schm" : "infer",
        "tabl" : "live_veh_moni",
        "mode" : {},
        "cols" : [
            "plate_num",   # 監控目標車牌號碼
            "reason_tag",  # 車輛管制標籤 (e.g., "Stln", "Tax", "Susp")
            "warn_level",  # 告警嚴重等級 ("High", "Medi", "Low")
            "is_active",   # 監控啟用開關 (1: 啟用, 0: 停用)
            "memo_text"    # 查緝備註與案件描述說明
        ],
        "ukey" : ["plate_num"]
    }
}