-- ##########################################################################################
-- 專案名稱: 多重目標自動辨識系統 - 資料表實體部署指令 (Table Entity Deployment Script)
-- 維護日期: 2026-09-20
-- 檔案路徑: MARS_Project/src/database/deploy_repo/infer/veh_recg.sql
-- ##########################################################################################

-- ===== [SQL-CMD 配置說明] =====
-- [架構] infer
-- [類別] veh_recg
-- [表名] live_veh_recg
-- [屬性] 領域: 車輛-VEH｜頻率: 即時｜名稱: 車輛辨識日誌

CREATE TABLE IF NOT EXISTS {{TARGET_PATH}} (
    -- [NOTE] 核心識別類別 --
    cam_id         VARCHAR(20),    -- 攝影機定址代碼 (<CAM_BRAND> e.g., "ACEPRO", "CAM_01")
    vid_name       VARCHAR(100),   -- 來源影片檔案名稱 (e.g., "INFER_ACEPRO_VEH_ROOF_20260705_120000.mp4")
    frame_idx      INT,            -- 影片當前影格序號 (curr_frms)
    recg_ts        TIMESTAMP,      -- 辨識影格實體時間戳記 (精準對應影格時間)
    veh_id         VARCHAR(30),    -- 該影格追蹤識別碼 (e.g., "VEH_00001")

    -- [NOTE] 物件特徵類別 --
    veh_type       VARCHAR(20),    -- Stage-1 車輛種類，未辨識出為 "UNK" (e.g., "car", "truck", "bus", "motorcycle")
    plate_num      VARCHAR(15),    -- Stage-2 車牌號碼，未辨識出為 "UNK" (e.g., "ABC1234", "軍C1234")
    plate_attr     VARCHAR(20),    -- Stage-2 車牌類別，未辨識出為 "UNK" (e.g., "STANDARD", "電動車", "軍車", "試車牌")
    veh_brand      VARCHAR(30),    -- Stage-2 車輛廠牌，未辨識出為 "UNK" (e.g., "toyota", "benz")
    veh_color      VARCHAR(20),    -- 車輛外觀顏色，預設 "UNK"
    conf_score     NUMERIC(5,2),   -- 演算法綜合辨識置信度 (0.0 ~ 1.0)
    crop_path      VARCHAR(255),   -- 局部車牌拉直或車體特徵影像儲存路徑

    -- [NOTE] 系統審計類別 --
    create_ts      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- 資料建立時間戳記
    update_ts      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- 資料更新時間戳記

    PRIMARY KEY (cam_id, vid_name, frame_idx, recg_ts, veh_id)
);