-- ##########################################################################################
-- 專案名稱: 多重目標自動辨識系統 - 資料表實體部署指令 (Table Entity Deployment Script)
-- 維護日期: 2026-09-17
-- 檔案路徑: MARS_Project/src/database/deploy_repo/infer/veh_moni.sql
-- ##########################################################################################

-- ===== [SQL-CMD 配置說明] =====
-- [架構] infer
-- [類別] veh_moni
-- [表名] live_veh_moni
-- [屬性] 領域: 車輛-VEH｜頻率: 即時｜名稱: 車輛監控日誌

CREATE TABLE IF NOT EXISTS {{TARGET_PATH}} (
    -- [NOTE] 核心識別類別 --
    plate_num      VARCHAR(15),   -- 監控目標車牌號碼

    -- [NOTE] 管制標籤類別 --
    reason_tag     VARCHAR(50),   -- 車輛管制標籤 (e.g., "Stln", "Tax", "Susp")
    warn_level     VARCHAR(10),   -- 告警嚴重等級 ("High", "Medi", "Low")
    is_active      SMALLINT,      -- 監控啟用開關 (1: 啟用, 0: 停用)
    memo_text      TEXT,          -- 查緝備註與案件描述說明

    -- [NOTE] 系統審計類別 --
    create_ts      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- 資料建立時間戳記
    update_ts      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- 資料更新時間戳記

    PRIMARY KEY (plate_num)
);
