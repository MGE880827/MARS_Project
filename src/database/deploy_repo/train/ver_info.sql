-- ##########################################################################################
-- 專案名稱: 多重目標自動辨識系統 - 資料表實體部署指令 (Table Entity Deployment Script)
-- 維護日期: 2026-09-16
-- 檔案路徑: MARS_Project/src/database/deploy_repo/train/ver_info.sql
-- ##########################################################################################

-- ===== [SQL-CMD 配置說明] =====
-- [架構] train
-- [類別] ver_info
-- [表名] modl_ver_info
-- [屬性] 領域: 模型訓練｜頻率: 單次｜名稱: 模型版本資訊

CREATE TABLE IF NOT EXISTS {{TARGET_PATH}} (
    -- [NOTE] 核心識別類別 --
    main_domain     VARCHAR(20),    -- 主領域識別代碼 (e.g., "VEH")          ※ 次階層時恆保留所屬主領域
    main_ver_id     VARCHAR(20),    -- 主領域版本代碼 (e.g., "v1.0.0")       ※ 次階層時恆保留所屬主版本
    sub_domain      VARCHAR(20),    -- 次領域識別代碼 (e.g., "LIC", "BRAND") ※ Stage-1 主階層恆寫入 "NONE"
    sub_ver_id      VARCHAR(20),    -- 次領域版本代碼 (e.g., "v1.1.0")       ※ Stage-1 主階層恆寫入 "NONE"

    -- [NOTE] 階層幾何類別 --
    stage_level     SMALLINT,       -- 當前模型階層 (1: Stage-1 宏觀檢測, 2: Stage-2 微觀檢測)
    bbox_type       VARCHAR(10),    -- 邊界框類型 ("HBB": 水平框, "OBB": 旋轉框)
    train_date      VARCHAR(20),    -- 訓練完成日期戳記 (YYYY-MM-DD)

    -- [NOTE] 演算法組態類別 --
    algo_name       VARCHAR(50),    -- 演算法架構 (e.g., "MARS-YOLO11")
    bone_name       VARCHAR(50),    -- 基礎預訓練骨幹權重 (e.g., "yolo11n.pt", "yolo11n-obb.pt")
    wght_path       VARCHAR(500),   -- 最佳權重實體儲存路徑 (weights/<DOMAIN>/.../best.pt)
    class_qty       SMALLINT,       -- 該領域物件類別數量
    
    -- [NOTE] 訓練參數類別 --
    img_size        SMALLINT,       -- 輸入影像之解析度 (e.g., 640, 480)
    batch_size      SMALLINT,       -- 訓練批次大小
    accumulate      SMALLINT,       -- 梯度累積步數
    epoch_qty       SMALLINT,       -- 總訓練輪次數
    learn_rate      NUMERIC(7,6),   -- 最佳化初始學習率

    -- [NOTE] 系統審計類別 --
    create_ts       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- 資料建立時間戳記
    update_ts       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- 資料更新時間戳記
    
    PRIMARY KEY (main_domain, main_ver_id, sub_domain, sub_ver_id)
);