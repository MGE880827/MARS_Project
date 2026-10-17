-- ##########################################################################################
-- 專案名稱: 多重目標自動辨識系統 - 資料表實體部署指令 (Table Entity Deployment Script)
-- 維護日期: 2026-09-16
-- 檔案路徑: MARS_Project/src/database/deploy_repo/train/fit_eval.sql
-- ##########################################################################################

-- ===== [SQL-CMD 配置說明] =====
-- [架構] train
-- [類別] fit_eval
-- [表名] modl_fit_eval
-- [屬性] 領域: 模型訓練｜頻率: 每輪｜名稱: 模型成效指標

CREATE TABLE IF NOT EXISTS {{TARGET_PATH}} (
    -- [NOTE] 核心識別類別 --
    main_domain     VARCHAR(20),    -- 主領域識別代碼 (e.g., "VEH")          ※ 次階層時恆保留所屬主領域
    main_ver_id     VARCHAR(20),    -- 主領域版本代碼 (e.g., "v1.0.0")       ※ 次階層時恆保留所屬主版本
    sub_domain      VARCHAR(20),    -- 次領域識別代碼 (e.g., "LIC", "BRAND") ※ Stage-1 主階層恆寫入 "NONE"
    sub_ver_id      VARCHAR(20),    -- 次領域版本代碼 (e.g., "v1.1.0")       ※ Stage-1 主階層恆寫入 "NONE"
    epoch_idx       SMALLINT,       -- 當前訓練輪數序號 (1 ~ epoch_qty)

    -- [NOTE] 損失收斂類別 --
    train_loss      NUMERIC(8,5),   -- 訓練損失收斂值 (Box Loss)
    val_loss        NUMERIC(8,5),   -- 驗證損失收斂值 (Box Loss)

    -- [NOTE] 評估指標類別 --
    map50           NUMERIC(6,5),   -- 邊界框平均精準度 (mAP@0.5)
    map95           NUMERIC(6,5),   -- 邊界框平均精準度 (mAP@0.5:0.95)
    prec_val        NUMERIC(6,5),   -- 模型精準率 (Precision)
    recall_val      NUMERIC(6,5),   -- 模型召回率 (Recall)

    -- [NOTE] 系統審計類別 --
    create_ts       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- 資料建立時間戳記
    update_ts       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   -- 資料更新時間戳記

    PRIMARY KEY (main_domain, main_ver_id, sub_domain, sub_ver_id, epoch_idx)
);