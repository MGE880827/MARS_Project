# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 系統程序啟動中心 (System Execution Entry Point)
# 維護日期: 2026-09-21
# 檔案路徑: MARS_Project/main.py
# ##########################################################################################

# [LOAD] 掛載外部依賴
import sys
import argparse
import traceback

# [LOAD] 掛載內部依賴
from utils import log
from utils.env import cache_remover, input_resolver
from src.database import WIPE_TBLS, INIT_TBLS, AUDT_TBLS
from train_pipe import TrainPipeline
from infer_pipe import InferPipeline

# ==========================================================================================
def initialize_system(wipe_db=False):
    """
    [名稱] Func.A 系統與資料庫初始化函式
    [功能] 啟動系統環境與資料庫組態之初始化調度，依序執行「PyCache 暫存清理」、「全域資料表移除」、「全域資料表建置」與「狀態盤點審計」。
    [參數] wipe_db: [bool] 是否強制清除全域舊有資料庫表實體 (預設 False)
    [輸出] None: 執行順利則產出初始化完成橫幅；若遇異常將印出終止橫幅，並將例外層級上拋交由核心入口接管。
    """
    try:
        log.BANNER(acnt="INIT-ENVR", msin="系統環境與資料庫初始化作業")
        log.SPLIT()
        log.IND_ON()

        # [EXEC] 暫存檔案移除階段
        cache_remover.remove_pycache()
        log.SPLIT()

        # [EXEC] 全域資料表移除階段
        if wipe_db:
            WIPE_TBLS()
            log.SPLIT()

        # [EXEC] 全域資料表建置階段
        INIT_TBLS() 
        log.SPLIT()

        # [EXEC] 全域資料表審計階段
        AUDT_TBLS()
        log.SPLIT()

        log.IND_OFF()
        log.FOOTER(acnt="INIT-DONE", rslt="已完成系統與資料庫初始化")
        log.SPLIT()

    except Exception:
        # 印出任務結束橫幅，同時接收來自底層的異常錯誤訊息，並統一交由 main.py「核心執行入口」處理
        log.IND_OFF()
        log.FOOTER(acnt="INIT-FAIL", rslt="系統初始化異常中斷流程，請查修")
        log.SPLIT()
        raise

# ============================================================
# ⭐｜核心進入｜主控制流程 (Main Control Flow)
# ============================================================
def main():
    """
    [名稱] Func.B 核心主控制流程函式
    [功能] 統籌全域系統生命週期，負責底層環境構建、參數解析與 MARS 視覺辨識/訓練管線之循序調度。
    [參數] None: 透過 argparse 接收外部終端機指令。
    [輸出] None: 執行結果將透過各模組之日誌工具輸出至 Console 或資料庫。
    """

    """ [STAGE-1] 命令提示字元定義與解析 """
    parser = argparse.ArgumentParser(description="MARS 多重目標自動辨識系統 - 系統核心啟動中心")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "infer", "init"],
        help="[系統模式] 類型 train(訓練管線), infer(辨識管線), init(僅初始化資料庫)"
    )
    parser.add_argument(
        "--wipe", action="store_true",
        help="[特殊操作] 強制清空並重建全域資料庫 (WIPE_TBLS)"
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        help="[執行階段] 訓練模式支援 prepare/train/all，辨識模式支援 infer"
    )
    parser.add_argument(
        "--ver_id", type=str, nargs="*", default=[],
        help="[模型版本] 格式 DOMAIN:VER_ID (e.g., --ver_id VEH:v1.0.1 MARA:v1.0.0)"
    )
    parser.add_argument(
        "--epochs", type=str, nargs="*", default=[],
        help="[訓練階段] 總訓練輪次數；格式 DOMAIN:EPOCHS (e.g., --epochs VEH:100 MARA:50)"
    )
    parser.add_argument(
        "--lr", type=str, nargs="*", default=[],
        help="[訓練階段] 初始學習率；格式 DOMAIN:LR (e.g., --lr VEH:0.01 MARA:0.02)"
    )
    parser.add_argument(
        "--no_trt", action="store_true",
        help="[辨識階段] 停用 TensorRT 靜態加速引擎，改採 PyTorch 原生引擎"
    )
    args = parser.parse_args()


    """ [STAGE-2] 系統環境建置與參數轉型 """
    # [EXEC] 系統環境與資料庫建置
    initialize_system(wipe_db=args.wipe)
    if args.mode == "init":
        # 若模式僅為初始化，執行完畢即安全退出
        sys.exit(0)

    # [EXEC] 解析 CLI 參數字典
    ver_id_dict     = input_resolver.parse_kv_pair(args.ver_id, val_type=str)
    epoch_qty_dict  = input_resolver.parse_kv_pair(args.epochs, val_type=int)
    learn_rate_dict = input_resolver.parse_kv_pair(args.lr, val_type=float)
    use_trt         = not args.no_trt


    """ [STAGE-3] 視覺管線調度執行 """
    if args.mode == "train":
        if args.stage not in ["prepare", "train", "all"]:
            print(f"\n [系統提示] 訓練模式下 --stage 必須為 prepare, train 或 all", flush=True)
            sys.exit(1)
            
        # [EXEC] 啟動自動化模型訓練管線
        pipeline = TrainPipeline()
        report = pipeline(
            exec_stage      = args.stage,
            ver_id_dict     = ver_id_dict,
            epoch_qty_dict  = epoch_qty_dict,
            learn_rate_dict = learn_rate_dict
        )
        sys.exit(0 if report.get("stat_code") in [200, 206] else 1)

    elif args.mode == "infer":
        if args.stage not in ["infer"]:
            print(f"\n [系統提示] 辨識模式下 --stage 必須為 infer", flush=True)
            sys.exit(1)

        # [EXEC] 啟動自動化影像辨識管線
        pipeline = InferPipeline()
        report = pipeline(
            exec_stage  = args.stage,
            ver_id_dict = ver_id_dict,
            use_trt     = use_trt
        )
        sys.exit(0 if report.get("stat_code") in [200, 206] else 1)

# ============================================================
# ⭐｜執行入口｜程序啟動與異常監控 (Main Entry Point)
# ============================================================
if __name__ == "__main__":
    """
    [名稱] 系統執行入口
    [功能] 啟動主程式執行入口，並配置全域異常攔截與使用者強制中斷機制。
    [參數] None: 無需使用外部參數。
    [輸出] None: 程序結束後向系統層回傳狀態碼 (0 為正常結束, 1 為異常崩潰)。
    """
    try:
        # [EXEC] 啟動 MARS 多重目標自動辨識系統
        main()

    except KeyboardInterrupt:
        # [STOP] 攔截終端機實體中斷指令 (Ctrl+C)，執行安全退出
        print(f"\n [系統提示] 接收到使用者強制中斷指令，程序安全結束 (Ctrl + C)", flush=True)
        sys.exit(0)

    except Exception as e:
        # [FAIL] 捕捉全域異常訊息，藉由異常中斷觸發 Raise 機制
        print(f"\n [系統崩潰] 發生全域未預期異常: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)