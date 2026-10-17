# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 系統程序啟動中心 (System Execution Entry Point)
# 維護日期: 2026-10-14
# 檔案路徑: MARS_Project/main.py
# ##########################################################################################

# [LOAD] 掛載外部依賴
import os
# 1. 強制關閉 Paddle 新版 PIR 中間表示執行器
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
# 2. 強制關閉 oneDNN / MKLDNN CPU 向量指令加速，改走原生穩定通道
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_onednn_verbose"] = "0"
# 3. 關閉 Paddle 底層 Google Logging 警告（消除控制台雜訊）
os.environ["GLOG_minloglevel"] = "2"
import sys
import argparse
import traceback

# [LOAD] 掛載內部依賴
from utils import log
from utils.env import cache_remover, input_resolver
from src.database import db_init
from train_pipe import TrainPipeline
from infer_pipe import InferPipeline

# ==========================================================================================
def init_database(wipe_db=False):
    """
    [名稱] Func.A 資料庫初始化函式
    [功能] 啟動全域資料庫組態之初始化調度，依序執行「全域資料表移除 (選用)」、「全域資料表建置」與「資料表狀態審計」。
    [參數] wipe_db: [bool] 是否強制清除全域舊有資料庫表實體 (預設 False)
    [輸出] None: 完成資料庫初始化作業；若遇異常則將例外上拋，交由主程式進入點統一接管。
    """
    try:
        log.BANNER(acnt="INIT-DB", msin="資料庫初始化作業")
        log.IND_ON()
        
        # [STEP-1] 全域資料表移除
        if wipe_db:
            db_init.WIPE_TBLS()
        # [STEP-2] 全域資料表建置
        db_init.INIT_TBLS() 
        # [STEP-3] 全域資料表審計
        db_init.AUDT_TBLS()

        log.IND_OFF()
        log.FOOTER(acnt="INIT-DONE", rslt="資料庫初始化完成")

    except Exception:
        # 紀錄異常日誌並中斷，將錯誤向上拋出交由 main 核心執行入口處理
        log.IND_OFF()
        log.FOOTER(acnt="INIT-FAIL", rslt="資料庫初始化發生異常，請調閱堆疊資訊排查")
        raise

# ============================================================
# ⭐｜核心進入｜主控制流程 (Main Control Flow)
# ============================================================
def main():
    """
    [名稱] Func.B 核心主控制流程函式
    [功能] 統籌系統生命週期，負責 CLI 參數解析、圖形化介面 (GUI) 啟動與視覺辨識/訓練管線之排程調度。
    [參數] None : 透過 argparse 接收命令列輸入之指令與配置參數。
    [輸出] int  : 回傳系統執行狀態碼 (0 代表執行成功，非 0 代表異常終止)，供主進入點判定退出行為。
    """
    """ [STAGE-1] 命令提示字元定義與解析 """
    parser = argparse.ArgumentParser(description="MARS 多重目標自動辨識系統 - 系統核心啟動中心")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["init", "gui", "train", "infer"],
        help="[Mode] 運行模式:\n"
             "  init  - 資料庫建置與狀態審計\n"
             "  gui   - 啟動圖形化操作介面\n"
             "  train - 啟動模型訓練管線\n"
             "  infer - 啟動即時辨識管線"
    )
    parser.add_argument(
        "--wipe", action="store_true",
        help="[Database] 強制清空資料表並重新建置 (預設: 僅建置與盤點審計)"
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        help="[Pipeline] 執行階段:\n"
             "  train 模式 - 支援 prepare (資料整備) / train (模型優化) / all (完整流程，預設)\n"
             "  infer 模式 - 支援 infer (即時辨識，預設)"
    )
    parser.add_argument(
        "--ver_id", type=str, nargs="*", default=[],
        help="[Pipeline] 模型版本；格式: DOMAIN:VER_ID\n"
             "(e.g., --ver_id VEH:v1.0.1 MARA:v1.0.0，辨識模式留空將自動偵測最新版)"
    )
    parser.add_argument(
        "--epochs", type=str, nargs="*", default=[],
        help="[Train] 訓練輪次；格式: DOMAIN:EPOCHS\n"
             "(e.g., --epochs VEH:100 MARA:50)"
    )
    parser.add_argument(
        "--lr", type=str, nargs="*", default=[],
        help="[Train] 初始學習速率；格式: DOMAIN:EPOCHS\n"
             "(e.g., --epochs VEH:100 MARA:50)"
    )
    parser.add_argument(
        "--no_trt", action="store_true",
        help="[Infer] 停用 TensorRT (.engine) 靜態編譯，降級使用原生 PyTorch (.pt)"
    )
    args = parser.parse_args()

    """ [MODE-1] GUI 圖形化操作介面 """
    if args.mode == "gui":
        from src.gui import run_app
        sys.exit(run_app())

    """ [MODE-2] CLI 指令式操作介面 """
    if args.mode == "init":
        # 系統環境與資料庫建置
        init_database(wipe_db=args.wipe)
        sys.exit(0)

    # 解析 CLI 參數字典
    ver_id_dict     = input_resolver.parse_kv_pair(args.ver_id, val_type=str)
    epoch_qty_dict  = input_resolver.parse_kv_pair(args.epochs, val_type=int)
    learn_rate_dict = input_resolver.parse_kv_pair(args.lr, val_type=float)
    use_trt         = not args.no_trt

    if args.mode == "train":
        # 模型訓練管線模式 (Train)
        if args.stage not in ["prepare", "train", "all"]:
            print(f"\n [系統提示] 訓練模式下 --stage 必須為 prepare, train 或 all", flush=True)
            sys.exit(1)
        pipeline = TrainPipeline()
        report = pipeline(
            exec_stage      = args.stage,
            ver_id_dict     = ver_id_dict,
            epoch_qty_dict  = epoch_qty_dict,
            learn_rate_dict = learn_rate_dict
        )
        sys.exit(0 if report.get("stat_code") in [200, 206] else 1)

    elif args.mode == "infer":
        # 即時辨識管線模式 (Infer)
        if args.stage not in ["infer"]:
            print(f"\n [系統提示] 辨識模式下 --stage 必須為 infer", flush=True)
            sys.exit(1)
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
    [輸出] None: 程式生命週期結束後向系統層回傳狀態碼 (0 代表正常結束，1 代表異常崩潰)。
    """
    try:
        # 啟動 MARS 多重目標自動辨識系統
        main()

    except KeyboardInterrupt:
        # 攔截終端機中斷指令 (Ctrl+C)，執行安全退出
        print(f"\n [系統提示] 接收到使用者強制中斷指令，程序安全結束 (Ctrl + C)", flush=True)
        sys.exit(0)

    except Exception as e:
        # 攔截全域未預期之異常訊息，藉由異常中斷觸發 Raise 機制
        print(f"\n [系統崩潰] 發生未預期異常錯誤: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)

    finally:
        try:
            # 程式結束前，一律清除 PyCache 檔案
            cache_remover.remove_pycache()

        except Exception:
            pass