# ##########################################################################################
# 專案名稱: 多重目標自動辨識系統 - 實體檔案管理工具 (Physical File Manager Tool)
# 維護日期: 2026-09-20
# 檔案路徑: MARS_Project/utils/env/file_manager.py
# ##########################################################################################

# 掛載外部依賴
import re
import shutil
import traceback
from pathlib import Path

# 掛載內部依賴
from .input_resolver import analyze_video_filename   # [NOTE] 同層資料夾，必須使用相對路徑
from config import path_config
from utils import log

# 共用常數定義: 任務佇列資料夾名稱之正規表達式規則 (e.g., task_001, task_012)
TASK_DIR_REGEX = re.compile(r"^task_\d+$", re.IGNORECASE)

# ==========================================================================================
def _resolve_task_queue_route(upload_domains, active_vids_dir, task_queue_dir):
    """
    [名稱] 待抽幀任務佇列路由解析函式
    [功能] 動態探測 raw_vids 及 task_queue 目錄下其檔案與佇列狀態，智慧分流並解析當次上傳影片之最終寫入目錄。
    [參數] 共計 3 組參數，以下說明:
           - upload_domains  : [list/set] 當次待上傳影片之領域代碼清單 (e.g., ['VEH', 'MARA'])
           - active_vids_dir : [str/Path] 存放原始訓練影片之根目錄路徑 (e.g., raw_vids)
           - task_queue_dir  : [str/Path] 存放任務佇列資料之根目錄路徑 (e.g., task_queue)
    [輸出] dict: 內含執行結果之結構化字典，共計 N 組鍵值 (依照領域數量多寡)，以下說明:
           - <DOMAIN> : [Path] 以領域類別 (如 VEH, MARA) 為鍵，其值為該領域最終存放之目錄路徑
             (e.g., {'VEH': Path('data/01_train/raw_vids'), 'MARA': Path('data/01_train/task_queue/task_001')})
    """
    active_vids_dir = Path(active_vids_dir)
    task_queue_dir  = Path(task_queue_dir)
    route_map = {}   # 儲存領域對應路徑之字典 { 'VEH': Path(...), 'MARA': Path(...) }

    if not upload_domains:
        return route_map

    """ [STAGE-1] 既有環境狀態掃描與模擬沙盒初始化階段 """
    # [STEP-1] 檢驗當前待辦區 (raw_vids/) 既有之影片領域分佈
    raw_vid_files = [
        raw_vid for raw_vid in active_vids_dir.iterdir()
        if raw_vid.is_file() and raw_vid.suffix.lower() in [".mp4", ".avi", ".mkv"]
    ] if active_vids_dir.exists() else []

    raw_domains = set()
    for raw_vid in raw_vid_files:
        meta = analyze_video_filename(raw_vid)
        if meta.get("is_valid"):
            raw_domains.add(meta.get("domain"))

    # [STEP-2] 掃描 task_queue/ 目錄下既有之任務資料夾 (task_001, task_002...)
    existing_tasks = sorted(
        [task_dir for task_dir in task_queue_dir.iterdir() if task_dir.is_dir() and TASK_DIR_REGEX.match(task_dir.name.lower())],
        key=lambda x: x.name.lower()
    ) if task_queue_dir.exists() else []

    # [STEP-3] 初始化既有佇列之動態領域對照表 (以 task_dir Path 為 Key)，建構同批次模擬分配沙盒
    simulated_task_domains = {}
    for task_dir in existing_tasks:
        task_vid_files = [
            task_vid for task_vid in task_dir.iterdir()
            if task_vid.is_file() and task_vid.suffix.lower() in [".mp4", ".avi", ".mkv"]
        ]
        task_domains = set()
        for task_vid in task_vid_files:
            meta = analyze_video_filename(task_vid)
            if meta.get("is_valid"):
                task_domains.add(meta.get("domain"))
        simulated_task_domains[task_dir] = task_domains

    simulated_raw_domains = set(raw_domains)

    """ [STAGE-2] 待上傳領域動態路由決策與智慧分流階段 """
    for domain in upload_domains:
        # 逐一檢驗每個待上傳領域，為其尋找最佳無衝突寫入目錄
        storage_dir = None

        # [STEP-1] 優先檢查「任務序列區」(task_queue) 有無領域衝突，再檢查「原始影片區」(raw_vids) 有無領域衝突
        has_task_conflict = any(domain in simulated_task_domains[task_dir] for task_dir in existing_tasks)
        if not has_task_conflict and domain not in simulated_raw_domains:
            # 若兩者均未衝突，則放置「原始影片區」(raw_vids) 優先訓練
            storage_dir = active_vids_dir
            simulated_raw_domains.add(domain)

        # [STEP-2] 若任一區域發生衝突，全域掃描「任務序列區」(task_queue) 並實施依序遞補
        if storage_dir is None:
            last_conflict_idx = -1
            for idx, task_dir in enumerate(existing_tasks):
                if domain in simulated_task_domains[task_dir]:
                    last_conflict_idx = idx

            if last_conflict_idx != -1:
                # 若「任務序列區」(task_queue) 存在相同領域，則放置於該領域最後一個任務序列之後
                target_idx = last_conflict_idx + 1
            else:
                # 若「任務序列區」(task_queue) 未存在相同領域，衝突源自 raw_vids，則放置於此序列中第一個 (task_001)
                target_idx = 0

            if target_idx >= len(existing_tasks):
                # 若目標索引超出目前既有任務序列範圍，則自動建立新的 task_XXX 資料夾
                task_id = len(existing_tasks) + 1
                new_task_dir = task_queue_dir / f"task_{task_id:03d}"
                existing_tasks.append(new_task_dir)
                simulated_task_domains.setdefault(new_task_dir, set()).add(domain)
                storage_dir = new_task_dir

            else:
                # 若目標索引在既有任務序列範圍內，直接指向該任務資料夾並在模擬沙盒中累加登記
                storage_dir = existing_tasks[target_idx]
                simulated_task_domains.setdefault(storage_dir, set()).add(domain)
        
        route_map[domain] = storage_dir

    return route_map

# ==========================================================================================
def upload_train_videos(src_vid_paths, active_vids_dir, task_queue_dir):
    """
    [名稱] 訓練階段影片檢驗上傳函式
    [功能] 上傳訓練影片前，先實施中介資料與檔名格式驗證，過濾不合法檔案後，按領域智慧分流引進至待辦區或佇列目錄。
    [參數] 共計 3 組參數，以下說明:
           - src_vid_paths   : [list[str/Path]] 待上傳影片檔案之路徑清單
           - active_vids_dir : [str/Path] 存放原始訓練影片之根目錄路徑 (e.g., raw_vids)
           - task_queue_dir  : [str/Path] 存放任務佇列資料之根目錄路徑 (e.g., task_queue)
    [輸出] int: 成功通過檢驗並引進之訓練影片數量
    """
    succ_cnt = 0   # 成功引進之影片數量

    try:
        # [STEP-1] 驗證檔案清單之存在性，攔截空清單或未選擇檔案之情境
        if not src_vid_paths:
            log.CONTENT(
                type = "FILE",
                targ = "UploadFiles",
                idnt = "UPLOAD_TRAIN_FILE",
                stat = "WARN",
                msge = "[任務: 訓練用影片檔案] 上傳警示 >>> 未接收到任何待上傳之影片路徑，跳過影片引進作業"
            )
            return 0

        # [STEP-2] 實施影片檔名格式與中介資料剖析檢驗，並按領域進行歸類
        domain_to_vids_map = {}
        for sgl_vid_path in src_vid_paths:
            sgl_vid_path = Path(sgl_vid_path)
            sgl_vid_name = sgl_vid_path.name

            # 檢驗是否為實體檔案並確認其存在性
            if not sgl_vid_path.exists() or not sgl_vid_path.is_file():
                log.CONTENT(
                    type = "FILE",
                    targ = sgl_vid_name,
                    idnt = "UPLOAD_TRAIN_FILE",
                    stat = "WARN",
                    msge = f"[檔案: {sgl_vid_name}] 上傳警示 >>> 實體檔案不存在或非標準檔案，拒絕引進"
                )
                continue

            # 檢驗訓練影片檔名格式規範
            meta = analyze_video_filename(sgl_vid_path)
            if not meta.get("is_valid", False):
                log.CONTENT(
                    type = "FILE",
                    targ = sgl_vid_name,
                    idnt = "UPLOAD_TRAIN_FILE",
                    stat = "WARN",
                    msge = f"[檔案: {sgl_vid_name}] 上傳警示 >>> 檔案名稱不符訓練命名規範，拒絕引進"
                )
                continue

            domain = meta.get("domain", "UNK")
            domain_to_vids_map.setdefault(domain, []).append(sgl_vid_path)

        if not domain_to_vids_map:
            log.CONTENT(
                type = "FILE",
                targ = "UploadFiles",
                idnt = "UPLOAD_TRAIN_FILE",
                stat = "WARN",
                msge = "[任務: 訓練用影片檔案] 上傳警示 >>> 候選影片均未通過命名規範檢驗，終止上傳"
            )
            return 0

        # [STEP-3] 計算各領域影片的目標寫入路徑 (route_map)
        upload_domains = list(domain_to_vids_map.keys())
        route_map = _resolve_task_queue_route(
            upload_domains  = upload_domains,
            active_vids_dir = active_vids_dir,
            task_queue_dir  = task_queue_dir
        )

        # [STEP-4] 依分流結果將影片檔案複製至指定目錄
        for domain, vid_paths in domain_to_vids_map.items():
            storage_dir = route_map[domain]
            storage_dir.mkdir(parents=True, exist_ok=True)
            for src_vid_path in vid_paths:
                src_vid_name = src_vid_path.name
                dst_vid_path = storage_dir / src_vid_name
                shutil.copy(str(src_vid_path), str(dst_vid_path))
                succ_cnt += 1

            log.CONTENT(
                type = "DIR",
                targ = storage_dir.name,
                idnt = "UPLOAD_TRAIN_FILE",
                stat = "SUCC",
                msge = f"[領域: {domain}] 上傳成功 >>> 共計引進 {len(vid_paths)} 部影片"
            )

        return succ_cnt

    except Exception:
        # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
        sys_err = traceback.format_exc()
        log.CONTENT(
            type = "FILE",
            targ = "UploadFiles",
            idnt = "UPLOAD_TRAIN_EXCEPTION",
            stat = "FAIL",
            msge = f"[任務: 訓練用影片檔案] 上傳失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
        )
        return succ_cnt

# ==========================================================================================
def upload_infer_videos(src_vid_paths, target_vids_dir):
    """
    [名稱] 辨識階段影片檢驗上傳函式
    [功能] 上傳辨識影片前，先實施中介資料與檔名格式驗證，過濾不合法檔案後，引進至生產待辦區。
    [參數] 共計 2 組參數，以下說明:
           - src_vid_paths   : [list[str/Path]] 待上傳影片檔案之路徑清單
           - target_vids_dir : [str/Path] 辨識待辦區目錄絕對路徑 (e.g., vids_todo)
    [輸出] int: 成功通過檢驗並引進之辨識影片數量
    """
    succ_cnt = 0   # 成功引進之影片數量
    target_vids_dir = Path(target_vids_dir)
    target_vids_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 驗證檔案清單之存在性，攔截空清單或未選擇檔案之情境
        if not src_vid_paths:
            log.CONTENT(
                type = "FILE",
                targ = "UploadFiles",
                idnt = "UPLOAD_INFER_FILE",
                stat = "WARN",
                msge = "[任務: 辨識用影片檔案] 上傳警示 >>> 未接收到任何待上傳之影片路徑，跳過影片引進作業"
            )
            return 0

        # [STEP-2] 實施影片檔名格式與中介資料剖析檢驗
        for sgl_vid_path in src_vid_paths:
            sgl_vid_path = Path(sgl_vid_path)
            sgl_vid_name = sgl_vid_path.name

            # 檢驗是否為實體檔案並確認其存在性
            if not sgl_vid_path.exists() or not sgl_vid_path.is_file():
                log.CONTENT(
                    type = "FILE",
                    targ = sgl_vid_name,
                    idnt = "UPLOAD_INFER_FILE",
                    stat = "WARN",
                    msge = f"[檔案: {sgl_vid_name}] 上傳警示 >>> 實體檔案不存在或非標準檔案，拒絕引進"
                )
                continue

            # 檢驗辨識影片檔名格式規範
            meta = analyze_video_filename(sgl_vid_path)
            if not meta.get("is_valid", False):
                log.CONTENT(
                    type = "FILE",
                    targ = sgl_vid_name,
                    idnt = "UPLOAD_INFER_FILE",
                    stat = "WARN",
                    msge = f"[檔案: {sgl_vid_name}] 上傳警示 >>> 檔案名稱不符辨識命名規範，拒絕引進"
                )
                continue             

            # 將上傳之辨識影片存放至指定目錄
            dst_vid_path = target_vids_dir / sgl_vid_name
            shutil.copy(str(sgl_vid_path), str(dst_vid_path))
            succ_cnt += 1

        if succ_cnt == 0:
            log.CONTENT(
                type = "FILE",
                targ = "UploadFiles",
                idnt = "UPLOAD_INFER_FILE",
                stat = "WARN",
                msge = "[任務: 辨識用影片檔案] 上傳警示 >>> 候選影片均未通過命名規範檢驗，終止上傳"
            )
            return 0

        log.CONTENT(
            type = "DIR",
            targ = target_vids_dir.name,
            idnt = "UPLOAD_INFER_FILE",
            stat = "SUCC",
            msge = f"[任務: 辨識用影片檔案] 上傳成功 >>> 共計引進 {succ_cnt} 部影片"
        )

        return succ_cnt

    except Exception:
        # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
        sys_err = traceback.format_exc()
        log.CONTENT(
            type = "FILE",
            targ = "UploadFiles",
            idnt = "UPLOAD_INFER_EXCEPTION",
            stat = "FAIL",
            msge = f"[任務: 辨識用影片檔案] 上傳失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
        )
        return succ_cnt

# ==========================================================================================
def dispatch_next_task(active_vids_dir, task_queue_dir):
    """
    [名稱] 佇列任務遞補與重命名調度函式
    [功能] 當主要待辦區為空時，自動將佇列中第一順位 (task_001) 之影片搬移至待辦區，隨後將後續任務資料夾 (task_002, task_003...) 依序向前推進重命名 (task_001, task_002...)。
    [參數] 共計 2 組參數，以下說明:
           - active_vids_dir : [str/Path] 存放原始訓練影片之根目錄路徑 (e.g., raw_vids)
           - task_queue_dir  : [str/Path] 存放任務佇列資料之根目錄路徑 (e.g., task_queue)
    [輸出] bool: 是否成功執行遞補任務 (True-成功遞補, False-無待處理任務或待辦區非空)
    """
    active_vids_dir = Path(active_vids_dir)
    task_queue_dir  = Path(task_queue_dir)

    if not active_vids_dir.exists() or not task_queue_dir.exists():
        return False
    
    # [STEP-1] 檢驗當前待辦區 (raw_vids/) 既有之影片；若存在影片，則不執行遞補任務
    existing_vids = [
        file for file in active_vids_dir.iterdir()
        if file.is_file() and file.suffix.lower() in [".mp4", ".avi", ".mkv"]
    ]
    if existing_vids:
        return False

    # [STEP-2] 掃描 task_queue/ 目錄下既有之任務資料夾 (task_001, task_002...)；若無任務佇列，則不執行遞補任務
    existing_tasks = sorted(
        [task_dir for task_dir in task_queue_dir.iterdir() if task_dir.is_dir() and TASK_DIR_REGEX.match(task_dir.name.lower())],
        key=lambda x: x.name.lower()
    )
    if not existing_tasks:
        return False

    # [STEP-3] 處理第一順位 task_001 資料夾
    first_task_dir = existing_tasks[0]
    task_vid_files = [
        task_vid for task_vid in first_task_dir.iterdir()
        if task_vid.is_file() and task_vid.suffix.lower() in [".mp4", ".avi", ".mkv"]
    ]

    if not task_vid_files:
        # 若資料夾內均為無效影片，直接清除該空資料夾並遞歸處理由下一個補充
        shutil.rmtree(first_task_dir)
        return dispatch_next_task(active_vids_dir=active_vids_dir, task_queue_dir=task_queue_dir)

    moved_names = []
    for task_vid in task_vid_files:
        # 搬移任務佇列影片 >>> active_vids_dir/
        task_vid_name = task_vid.name
        dst_vid_path  = active_vids_dir / task_vid_name
        shutil.move(str(task_vid), str(dst_vid_path))
        moved_names.append(task_vid_name)

    shutil.rmtree(first_task_dir)

    # [STEP-4] 將剩餘任務資料夾依序向前對齊重新命名 (task_002 -> task_001...)
    remaining_tasks = sorted(
        [task_dir for task_dir in task_queue_dir.iterdir() if task_dir.is_dir() and TASK_DIR_REGEX.match(task_dir.name.lower())],
        key=lambda x: x.name.lower()
    )
    for idx, task_dir in enumerate(remaining_tasks, start=1):
        new_task_dir = task_queue_dir / f"task_{idx:03d}"
        if task_dir != new_task_dir:
            task_dir.rename(new_task_dir)

    log.CONTENT(
        type = "QUEUE",
        targ = "TaskQueue",
        idnt = "DISPATCH_TASK_QUEUE",
        stat = "SUCC",
        msge = f"[目錄: {first_task_dir.name}] 遞補成功 >>> 共計轉移 {len(moved_names)} 部影片至 {active_vids_dir.name}，後續任務序列已自動對齊"
    )
    return True

# ==========================================================================================
def archive_train_file(sgl_vid_path, dst_base_dir, train_date, ver_id):
    """
    [名稱] 訓練資料自動歸檔函式
    [功能] 將 raw_vids (原始影片)、frames (影像抽幀) 與 roboflow (物件標註) 資料夾下檔案，同步搬移至 archive 根目錄下。
    [參數] 共計 4 組參數，以下說明:
           - sgl_vid_path : [str/Path] 單一來源影片檔案之路徑，位於 raw_vids 下
           - dst_base_dir : [str/Path] 檔案歸檔集之根目錄路徑，位於 archive 下
           - train_date   : [str] 模型訓練日期，用於生成實體標記檔案 TXT (e.g., YYYY-MM-DD)
           - ver_id       : [str] 模型版本代碼，用於區分資料版本 (e.g., "v1.0.0")
    [輸出] bool: 歸檔作業是否成功 (True/False)
    """
    sgl_vid_path = Path(sgl_vid_path)
    dst_base_dir = Path(dst_base_dir)
    sgl_vid_name = sgl_vid_path.name
    sgl_vid_stem = sgl_vid_path.stem

    try:
        # 驗證來源檔案之存在性，攔截影片遺失或已遭其他程序移除之情境
        if not sgl_vid_path.exists():
            log.CONTENT(
                type = "VIDEO",
                targ = sgl_vid_name,
                idnt = "VALIDATE_SOURCE_FILE",
                stat = "WARN",
                msge = f"[檔案集: {sgl_vid_stem}] 歸檔警示 >>> 原始影片不存在，可能已被手動移除"
            )
            return False
        
        meta = analyze_video_filename(sgl_vid_path)
        domain = meta["domain"]

        """ [STAGE-1] 建構歸檔目錄規範 """
        # 版本封存根目錄: data/01_train/archive/<DOMAIN>/<VER_ID>/
        arch_ver_dir  = dst_base_dir / domain / ver_id
        arch_vids_dir = arch_ver_dir / "vids"
        arch_imgs_dir = arch_ver_dir / "imgs" / sgl_vid_stem
        arch_robo_dir = arch_ver_dir / "roboflow"

        arch_vids_dir.mkdir(parents=True, exist_ok=True)
        arch_imgs_dir.mkdir(parents=True, exist_ok=True)

        # 建立訓練日期標註檔: <YYYY-MM-DD>.txt
        if train_date:
            date_mark = arch_ver_dir / f"{train_date}.txt"
            date_mark.touch(exist_ok=True)

        """ [STAGE-2] 檔案同步搬移作業 """
        # 模型訓練資料根目錄: data/01_train/
        train_base_dir = path_config.train.base_dir

        # [STEP-1] 搬移原始影片 >>> vids/
        shutil.move(str(sgl_vid_path), str(arch_vids_dir / sgl_vid_name))

        # [STEP-2] 搬移抽幀影像 >>> imgs/<VID_STEM>/
        src_frms_dir = train_base_dir / "frames" / domain / sgl_vid_stem
        
        if src_frms_dir.exists():
            for frame in src_frms_dir.iterdir():
                frm_name = frame.name
                shutil.move(str(frame), str(arch_imgs_dir / frm_name))
            src_frms_dir.rmdir()

        # [STEP-2] 搬移標註資料集 >>> roboflow/ (含主領域 data.yaml、劃分集與 sub/ 次領域及其血統溯源檔)
        src_robo_dir = train_base_dir / "roboflow" / domain
        if src_robo_dir.exists():
            # 冪等性防禦: 如遇同領域多部原始影片下，當第一部搬移後，後續影片不再進行此步驟
            if arch_robo_dir.exists():
                # 預先清空目標目錄，防範舊資料殘留與 shutil.move 資料夾多重嵌套
                shutil.rmtree(arch_robo_dir)
            shutil.move(str(src_robo_dir), str(arch_robo_dir))
            log.CONTENT(
                type = "VIDEO",
                targ = sgl_vid_name,
                idnt = "ARCHIVE_TRAIN_FILES",
                stat = "SUCC",
                msge = f"[檔案集: {sgl_vid_stem}] 歸檔成功 >>> 原始影片、抽幀影像與標註全集(含次領域)已搬遷至 ({domain}/{ver_id})"
            )
            return True

        log.CONTENT(
            type = "VIDEO",
            targ = sgl_vid_name,
            idnt = "ARCHIVE_TRAIN_FILES",
            stat = "SUCC",
            msge = f"[檔案集: {sgl_vid_stem}] 歸檔成功 >>> 原始影片及抽幀影像已搬遷至 ({domain}/{ver_id})"
        )
        return True

    except Exception:
        # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
        sys_err = traceback.format_exc()
        log.CONTENT(
            type = "VIDEO",
            targ = sgl_vid_name,
            idnt = "ARCHIVE_TRAIN_EXCEPTION",
            stat = "FAIL",
            msge = f"[檔案集: {sgl_vid_stem}] 歸檔失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
        )
        return False

# ==========================================================================================
def archive_infer_file(sgl_vid_path, dst_base_dir, infer_date, ver_id):
    """
    [名稱] 辨識資料自動歸檔函式
    [功能] 將 vids_todo (待辨識之影片) 處理完畢後，自動分流搬移至 vids_done 根目錄下。
    [參數] 共計 4 組參數，以下說明:
           - sgl_vid_path : [str/Path] 單一來源影片檔案之路徑，位於 vids_todo 下
           - dst_base_dir : [str/Path] 檔案歸檔區之根目錄路徑，位於 vids_done 下
           - infer_date   : [str] 模型訓練日期，用於區分資料版本 (e.g., YYYY-MM-DD)
           - ver_id       : [str] 模型版本代碼，用於生成實體標記檔案 TXT (e.g., "v1.0.0")
    [輸出] bool: 歸檔作業是否成功 (True/False)
    """
    sgl_vid_path = Path(sgl_vid_path)
    dst_base_dir = Path(dst_base_dir)
    sgl_vid_name = sgl_vid_path.name

    try:
        # 驗證來源檔案存在性，攔截影片遺失或已遭其他程序移除之情境
        if not sgl_vid_path.exists():
            log.CONTENT(
                type = "VIDEO",
                targ = sgl_vid_name,
                idnt = "VALIDATE_SOURCE_FILE",
                stat = "WARN",
                msge = f"[影片: {sgl_vid_name}] 歸檔警示 >>> 辨識影片不存在，可能已被手動移除"
            )
            return False
        
        meta = analyze_video_filename(sgl_vid_path)
        domain = meta["domain"]

        """ [STAGE-1] 建構歸檔目錄規範 """
        # 日期封存根目錄: data/02_infer/vids_done/<DOMAIN>/<YYYY-MM-DD>/
        arch_date_dir = dst_base_dir / domain / infer_date
        arch_date_dir.mkdir(parents=True, exist_ok=True)

        # 建立資料版本標註檔: <VER_ID>.txt
        if ver_id:
            ver_id_mark = arch_date_dir / f"{ver_id}.txt"
            ver_id_mark.touch(exist_ok=True)

        """ [STAGE-2] 檔案同步搬移作業 """
        # [STEP-1] 搬移原始影片 >>> <YYYY-MM-DD>/
        shutil.move(str(sgl_vid_path), str(arch_date_dir / sgl_vid_name))

        log.CONTENT(
            type = "VIDEO",
            targ = sgl_vid_name,
            idnt = "ARCHIVE_INFER_FILE",
            stat = "SUCC",
            msge = f"[影片: {sgl_vid_name}] 歸檔成功 >>> 辨識影片已搬遷至 ({domain}/{infer_date})"
        )
        return True
                    
    except Exception:
        # 捕捉系統原始異常，進行堆疊追蹤並推送結構化例外日誌
        sys_err = traceback.format_exc()
        log.CONTENT(
            type = "VIDEO",
            targ = sgl_vid_name,
            idnt = "ARCHIVE_INFER_EXCEPTION",
            stat = "FAIL",
            msge = f"[影片: {sgl_vid_name}] 歸檔失敗 >>> 非預期系統異常，堆疊資訊如下: \n{sys_err}"
        )
        return False

# ==========================================================================================
def find_latest_weight(domain, wght_root_dir, sub_domain=None):
    """
    [名稱] 最新權重版本探測函式
    [功能] 依據主/次領域與權重根目錄動態解析路徑，依語意化版本編號 (SemVer, e.g., v1.0.0) 自動探測最新版本代碼。
    [參數] 共計 3 組參數，以下說明:
           - domain        : [str] 目標領域識別代碼 (e.g., 'VEH', 'MARA')
           - wght_root_dir : [str/Path] 最佳權重儲存根目錄路徑 (e.g., weights/)
           - sub_domain    : [str, optional] 次領域識別代碼 (e.g., 'LIC', 'BRAND')；預設為 None
    [輸出] str/None: 最新版本代碼 (e.g., "v1.0.0")；若無權重或目錄不存在則回傳 None
    """
    if not domain:
        return None

    main_domain   = str(domain).strip().upper()
    sub_domain    = str(sub_domain).strip().upper() if sub_domain else None
    wght_root_dir = Path(wght_root_dir)

    # [STEP-1] 決定該階層權重的版本聚合目錄:
    # 主領域權重路徑: weights/<DOMAIN>/main/<VER_ID>/best.engine
    # 次領域權重路徑: weights/<DOMAIN>/sub/<SUB_DOMAIN>/<VER_ID>/best.engine
    if sub_domain:
        ver_base_dir = wght_root_dir / main_domain / "sub" / sub_domain
    else:
        ver_base_dir = wght_root_dir / main_domain / "main"

    if not ver_base_dir.exists() or not ver_base_dir.is_dir():
        return None

    # [STEP-2] 檢索所有符合語意化版本規範 (vX.Y.Z) 且內含有效權重 (best.engine 或 best.pt) 之資料夾
    valid_ver_dirs = []
    for ver_id_dir in ver_base_dir.iterdir():
        if ver_id_dir.is_dir() and re.match(r"^v\d+\.\d+\.\d+$", ver_id_dir.name):
            has_engine = (ver_id_dir / "best.engine").exists()
            has_pt     = (ver_id_dir / "best.pt").exists()
            # 確認目錄下確實存在最佳權重實體檔，杜絕空資料夾干擾
            if has_engine or has_pt:
                valid_ver_dirs.append(ver_id_dir)

    if not valid_ver_dirs:
        return None
    
    # [STEP-3] 提取語意化版本號 (Major, Minor, Patch) 並進行整數排序
    def _parse_semver(dir_path):
        match = re.search(r"v(\d+)\.(\d+)\.(\d+)", dir_path.name)
        return tuple(map(int, match.groups())) if match else (0, 0, 0)

    latest_dir = max(valid_ver_dirs, key=_parse_semver)

    return latest_dir.name