import os
import shutil
import datetime
import glob
import stat

# Configuration
MAX_BACKUPS = 40
HISTORY_DIR = os.path.join("..", "ARC Backups")
PROJECT_ROOT = "."
# Directories to exclude. "dumps" holds multi-GB WER crash dumps that must
# never be copied into snapshots (was ballooning backups to 100+ GB).
EXCLUDE_DIRS = {".vs", "logs", "dumps"}
EXCLUDE_FILES = set()
# Path-aware excludes: (parent_dir_name, child_name). Skipped only when the
# child appears under that specific parent — avoids basename collisions with
# unrelated directories that happen to share a common name like "output".
EXCLUDE_NESTED = {("profiling", "output")}

def on_rm_error(func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's try to change its permissions
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except:
        pass

def get_next_id():
    # Priority: Scan folder names for "(ID <num>)"
    max_id = 0
    try:
        all_dirs = glob.glob(os.path.join(HISTORY_DIR, "*"))
        for d in all_dirs:
            # Look for pattern "...(ID <N>)" at end of name
            # This handles both "CHECKPOINT... (ID N)" and "backup... (ID N)"
            if "(ID " in d:
                try:
                    # simplistic parse: split by "(ID " take last part, strip ")"
                    # Example: "backups\backup 28-01-2026 (ID 52)" -> "52)" -> 52
                    part = d.split("(ID ")[-1]
                    val = int(part.rstrip(")"))
                    if val > max_id:
                        max_id = val
                except:
                    continue
    except:
        pass

    # If we found IDs, continue sequence
    if max_id > 0:
        return max_id + 1

    # Fallback: If 0 backups have IDs (first run after migration), estimate count
    count = len(glob.glob(os.path.join(HISTORY_DIR, "backup *"))) + \
            len(glob.glob(os.path.join(HISTORY_DIR, "CHECKPOINT *")))
    return count + 1

def create_backup():
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

    backup_id = get_next_id()
    
    # Milestone: Every 40 backups
    is_checkpoint = (backup_id % MAX_BACKUPS == 0)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %I-%M-%S %p")
    
    if is_checkpoint:
        backup_name = f"CHECKPOINT {timestamp} (ID {backup_id})"
        print(f"[*] Backup: Milestone reached! Creating '{backup_name}'...")
    else:
        backup_name = f"backup {timestamp} (ID {backup_id})"
        print(f"[*] Backup: Creating snapshot '{backup_name}'...")

    backup_path = os.path.join(HISTORY_DIR, backup_name)

    def ignore_func(directory, files):
        parent_name = os.path.basename(os.path.normpath(directory))
        skipped = []
        for f in files:
            if f in EXCLUDE_DIRS or f in EXCLUDE_FILES:
                skipped.append(f)
            elif (parent_name, f) in EXCLUDE_NESTED:
                skipped.append(f)
        return skipped

    try:
        shutil.copytree(PROJECT_ROOT, backup_path, ignore=ignore_func)
    except Exception as e:
        print(f"[!] Backup failed: {e}")
        return

    # Rotation Logic: Only rotate standard 'backup *' folders. Checkpoints are kept forever.
    current_std_backups = glob.glob(os.path.join(HISTORY_DIR, "backup *"))
    current_std_backups.sort(key=os.path.getmtime)
    
    if len(current_std_backups) > MAX_BACKUPS:
        excess = len(current_std_backups) - MAX_BACKUPS
        print(f"[*] Backup: Rotating Standard... deleting {excess} old snapshots.")
        for i in range(excess):
            to_delete = current_std_backups[i]
            deleted = False
            for retry in range(3):
                try:
                    shutil.rmtree(to_delete, onerror=on_rm_error)
                    deleted = True
                    break
                except:
                    import time
                    time.sleep(0.5)
            
            if not deleted:
                try:
                    shutil.rmtree(to_delete, onerror=on_rm_error)
                except:
                    pass

if __name__ == "__main__":
    create_backup()
