import json
from datetime import datetime
from typing import List

from src.helper import ensure_diretory


class SimpleLogger:
    info_msgs: List[str] = []
    warning_msgs: List[str] = []
    error_msgs: List[str] = []

    def info(self, msg: str) -> None:
        now: datetime = datetime.now()
        self.info_msgs.append(f"INFO: {now.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")
        # print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} - INFO - {msg}")

    def warning(self, msg: str) -> None:
        now: datetime = datetime.now()
        self.warning_msgs.append(
            f"WARNING: {now.strftime('%Y-%m-%d %H:%M:%S')} - {msg}"
        )
        # print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {msg}")

    def error(self, msg: str) -> None:
        now: datetime = datetime.now()
        self.error_msgs.append(f"ERROR: {now.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")
        print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {msg}")

    def dump_log(self, path: str) -> None:
        ensure_diretory(path)
        with open(path, "w+") as f:
            json.dump(self.info_msgs + self.warning_msgs + self.error_msgs, f, indent=4)
