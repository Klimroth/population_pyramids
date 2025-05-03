import os
from typing import Any, Dict

import pandas as pd


def ensure_diretory(path: str) -> None:
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_dictionary_as_excel(dict_to_save: Dict[Any, Any], path: str) -> None:
    ensure_diretory(path)
    pd.DataFrame(dict_to_save).to_excel(path, index=False)
