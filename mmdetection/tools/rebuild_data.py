from copy import deepcopy
from typing import Dict, Any


def hacking_rebuild_data(data: Dict[str, Any], old_data_root: str, new_data_root,) -> Dict[str, Any]:
    new_data = deepcopy(data)

    def rebuild_value(old_value: str):
        return new_data_root + old_value[len(old_data_root):]

    def rebuild_help(first_key: str, second_key: str):
        new_data[first_key][second_key] = rebuild_value(data[first_key][second_key])

    for key in ("train", "val", "test"):
        rebuild_help(key, "ann_file")
        rebuild_help(key, "img_prefix")

    return new_data


def rebuild_data(data: Dict[str, Any], old_data_root: str, new_data_root: str) -> Dict[str, Any]:
    return _rebuild_data_help(deepcopy(data), old_data_root, new_data_root)


def _rebuild_data_help(data: Dict[str, Any], old_data_root: str, new_data_root: str) -> Dict[str, Any]:
    def rebuild_value(old_value: str):
        return new_data_root + old_value[len(old_data_root):]
    for key in data:
        if isinstance(data[key], dict): _rebuild_data_help(data[key], old_data_root, new_data_root)
        elif isinstance(data[key], str):
            temp: str = data[key]
            if temp.startswith(old_data_root): data[key] = rebuild_value(temp)
    return data
