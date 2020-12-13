import os
import pickle
from typing import *
from pathlib import Path


def check_does_file_exist(path: str) -> bool:
    """Check does file exist on given path.

    Args:
        path: path to the file

    Returns:
        True if file exists. False otherwise.
    """
    return os.path.exists(path) and os.path.isfile(path)


def check_does_dir_exist(path: str, create_dir: bool) -> Union[bool, None]:
    """Check does directory exist on given path and create it if doesn't exist and it is necessary.

    Args:
        path: path to the directory
        create_dir: flag that indicates should directory be created in case if it doesn't exist

    Returns:
        True if directory exists. False otherwise.
    """
    dir_exists = os.path.exists(path) and os.path.isdir(path)

    if not dir_exists:
        if not create_dir:
            return False
        Path(path).mkdir(parents=True)
    return True


def serialize_data(data: Any, path: str) -> None:
    """Serialize data on given path using pickle.

    Args:
        data: Object that should be serialized
        path: path to the file where data will be serialized

    Returns:
        no value
    """
    parent_dir_path = os.path.dirname(path)
    if parent_dir_path:
        check_does_dir_exist(path=parent_dir_path, create_dir=True)
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_data(path: str) -> Any:
    """Deserialize data from given path using pickle.

    Args:
        path: path to the file from where data should be deserialized

    Returns:
        Deserialized object
    """
    if check_does_file_exist(path):
        with open(path, 'rb') as file:
            return pickle.load(file)
    raise FileNotFoundError
