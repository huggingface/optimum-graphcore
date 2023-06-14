import ctypes
import os


def load_custom_ops():
    file_dir = os.path.realpath(".")
    CUSTOM_OP_PATH = os.path.join(file_dir, "custom_ops.so")
    if os.path.exists(CUSTOM_OP_PATH):
        print(CUSTOM_OP_PATH)
        ctypes.cdll.LoadLibrary(CUSTOM_OP_PATH)
    else:
        print("Could not find custom_ops.so. Execute `make` before running this script.")


load_custom_ops()
