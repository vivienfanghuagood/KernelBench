#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from ctypes import CDLL, POINTER, byref, c_int


HIP_RUNTIME_LIB = "libamdhip64.so"
HIP_HEADER_PATH = "/opt/rocm/include/hip/hip_runtime_api.h"


def _load_enum_values(header_path: str) -> dict[str, int]:
    with open(header_path, "r", encoding="utf-8") as f:
        header = f.read()

    m = re.search(
        r"typedef\s+enum\s+hipDeviceAttribute_t\s*\{(.*?)\}\s*hipDeviceAttribute_t;",
        header,
        re.S,
    )
    if not m:
        raise RuntimeError("hipDeviceAttribute_t enum not found in HIP header")

    enum_block = m.group(1)
    # Remove block and line comments.
    enum_block = re.sub(r"/\*.*?\*/", "", enum_block, flags=re.S)
    enum_block = re.sub(r"//.*?$", "", enum_block, flags=re.M)

    entries = [e.strip() for e in enum_block.split(",") if e.strip()]
    values: dict[str, int] = {}
    current = 0
    for entry in entries:
        if "=" in entry:
            name, value = [x.strip() for x in entry.split("=", 1)]
            if value.isdigit():
                current = int(value)
            elif value in values:
                current = values[value]
            else:
                # Fallback: ignore unknown explicit expressions.
                current = current
            values[name] = current
        else:
            name = entry
            values[name] = current
        current += 1
    return values


def _parse_rocminfo() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(["rocminfo"], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return [{"rocminfo_error": str(exc)}]

    devices = []
    current = {}
    for line in output.splitlines():
        if line.strip().startswith("*******"):
            if current.get("Device Type") == "GPU":
                devices.append(current)
            current = {}
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()

    if current.get("Device Type") == "GPU":
        devices.append(current)

    return devices


def main() -> int:
    try:
        enum_values = _load_enum_values(HIP_HEADER_PATH)
    except Exception as exc:
        print(f"[ERROR] Failed to parse HIP headers: {exc}", file=sys.stderr)
        return 1

    try:
        hip = CDLL(HIP_RUNTIME_LIB)
    except Exception as exc:
        print(f"[ERROR] Failed to load {HIP_RUNTIME_LIB}: {exc}", file=sys.stderr)
        return 1

    hipGetDeviceCount = hip.hipGetDeviceCount
    hipGetDeviceCount.argtypes = [POINTER(c_int)]
    hipGetDeviceCount.restype = c_int

    hipDeviceGetAttribute = hip.hipDeviceGetAttribute
    hipDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, c_int]
    hipDeviceGetAttribute.restype = c_int

    count = c_int()
    if hipGetDeviceCount(byref(count)) != 0:
        print("[ERROR] hipGetDeviceCount failed", file=sys.stderr)
        return 1

    rocminfo_devices = _parse_rocminfo()

    attrs = {
        "Max Registers per Block": "hipDeviceAttributeMaxRegistersPerBlock",
        "Max Shared Memory per Block": "hipDeviceAttributeMaxSharedMemoryPerBlock",
        "Max Threads per Block": "hipDeviceAttributeMaxThreadsPerBlock",
        "Max Threads per CU": "hipDeviceAttributeMaxThreadsPerMultiProcessor",
        "Shared Memory per CU": "hipDeviceAttributeSharedMemPerMultiprocessor",
        "Warp Size": "hipDeviceAttributeWarpSize",
    }

    for device_index in range(count.value):
        result = {
            "device_index": device_index,
        }
        if device_index < len(rocminfo_devices):
            result.update(
                {
                    "rocminfo_name": rocminfo_devices[device_index].get("Name"),
                    "rocminfo_marketing_name": rocminfo_devices[device_index].get(
                        "Marketing Name"
                    ),
                }
            )

        for label, enum_name in attrs.items():
            enum_value = enum_values.get(enum_name)
            if enum_value is None:
                result[label] = None
                continue
            out = c_int()
            status = hipDeviceGetAttribute(byref(out), enum_value, device_index)
            result[label] = out.value if status == 0 else None

        print(json.dumps(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
