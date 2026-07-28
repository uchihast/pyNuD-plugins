"""
Minimal ASD (AFM Scanning Data) loader for pyNuD-simulator.

- Reads header + one frame of 1ch (and optional 2ch) data.
- No OpenCV dependency.
- Returns NumPy arrays + metadata (no global state).
"""

from __future__ import annotations

import datetime
import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ASDHeader:
    file_type: int
    file_header_size: int
    frame_header_size: int
    text_encoding: int
    ope_name_size: int
    comment_size: int
    data_type_1ch: int
    data_type_2ch: int
    frame_num: int
    image_num: int
    scan_direction: int
    x_pixel: int
    y_pixel: int
    x_scan_size: int
    y_scan_size: int
    frame_time: float
    piezo_const_z: float
    driver_gain_z: float
    ope_name: str
    comment: str


def _u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _i32(f) -> int:
    return struct.unpack("<i", f.read(4))[0]


def _i16(f) -> int:
    return struct.unpack("<h", f.read(2))[0]


def _u16(f) -> int:
    return struct.unpack("<H", f.read(2))[0]


def _u8(f) -> int:
    return struct.unpack("<B", f.read(1))[0]


def _f32(f) -> float:
    return struct.unpack("<f", f.read(4))[0]


def _decode_text(data: bytes, text_encoding: int) -> str:
    encodings = (
        ("cp932", "utf-8")
        if int(text_encoding) == 932
        else ("utf-8", "cp932")
    )
    for enc in encodings:
        try:
            return data.decode(enc).rstrip("\x00")
        except UnicodeDecodeError:
            pass
    return data.decode(encodings[0], errors="replace").rstrip("\x00")


def write_asd_height_frame(
    filepath: str,
    height_nm: np.ndarray,
    x_scan_size_nm: float,
    y_scan_size_nm: float,
    *,
    comment: str = "",
    operator_name: str = "Nobody",
) -> None:
    """Write one height frame in the ASD layout consumed by this module.

    ASD height samples are unsigned 16-bit values. The height map is shifted so
    its minimum is zero (matching ``_convert_1ch`` on read), and the Z-piezo
    scale is enlarged when necessary so tall structures do not wrap around.
    """
    height = np.asarray(height_nm, dtype=np.float64)
    if height.ndim != 2 or height.size == 0:
        raise ValueError("ASD height data must be a non-empty 2D array")
    if not np.all(np.isfinite(height)):
        raise ValueError("ASD height data contains non-finite values")

    y_pixels, x_pixels = height.shape
    normalized = height - float(np.min(height))
    driver_gain_z = 2.0
    max_height_nm = float(np.max(normalized))
    piezo_const_z = max(
        20.0,
        max_height_nm / (5.0 * driver_gain_z),
    )

    raw_float = (
        5.0 - normalized / piezo_const_z / driver_gain_z
    ) * 4096.0 / 10.0
    raw_u16 = np.clip(np.rint(raw_float), 0, np.iinfo(np.uint16).max).astype(
        "<u2"
    )

    operator_bytes = str(operator_name).encode("cp932", errors="replace")
    comment_bytes = str(comment).encode("cp932", errors="replace")
    file_header_size = 165 + len(operator_bytes) + len(comment_bytes)
    frame_header_size = 32
    now = datetime.datetime.now()

    with open(filepath, "wb") as f:
        # File header.
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", file_header_size))
        f.write(struct.pack("<i", frame_header_size))
        f.write(struct.pack("<i", 932))
        f.write(struct.pack("<i", len(operator_bytes)))
        f.write(struct.pack("<i", len(comment_bytes)))
        f.write(struct.pack("<i", 20564))
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", int(x_pixels)))
        f.write(struct.pack("<i", int(y_pixels)))
        f.write(struct.pack("<i", int(x_scan_size_nm)))
        f.write(struct.pack("<i", int(y_scan_size_nm)))
        f.write(struct.pack("<B", 0))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", now.year))
        f.write(struct.pack("<i", now.month))
        f.write(struct.pack("<i", now.day))
        f.write(struct.pack("<i", now.hour))
        f.write(struct.pack("<i", now.minute))
        f.write(struct.pack("<i", now.second))
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<f", 1.0))
        f.write(struct.pack("<f", 1.0))
        f.write(struct.pack("<f", 1.0))
        f.write(struct.pack("<iiii", 0, 0, 0, 0))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", 262144))
        f.write(struct.pack("<i", 12))
        f.write(struct.pack("<f", 4000.0))
        f.write(struct.pack("<f", 1700.0))
        f.write(struct.pack("<f", 1.0))
        f.write(struct.pack("<f", 1.0))
        f.write(struct.pack("<f", piezo_const_z))
        f.write(struct.pack("<f", driver_gain_z))
        f.write(operator_bytes)
        f.write(comment_bytes)

        # Frame header. These extrema describe the encoded samples, not nm.
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<H", int(np.max(raw_u16))))
        f.write(struct.pack("<H", int(np.min(raw_u16))))
        f.write(struct.pack("<h", 0))
        f.write(struct.pack("<h", 0))
        f.write(struct.pack("<f", 0.0))
        f.write(struct.pack("<f", 0.0))
        f.write(struct.pack("<B", 0))
        f.write(struct.pack("<B", 0))
        f.write(struct.pack("<h", 0))
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<i", 0))

        f.write(raw_u16.tobytes(order="C"))


def read_asd_header(f) -> ASDHeader:
    file_type = _i32(f)
    if file_type != 1:
        raise ValueError(f"Unsupported ASD FileType={file_type} (expected 1)")

    file_header_size = _i32(f)
    frame_header_size = _i32(f)
    text_encoding = _i32(f)
    ope_name_size = _i32(f)

    comment_size = _i32(f)
    data_type_1ch = _i32(f)
    data_type_2ch = _i32(f)
    frame_num = _i32(f)
    image_num = _i32(f)
    scan_direction = _i32(f)
    _scan_try_num = _i32(f)  # unused
    x_pixel = _i32(f)
    y_pixel = _i32(f)

    if x_pixel <= 0 or y_pixel <= 0:
        # Keep behavior aligned with legacy loader
        x_pixel, y_pixel = 256, 256

    x_scan_size = _i32(f)
    y_scan_size = _i32(f)
    _ave_flag = _u8(f)
    _ave_num = _i32(f)
    _year = _i32(f)
    _month = _i32(f)
    _day = _i32(f)
    _hour = _i32(f)
    _minute = _i32(f)
    _second = _i32(f)
    _x_round = _i32(f)
    _y_round = _i32(f)
    frame_time = _f32(f)
    _sensitivity = _f32(f)
    _phase_sens = _f32(f)
    # Legacy fileio reads four offsets but overwrites into one var; just skip the bytes.
    f.read(4 * 4)
    _machine_no = _i32(f)
    _ad_range = _i32(f)
    _ad_resolution = _i32(f)
    _max_scan_size_x = _f32(f)
    _max_scan_size_y = _f32(f)
    _piezo_const_x = _f32(f)
    _piezo_const_y = _f32(f)
    piezo_const_z = _f32(f)
    driver_gain_z = _f32(f)

    # Operator name and comment live at the end of the file header.
    ope_name = ""
    comment = ""
    try:
        ope_name_offset = file_header_size - comment_size - ope_name_size
        if ope_name_offset < 0:
            ope_name_offset = 0
        f.seek(ope_name_offset, 0)
        ope_name = _decode_text(
            f.read(max(ope_name_size, 0)),
            text_encoding,
        )
        comment = _decode_text(
            f.read(max(comment_size, 0)),
            text_encoding,
        )
    finally:
        f.seek(file_header_size, 0)

    return ASDHeader(
        file_type=file_type,
        file_header_size=file_header_size,
        frame_header_size=frame_header_size,
        text_encoding=text_encoding,
        ope_name_size=ope_name_size,
        comment_size=comment_size,
        data_type_1ch=data_type_1ch,
        data_type_2ch=data_type_2ch,
        frame_num=frame_num,
        image_num=image_num,
        scan_direction=scan_direction,
        x_pixel=x_pixel,
        y_pixel=y_pixel,
        x_scan_size=x_scan_size,
        y_scan_size=y_scan_size,
        frame_time=frame_time,
        piezo_const_z=piezo_const_z,
        driver_gain_z=driver_gain_z,
        ope_name=ope_name,
        comment=comment,
    )


def _convert_1ch(raw_u16: np.ndarray, header: ASDHeader) -> np.ndarray:
    ary = raw_u16.astype(np.float64)

    if header.data_type_1ch == 18512:
        # Voltage (no nm conversion)
        out = (5.0 - ((ary * 10.0) / 4096.0))
        return out

    # Default: nm conversion
    out = (5.0 - ((ary * 10.0) / 4096.0)) * header.piezo_const_z * header.driver_gain_z

    # Height (nm): normalize minimum to 0
    if header.data_type_1ch == 20564:
        out = out - float(np.min(out))

    return out


def read_asd_frame(
    filepath: str, frame_index: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Returns:
      data_1ch (float64): shape (y_pixel, x_pixel)
      data_2ch (float64|None): shape (y_pixel, x_pixel) in volts (10V range) if present
      meta (dict): pixels + scan size + scan direction + header fields
    """
    with open(filepath, "rb") as f:
        header = read_asd_header(f)

        frame_num = max(int(header.frame_num), 0)
        if frame_num <= 0:
            raise ValueError("ASD contains zero frames")

        idx = int(frame_index)
        if idx < 0:
            idx = 0
        if idx >= frame_num:
            idx = frame_num - 1

        count = int(header.x_pixel) * int(header.y_pixel)
        bytes_per_frame = int(header.frame_header_size) + (2 * count)

        # 1ch frame: [frame_header][uint16 image]
        f.seek(int(header.file_header_size) + bytes_per_frame * idx, 0)

        # Frame header (mostly unused, but included for completeness)
        current_num = _u32(f)
        max_data = _u16(f)
        min_data = _u16(f)
        x_offset = _i16(f)
        y_offset = _i16(f)
        x_tilt = _f32(f)
        y_tilt = _f32(f)
        laser_flag = _u8(f)
        f.seek(3, 1)
        temp_flag = _i32(f)
        f.seek(4, 1)

        raw = f.read(2 * count)
        if len(raw) != 2 * count:
            raise ValueError(f"ASD read error: expected {2 * count} bytes, got {len(raw)}")
        raw_u16 = np.frombuffer(raw, dtype="<u2", count=count).reshape((header.y_pixel, header.x_pixel))
        data_1ch = _convert_1ch(raw_u16, header)

        data_2ch = None
        if int(header.data_type_2ch) != 0:
            # Layout: after all 1ch frames, 2ch frames follow in same [frame_header][image] layout.
            offset_2ch = int(header.file_header_size) + bytes_per_frame * frame_num + bytes_per_frame * idx
            f.seek(offset_2ch + int(header.frame_header_size), 0)
            raw2 = f.read(2 * count)
            if len(raw2) == 2 * count:
                raw2_u16 = np.frombuffer(raw2, dtype="<u2", count=count).reshape((header.y_pixel, header.x_pixel))
                data_2ch = raw2_u16.astype(np.float64) * 10.0 / 4096.0

        meta: Dict[str, Any] = {
            "x_pixel": int(header.x_pixel),
            "y_pixel": int(header.y_pixel),
            "x_scan_size": float(header.x_scan_size),
            "y_scan_size": float(header.y_scan_size),
            "scan_direction": int(header.scan_direction),
            "data_type_1ch": int(header.data_type_1ch),
            "data_type_2ch": int(header.data_type_2ch),
            "frame_index": idx,
            "frame_num": frame_num,
            "frame_header": {
                "current_num": int(current_num),
                "max_data": int(max_data),
                "min_data": int(min_data),
                "x_offset": int(x_offset),
                "y_offset": int(y_offset),
                "x_tilt": float(x_tilt),
                "y_tilt": float(y_tilt),
                "laser_flag": int(laser_flag),
                "temp_flag": int(temp_flag),
            },
            "comment": header.comment,
            "ope_name": header.ope_name,
        }

        return data_1ch, data_2ch, meta
