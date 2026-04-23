# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython accelerators for Vaa3D image loading."""

from __future__ import annotations

import os
import struct

import numpy as np
cimport numpy as np
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE
from libc.stdint cimport uint8_t, uint16_t


DEF V3DRAW_HEADER_SIZE = 43
DEF V3DPBD_HEADER_SIZE = 43

FORMAT_KEY_V3DRAW = b"raw_image_stack_by_hpeng"
FORMAT_KEY_V3DPBD = b"v3d_volume_pkbitdf_encod"


cdef inline uint16_t _read_u16(const uint8_t* data, Py_ssize_t pos, bint little) noexcept:
    if little:
        return <uint16_t>(data[pos] | (data[pos + 1] << 8))
    return <uint16_t>((data[pos] << 8) | data[pos + 1])


cdef inline int _pbd_delta(unsigned int encoded, unsigned int threshold) noexcept:
    if encoded > threshold:
        return <int>threshold - <int>encoded
    return <int>encoded


cdef void _decompress_pbd8(
    const uint8_t* comp,
    Py_ssize_t comp_len,
    uint8_t* out,
    Py_ssize_t out_len,
):
    cdef:
        Py_ssize_t cp = 0
        Py_ssize_t dp = 0
        Py_ssize_t count
        Py_ssize_t j
        unsigned int code
        unsigned int packed = 0
        unsigned int delta_code
        int delta
        int prior = 0
        int value
        uint8_t repeat_value

    while cp < comp_len:
        code = comp[cp]
        cp += 1

        if code < 33:
            count = code + 1
            if cp + count > comp_len or dp + count > out_len:
                raise ValueError("Malformed Vaa3D PBD8 literal block.")
            for j in range(count):
                out[dp + j] = comp[cp + j]
            prior = out[dp + count - 1]
            dp += count
            cp += count
        elif code < 128:
            count = code - 32
            if cp + ((count + 3) >> 2) > comp_len or dp + count > out_len:
                raise ValueError("Malformed Vaa3D PBD8 delta block.")
            for j in range(count):
                if (j & 3) == 0:
                    packed = comp[cp]
                    cp += 1
                delta_code = (packed >> (2 * (j & 3))) & 3
                delta = -1 if delta_code == 3 else <int>delta_code
                value = prior + delta
                out[dp] = <uint8_t>value
                prior = out[dp]
                dp += 1
        else:
            count = code - 127
            if cp >= comp_len or dp + count > out_len:
                raise ValueError("Malformed Vaa3D PBD8 repeat block.")
            repeat_value = comp[cp]
            cp += 1
            for j in range(count):
                out[dp + j] = repeat_value
            prior = repeat_value
            dp += count

    if dp != out_len:
        raise ValueError("Vaa3D PBD8 decompressed size does not match the header.")


cdef void _decompress_pbd16(
    const uint8_t* comp,
    Py_ssize_t comp_len,
    uint16_t* out,
    Py_ssize_t out_len,
    bint little,
):
    cdef:
        Py_ssize_t cp = 0
        Py_ssize_t dp = 0
        Py_ssize_t count
        Py_ssize_t j
        Py_ssize_t k
        Py_ssize_t bitpos
        Py_ssize_t byte_count
        unsigned int code
        unsigned int bits
        unsigned int threshold
        unsigned int delta_code
        int delta
        int value
        unsigned int prior = 0
        uint16_t repeat_value

    while cp < comp_len:
        code = comp[cp]
        cp += 1

        if code < 32:
            count = code + 1
            if cp + count * 2 > comp_len or dp + count > out_len:
                raise ValueError("Malformed Vaa3D PBD16 literal block.")
            for j in range(count):
                out[dp + j] = _read_u16(comp, cp + j * 2, little)
            prior = out[dp + count - 1]
            dp += count
            cp += count * 2
            continue

        if code >= 223:
            count = code - 222
            if cp + 2 > comp_len or dp + count > out_len:
                raise ValueError("Malformed Vaa3D PBD16 repeat block.")
            repeat_value = _read_u16(comp, cp, little)
            cp += 2
            for j in range(count):
                out[dp + j] = repeat_value
            prior = repeat_value
            dp += count
            continue

        if code < 80:
            count = code - 31
            bits = 3
            threshold = 4
        elif code < 183:
            count = code - 79
            bits = 4
            threshold = 8
        else:
            count = code - 182
            bits = 5
            threshold = 16

        byte_count = (count * bits + 7) >> 3
        if cp + byte_count > comp_len or dp + count > out_len:
            raise ValueError("Malformed Vaa3D PBD16 delta block.")

        bitpos = 0
        for j in range(count):
            delta_code = 0
            for k in range(bits):
                delta_code = (
                    (delta_code << 1)
                    | ((comp[cp + (bitpos >> 3)] >> (7 - (bitpos & 7))) & 1)
                )
                bitpos += 1
            delta = _pbd_delta(delta_code, threshold)
            value = <int>prior + delta
            out[dp] = <uint16_t>(value & 0xFFFF)
            prior = out[dp]
            dp += 1

        cp += byte_count

    if dp != out_len:
        raise ValueError("Vaa3D PBD16 decompressed size does not match the header.")


def load_v3draw(path):
    """Load a Vaa3D v3draw/raw file using direct NumPy file reads."""
    cdef:
        bytes header
        str endian
        short datatype
        tuple dims
        object dtype
        object native_dtype
        object image
        long long total_units
        long long expected_size

    path = os.fspath(path)
    with open(path, "rb") as handle:
        header = handle.read(V3DRAW_HEADER_SIZE)

    if len(header) < V3DRAW_HEADER_SIZE:
        raise ValueError(f"Vaa3D raw file is truncated: {path}")
    if header[:len(FORMAT_KEY_V3DRAW)] != FORMAT_KEY_V3DRAW:
        raise ValueError(f"Unsupported Vaa3D raw header in {path}.")

    if header[len(FORMAT_KEY_V3DRAW):len(FORMAT_KEY_V3DRAW) + 1] == b"B":
        endian = ">"
    elif header[len(FORMAT_KEY_V3DRAW):len(FORMAT_KEY_V3DRAW) + 1] == b"L":
        endian = "<"
    else:
        raise ValueError(f"Unsupported Vaa3D endian code in {path}.")

    datatype = struct.unpack(f"{endian}h", header[25:27])[0]
    dims = struct.unpack(f"{endian}iiii", header[27:43])
    if datatype == 1:
        dtype = np.dtype("u1")
    elif datatype == 2:
        dtype = np.dtype(f"{endian}u2")
    elif datatype == 4:
        dtype = np.dtype(f"{endian}f4")
    else:
        raise ValueError(f"Unsupported Vaa3D datatype code {datatype!r}.")

    total_units = <long long>dims[0] * dims[1] * dims[2] * dims[3]
    expected_size = V3DRAW_HEADER_SIZE + total_units * np.dtype(dtype).itemsize
    if os.path.getsize(path) != expected_size:
        raise ValueError(f"Vaa3D raw file size does not match the header: {path}")

    image = np.fromfile(path, dtype=dtype, count=total_units, offset=V3DRAW_HEADER_SIZE)
    image = image.reshape((dims[3], dims[2], dims[1], dims[0]))
    native_dtype = np.dtype(dtype).newbyteorder("=")
    return image.astype(native_dtype, copy=False)


def load_v3dpbd(path):
    """Load a Vaa3D PBD-compressed file with C-level decompression loops."""
    cdef:
        bytes data
        const uint8_t* comp
        Py_ssize_t comp_len
        str endian
        bint little
        short datatype
        tuple dims
        Py_ssize_t total_units
        np.ndarray[np.uint8_t, ndim=1] out8
        np.ndarray[np.uint16_t, ndim=1] out16

    path = os.fspath(path)
    with open(path, "rb") as handle:
        data = handle.read()

    if PyBytes_GET_SIZE(data) < V3DPBD_HEADER_SIZE:
        raise ValueError(f"Vaa3D PBD file is truncated: {path}")
    if data[:len(FORMAT_KEY_V3DPBD)] != FORMAT_KEY_V3DPBD:
        raise ValueError(f"Unsupported Vaa3D PBD header in {path}.")

    if data[len(FORMAT_KEY_V3DPBD):len(FORMAT_KEY_V3DPBD) + 1] == b"B":
        endian = ">"
        little = False
    elif data[len(FORMAT_KEY_V3DPBD):len(FORMAT_KEY_V3DPBD) + 1] == b"L":
        endian = "<"
        little = True
    else:
        raise ValueError(f"Unsupported Vaa3D endian code in {path}.")

    datatype = struct.unpack(f"{endian}h", data[25:27])[0]
    dims = struct.unpack(f"{endian}iiii", data[27:43])
    total_units = <Py_ssize_t>dims[0] * dims[1] * dims[2] * dims[3]
    comp = <const uint8_t*>PyBytes_AS_STRING(data) + V3DPBD_HEADER_SIZE
    comp_len = PyBytes_GET_SIZE(data) - V3DPBD_HEADER_SIZE

    if datatype == 1:
        out8 = np.empty(total_units, dtype=np.uint8)
        _decompress_pbd8(comp, comp_len, <uint8_t*>out8.data, total_units)
        return out8.reshape((dims[3], dims[2], dims[1], dims[0]))

    if datatype == 2:
        out16 = np.empty(total_units, dtype=np.uint16)
        _decompress_pbd16(comp, comp_len, <uint16_t*>out16.data, total_units, little)
        return out16.reshape((dims[3], dims[2], dims[1], dims[0]))

    if datatype == 33:
        raise NotImplementedError("Vaa3D PBD datatype 33 is not implemented.")
    raise ValueError(f"Unsupported Vaa3D PBD datatype code {datatype!r}.")
