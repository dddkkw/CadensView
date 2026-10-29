"""Private ORT sessions and measured Windows adapter selection; no Qt/torch."""
import ctypes as C
from ctypes import wintypes as W
import statistics
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

MODEL_SHA256 = 'af5bc7dcbc517613d0d7fb39ee97add80e52a05d8de4717652e500a5bf573810'
WEIGHTS_SHA256 = 'a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de'


class GPUError(RuntimeError):
    """The coordinator may retry once in a fresh CPU process."""


def color_histograms(frames):
    # Promoted unchanged from the validated split-model research implementation.
    bins = frames.astype(np.int32) // 32
    bins = bins[..., 0] * 64 + bins[..., 1] * 8 + bins[..., 2]
    bins = bins.reshape(100, 1296) + np.arange(100, dtype=np.int32)[:, None] * 512
    counts = np.bincount(bins.ravel(), minlength=100 * 512).reshape(1, 100, 512).astype(np.float32)
    return counts / np.sqrt(np.sum(counts * counts, axis=2, keepdims=True, dtype=np.float32))


def adapters():
    if sys.platform != 'win32':
        return []
    class GUID(C.Structure):
        _fields_ = [('a', W.DWORD), ('b', W.WORD), ('c', W.WORD), ('d', C.c_ubyte * 8)]
    class LUID(C.Structure):
        _fields_ = [('lo', W.DWORD), ('hi', W.LONG)]
    class Desc(C.Structure):
        _fields_ = [('name', W.WCHAR * 128), ('vendor', W.UINT), ('device', W.UINT),
                    ('subsystem', W.UINT), ('revision', W.UINT),
                    ('dedicated_video', C.c_size_t), ('dedicated_system', C.c_size_t),
                    ('shared', C.c_size_t), ('luid', LUID), ('flags', W.UINT)]
    def method(ptr, index, restype, *types):
        table = C.cast(ptr, C.POINTER(C.POINTER(C.c_void_p))).contents
        return C.WINFUNCTYPE(restype, C.c_void_p, *types)(table[index])
    guid = GUID(0x770aae78, 0xf26f, 0x4dba, (C.c_ubyte * 8)(0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87))
    factory = C.c_void_p()
    if C.WinDLL('dxgi.dll').CreateDXGIFactory1(C.byref(guid), C.byref(factory)):
        raise OSError('DXGI adapter enumeration failed')
    result, seen = [], set()
    try:
        for i in range(32):
            adapter = C.c_void_p()
            if method(factory, 12, W.LONG, W.UINT, C.POINTER(C.c_void_p))(factory, i, C.byref(adapter)):
                break
            try:
                desc = Desc()
                if method(adapter, 10, W.LONG, C.POINTER(Desc))(adapter, C.byref(desc)):
                    continue
                luid = f'{desc.luid.hi & 0xffffffff:08x}:{desc.luid.lo:08x}'
                if not desc.flags & 2 and luid not in seen:
                    seen.add(luid)
                    # DXGI's documented UMD version query; never infer DML ids from CUDA.
                    device_guid = GUID(0x54ec77fa, 0x1377, 0x44e6, (C.c_ubyte * 8)(0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c))
                    version = C.c_longlong()
                    supported = method(adapter, 9, W.LONG, C.POINTER(GUID), C.POINTER(C.c_longlong))(
                        adapter, C.byref(device_guid), C.byref(version)) == 0
                    result.append(dict(device_id=i, name=desc.name, luid=luid,
                                       driver_version=str(version.value) if supported else None,
                                       hardware_id=[desc.vendor, desc.device, desc.subsystem, desc.revision],
                                       dedicated_video_bytes=desc.dedicated_video))
            finally:
                method(adapter, 2, W.ULONG)(adapter)
    finally:
        method(factory, 2, W.ULONG)(factory)
    return result


class OnnxSession:
    def __init__(self, model, device='cpu', adapter=None, threads=4):
        import onnxruntime as ort
        adapter = adapter if device == 'dml' else None
        self.device, self.adapter = device, adapter
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = device != 'dml'
        provider = 'DmlExecutionProvider' if device == 'dml' else 'CPUExecutionProvider'
        if device == 'dml':
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        providers = [(provider, {'device_id': adapter['device_id']})] if device == 'dml' else [provider]
        try:
            self.session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
            self.session.disable_fallback()
            if provider not in self.session.get_providers():
                raise RuntimeError(f'Requested {provider}, got {self.session.get_providers()}')
        except Exception as error:
            if device == 'dml':
                raise GPUError(f'DirectML 初始化失败：{error}') from error
            raise
        inputs = [(i.name, i.shape, i.type) for i in self.session.get_inputs()]
        if inputs != [('frames', [1, 100, 27, 48, 3], 'tensor(uint8)'),
                      ('histograms', [1, 100, 512], 'tensor(float)')]:
            raise ValueError('Expected the validated split FP32 TransNet model')
        self.metadata = dict(onnxruntime=ort.__version__, providers=self.session.get_providers(),
                             device_name=adapter['name'] if adapter else 'CPU', adapter=adapter)

    def predict(self, frames):
        try:
            values = self.session.run(['single', 'many'],
                                      {'frames': frames, 'histograms': color_histograms(frames)})
            if any(v.shape != (1, 100, 1) or not np.isfinite(v).all() or
                   np.min(v) < 0 or np.max(v) > 1 for v in values):
                raise RuntimeError('Invalid ONNX prediction')
            return values
        except Exception as error:
            if self.device == 'dml':
                raise GPUError(f'DirectML 推理失败：{error}') from error
            raise


def select_session(model, mode, frames, threads=4, adapter_id=None, check=lambda: None, cache_path=None):
    """One warmup plus two synchronized real-input trials; require 15% benefit."""
    if mode not in ('auto', 'cpu', 'dml'):
        raise ValueError('ONNX device must be auto, cpu or dml')
    if mode == 'cpu':
        return OnnxSession(model, threads=threads), []
    import onnxruntime as ort
    try:
        hardware = adapters() if 'DmlExecutionProvider' in ort.get_available_providers() else []
    except OSError as error:
        if mode == 'dml':
            raise GPUError(str(error)) from error
        hardware = []
    if adapter_id is not None:
        hardware = [a for a in hardware if a['device_id'] == adapter_id]
    if not hardware:
        if mode == 'dml':
            raise GPUError('没有可用的 DirectML 硬件适配器')
        return OnnxSession(model, threads=threads), []

    # Only a decision is persisted. Each task owns a new private Session.
    identity = dict(policy=1, model=MODEL_SHA256, runtime=ort.__version__, threads=threads,
                    system=platform.version(), cpu=os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
                    hardware=[{k: a.get(k) for k in ('luid', 'name', 'hardware_id', 'driver_version')}
                              for a in sorted(hardware, key=lambda a: a['luid'])])
    cache = Path(cache_path) if cache_path and mode == 'auto' and all(a.get('driver_version') for a in hardware) else None
    if cache:
        try:
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if saved['identity'] == identity:
                choice = saved['choice']
                adapter = next((a for a in hardware if a['luid'] == choice.get('luid')), None)
                if choice['device'] == 'cpu' or (choice['device'] == 'dml' and adapter is not None):
                    check()
                    selected = OnnxSession(model, choice['device'], adapter, threads)
                    selected.predict(frames)  # Validate a new Session before publishing any cut.
                    selected.metadata['selection_cache'] = 'hit'
                    return selected, saved['trials']
        except (OSError, ValueError, KeyError, TypeError, GPUError):
            pass  # A failed cached GPU is re-tested, then uses the existing CPU-process recovery.

    def trial(session):
        check()
        output = session.predict(frames)
        times = []
        for _ in range(2):
            check()
            began = time.perf_counter()
            session.predict(frames)  # CPU outputs make the GPU call synchronous.
            times.append(time.perf_counter() - began)
        return statistics.median(times), output

    best = OnnxSession(model, threads=threads)
    cpu_seconds, reference = trial(best)
    measurements = [dict(device='cpu', seconds=cpu_seconds)]
    best_seconds = cpu_seconds * 0.85 if mode == 'auto' else float('inf')
    for adapter in hardware:
        check()
        candidate = OnnxSession(model, 'dml', adapter, threads)
        seconds, output = trial(candidate)
        error = max(float(np.max(np.abs(a - b))) for a, b in zip(reference, output))
        if error > 1e-5 or any(not np.array_equal(a > .5, b > .5) for a, b in zip(reference, output)):
            raise GPUError('DirectML 设备自检预测与 CPU 不一致')
        measurements.append(dict(device='dml', adapter=adapter, seconds=seconds, max_abs=error))
        if seconds < best_seconds:
            best, best_seconds = candidate, seconds
        del candidate  # Release losing trials immediately; no CPU/last-GPU alias survives.
    best.metadata['selection_cache'] = 'miss' if cache else 'disabled'
    if cache:
        temporary = cache.with_name(cache.name + f'.{os.getpid()}.tmp')
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(dict(identity=identity, choice=dict(device=best.device,
                luid=best.adapter['luid'] if best.adapter else None), trials=measurements)), encoding='utf-8')
            temporary.replace(cache)
        except OSError:
            pass  # A read-only cache does not prevent detection.
        finally:
            temporary.unlink(missing_ok=True)
    return best, measurements
