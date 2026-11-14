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
SELECTION_POLICY = 2


class GPUError(RuntimeError):
    """The coordinator may retry once in a fresh CPU process."""


OUTPUT_ROUNDOFF = float(np.finfo(np.float32).eps)


def validate_outputs(values, device, providers, diagnostics=None):
    """Validate both FP32 sigmoid heads; diagnostics contain no media or paths."""
    if len(values) != 2:
        raise RuntimeError(f'ONNX output count: expected=2 actual={len(values)} backend={device}')
    outputs = []
    for name, value in zip(('single', 'many'), values):
        finite = np.isfinite(value)
        valid = value[finite]
        stats = dict(head=name, backend=device, providers=providers, shape=list(value.shape),
                     dtype=str(value.dtype), min=float(valid.min()) if valid.size else None,
                     max=float(valid.max()) if valid.size else None,
                     nan=int(np.isnan(value).sum()), inf=int(np.isinf(value).sum()),
                     below_zero=int((value < 0).sum()), above_one=int((value > 1).sum()),
                     onnxruntime=(diagnostics or {}).get('onnxruntime'), architecture=platform.machine(),
                     model_sha256=MODEL_SHA256)
        reason = ('shape' if value.shape != (1, 100, 1) else 'dtype' if value.dtype != np.float32
                  else 'nonfinite' if not finite.all()
                  else 'range' if stats['min'] < -OUTPUT_ROUNDOFF or stats['max'] > 1 + OUTPUT_ROUNDOFF else None)
        if reason:
            raise RuntimeError('ONNX output ' + reason + ': ' + json.dumps(stats, allow_nan=False))
        if stats['below_zero'] or stats['above_one']:
            # ORT 1.20.1 ARM CPU Sigmoid's polynomial reaches +/- one FP32
            # epsilon beyond its mathematical range. Verified by the native
            # sigmoid probe. Endpoints are far from .5: responses cannot change.
            if diagnostics is not None:
                record = diagnostics.setdefault('output_roundoff', {}).setdefault(name, dict(values=0, min=0., max=1.))
                record['values'] += stats['below_zero'] + stats['above_one']
                record['min'] = min(record['min'], stats['min'])
                record['max'] = max(record['max'], stats['max'])
            value = np.clip(value, 0, 1)
        outputs.append(value)
    return outputs


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
                # Filter software adapters and identical DXGI identities, keeping
                # raw ordinals. A model/PCI ID does not identify a physical board.
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


def free_video_bytes(adapter):
    """Dedicated VRAM not used by any process on this adapter, or None if unknown.

    DXGI budgets are per process; only the system GPU memory counters
    (Windows 10 1709+) include the desktop and other windows' workers.
    """
    if sys.platform != 'win32':
        return None
    class Item(C.Structure):
        _fields_ = [('name', W.LPWSTR), ('status', W.DWORD), ('value', C.c_longlong)]
    pdh = C.WinDLL('pdh.dll')
    query, counter = C.c_void_p(), C.c_void_p()
    if pdh.PdhOpenQueryW(None, None, C.byref(query)):
        return None
    try:
        if (pdh.PdhAddEnglishCounterW(query, '\\GPU Adapter Memory(*)\\Dedicated Usage', None, C.byref(counter))
                or pdh.PdhCollectQueryData(query)):
            return None
        size, count = W.DWORD(), W.DWORD()
        pdh.PdhGetFormattedCounterArrayW(counter, 0x400, C.byref(size), C.byref(count), None)  # PDH_FMT_LARGE
        items = (C.c_byte * size.value)()
        if pdh.PdhGetFormattedCounterArrayW(counter, 0x400, C.byref(size), C.byref(count), items):
            return None
        hi, lo = adapter['luid'].split(':')
        prefix = f'luid_0x{hi}_0x{lo}_'
        used = sum(item.value for item in C.cast(items, C.POINTER(Item))[:count.value]
                   if item.name.lower().startswith(prefix) and item.status in (0, 1))
        return max(0, adapter['dedicated_video_bytes'] - used)
    finally:
        pdh.PdhCloseQuery(query)


class OnnxSession:
    def __init__(self, model, device='cpu', adapter=None, threads=4, profile_prefix=None, provider_options=None):
        import onnxruntime as ort
        adapter = adapter if device == 'dml' else None
        self.device, self.adapter = device, adapter
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = device != 'dml'
        provider = {'cpu': 'CPUExecutionProvider', 'dml': 'DmlExecutionProvider', 'coreml': 'CoreMLExecutionProvider'}[device]
        if profile_prefix:
            options.enable_profiling = True
            options.profile_file_prefix = profile_prefix
        if device == 'dml':
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        providers = [(provider, {'device_id': adapter['device_id']})] if device == 'dml' else [provider]
        if device == 'coreml':
            from platforms.macos.coreml import provider_options as coreml_options
            providers = [(provider, provider_options or coreml_options()), 'CPUExecutionProvider']
        try:
            session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
            session.disable_fallback()
            active = session.get_providers()
            if provider not in active:
                raise RuntimeError(f'Requested {provider}, got {active}')
        except Exception as error:
            if device == 'dml':
                raise GPUError(f'DirectML 初始化失败：{error}') from error
            if device == 'coreml':
                raise GPUError(f'Core ML initialization failed: {error}') from error
            raise
        inputs = [(i.name, i.shape, i.type) for i in session.get_inputs()]
        if inputs != [('frames', [1, 100, 27, 48, 3], 'tensor(uint8)'),
                      ('histograms', [1, 100, 512], 'tensor(float)')]:
            raise ValueError('Expected the validated split FP32 TransNet model')
        self.session = session
        self.metadata = dict(onnxruntime=ort.__version__, providers=active,
                             device_name=adapter['name'] if adapter else 'Apple Core ML' if device == 'coreml' else 'CPU', adapter=adapter)

    def predict(self, frames):
        try:
            values = self.session.run(['single', 'many'],
                                      {'frames': frames, 'histograms': color_histograms(frames)})
            return validate_outputs(values, self.device, self.session.get_providers(), self.metadata)
        except Exception as error:
            if self.device == 'dml':
                raise GPUError(f'DirectML 推理失败：{error}') from error
            if self.device == 'coreml':
                raise GPUError(f'Core ML inference failed: {error}') from error
            raise


def select_session(model, mode, frames, threads=4, adapter_id=None, check=lambda: None, cache_path=None):
    """One warmup plus two synchronized real-input trials; require 15% benefit."""
    if sys.platform == 'darwin':
        from platforms.macos.coreml import select_session as select_coreml
        return select_coreml(model, mode, frames, threads, check)
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

    if mode == 'auto':
        # One probe also excludes low-VRAM integrated adapters when a larger
        # GPU is present. Distinct LUIDs may still represent identical models.
        hardware = [max(hardware, key=lambda a: (a['dedicated_video_bytes'], -a['device_id']))]

    # Only a decision is persisted. Each task owns a new private Session.
    identity = dict(policy=SELECTION_POLICY, model=MODEL_SHA256, runtime=ort.__version__, threads=threads,
                    system=platform.version(), cpu=os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
                    hardware=[{k: a.get(k) for k in ('luid', 'name', 'hardware_id', 'driver_version', 'dedicated_video_bytes')}
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
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Corrupt decision data can be replaced. GPUError reaches the CPU-worker retry.

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
