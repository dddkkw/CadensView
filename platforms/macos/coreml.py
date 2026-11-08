"""Measured Core ML selection for the unchanged split FP32 ONNX graph."""
from collections import Counter
import json
import os
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import time

import numpy as np


def provider_options(compute_units=None):
    units = compute_units or os.environ.get("CADENSVIEW_COREML_COMPUTE_UNITS", "ALL")
    if units not in ("ALL", "CPUOnly", "CPUAndGPU", "CPUAndNeuralEngine"):
        raise ValueError("Invalid Core ML compute units")
    return dict(ModelFormat="MLProgram", MLComputeUnits=units, RequireStaticInputShapes="1",
                EnableOnSubgraphs="0", AllowLowPrecisionAccumulationOnGPU="0")


def profile_summary(path):
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [event for event in events if event.get("cat") == "Node" and event.get("args", {}).get("provider")]
    counts = Counter(event["args"]["provider"] for event in nodes)
    durations = Counter()
    cpu_ops = Counter()
    for event in nodes:
        provider = event["args"]["provider"]
        durations[provider] += event.get("dur", 0)
        if provider == "CPUExecutionProvider":
            cpu_ops[event["args"].get("op_name", "unknown")] += 1
    return dict(provider_events=dict(counts), coreml_events=counts["CoreMLExecutionProvider"],
                provider_duration_us=dict(durations), cpu_operator_events=dict(cpu_ops),
                hardware_verified=False)


def compare(reference, candidate):
    errors = [float(np.max(np.abs(a - b))) for a, b in zip(reference, candidate)]
    return dict(max_abs=max(errors), heads_max_abs=errors,
                threshold_equal=all(np.array_equal(a > .5, b > .5) for a, b in zip(reference, candidate)))


def select_session(model, mode, frames, threads=4, check=lambda: None):
    import onnxruntime as ort
    from shotlab.detection.onnx_runtime import GPUError, OnnxSession

    if mode not in ("auto", "cpu", "coreml"):
        raise ValueError("macOS ONNX device must be auto, cpu or coreml")
    cpu = OnnxSession(model, threads=threads)
    if mode == "cpu":
        return cpu, []
    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        if mode == "coreml":
            raise GPUError("CoreMLExecutionProvider is unavailable")
        cpu.metadata["coreml_status"] = "provider_unavailable"
        return cpu, []

    def trial(session):
        check()
        reference = session.predict(frames)
        times = []
        for _ in range(2):
            check()
            began = time.perf_counter()
            session.predict(frames)  # CPU NumPy outputs synchronize completion.
            times.append(time.perf_counter() - began)
        return statistics.median(times), reference

    cpu_seconds, reference = trial(cpu)
    trials = [dict(device="cpu", seconds=cpu_seconds)]
    try:
        with TemporaryDirectory(prefix="cadensview-coreml-") as folder:
            began = time.perf_counter()
            candidate = OnnxSession(model, "coreml", threads=threads, profile_prefix=str(Path(folder) / "ort"))
            seconds, output = trial(candidate)
            profile = profile_summary(candidate.session.end_profiling())
            parity = compare(reference, output)
            measurement = dict(device="coreml", seconds=seconds, initialization_and_trial_seconds=time.perf_counter()-began,
                               **profile, **parity)
            trials.append(measurement)
            if not profile["coreml_events"]:
                raise GPUError("Core ML delegated no nodes; CPU-only execution rejected")
            if parity["max_abs"] > 1e-5 or not parity["threshold_equal"]:
                raise GPUError("Core ML self-test differs from CPU predictions")
            candidate.metadata.update(coreml_profile=profile, coreml_self_test=parity,
                                      hardware_execution="CoreML delegated; GPU/ANE placement unverified")
            if mode == "coreml" or seconds < cpu_seconds * .85:
                return candidate, trials
            cpu.metadata["coreml_status"] = "less_than_15_percent_benefit"
    except (OSError, ValueError, RuntimeError) as error:
        # A failed provider may leave native state behind, even during auto
        # selection. Let the coordinator retry in a fresh CPU-only worker.
        raise GPUError(str(error)) from error
    return cpu, trials
