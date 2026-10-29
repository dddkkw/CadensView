# Third-party attribution

This repository contains selected CadensView core source code, not the complete
desktop application or a Windows binary distribution.

CadensView's original source code is licensed under the MIT License; see LICENSE.

## TransNet V2

The detection integration and external color histogram implementation are based
on TransNet V2 by Tomas Soucek (Tomáš Souček):
https://github.com/soCzech/TransNetV2

The upstream MIT copyright and license are preserved in
licenses/TransNetV2-LICENSE.txt.

Pretrained weights and exported ONNX models are not included. The ONNX adapter
expects the previously validated split FP32 model with the SHA-256 recorded in
shotlab/detection/onnx_runtime.py. Supplying a model is separate from installing
the Python dependencies; upstream models retain their own licensing terms.

Dependencies listed in requirements.txt are installed separately and retain
their own licenses. No third-party runtime binaries, FFmpeg executable, brand
assets, payment QR codes, user projects, or video material are included.
