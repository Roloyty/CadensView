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

## AI 拉片协议参考 (2026-11-16)

The narrative-analysis protocol in `shotlab/lapian.py` (analysis package layout,
prompt discipline clauses, tolerant JSON import strategy, contact-sheet sizing,
intensity/valence curve modeling and the two-pass deep-dive flow) was designed
after reviewing lapian-notes by bkingfilm: https://github.com/bkingfilm/lapian-notes
(MIT). No lapian-notes source code is bundled or copied; the upstream
implementation is TypeScript for the browser and CadensView's implementation is
independent Python sharing only the interchange-protocol ideas. Differences are
deliberate: scenes anchor to shot ordinals instead of wall-clock seconds, the
manifest carries local measurement evidence (cut points, scale, VLM labels,
motion, loudness, dialogue) absent upstream, and the 1-second blind sampling is
replaced by per-shot representative frames.

## Research workflow references (2026-11-04)

The research features were designed after checking the following projects,
formats and datasets. Their concepts and public specifications informed
CadensView's independently written code; their source code, models, annotations,
video frames and database records are **not** bundled. See
`docs/open-source-reuse.md` for verified URLs, license checks and the exact
scope of reuse.

- ELAN/EAF 3.0 — Max Planck Institute for Psycholinguistics / The Language
  Archive. EAF is a public XML interchange specification; ELAN itself is GPL v3.
  The minimal EAF 3.0 projection is implemented independently: alignable tiers,
  ANNOTATION_VALUE, a controlled vocabulary and typed CROSS_REF_LINK references.
  The full versioned study data remains in the CadensView JSON projection.
- ClipShots — Copyright (c) 2019 tangshitao, MIT. The frame-interval shot
  annotation shape and overlap-based evaluation idea are documented in
  `docs/open-source-reuse.md`; no upstream source or dataset is copied.
- AdA Film-Analytical Annotations / Advene / W3C Web Annotation and Media
  Fragments — FU Berlin AdA project and Advene project. AdA annotations and
  ontology are CC BY-SA 3.0. Typed timed fragments, author/source metadata and
  evidence-versus-interpretation modeling informed the design; no RDF corpus or
  ontology text is included.
- FilmColors/VIAN — GNU GPL v3. Region masks and color-observation workflow
  concepts only; no source code or assets are incorporated.
- Sonic Visualiser — GNU GPL v2. Time-aligned waveform/spectrum presentation
  concept only; no source code is incorporated.
- Cinemetrics, MovieNet/MovieShots, AnimeShooter, Condensed Movies and Blender
  Open Movies — external references or optional user-supplied samples. Their
  respective platform, non-commercial, attribution and original-rightsholder
  terms apply; no videos, frames, annotations, masks or pretrained weights are
  redistributed with CadensView.
- inaSpeechSegmenter — Copyright (c) 2018 Ina (David Doukhan), MIT;
  WhisperX — Copyright (c) 2024 Max Bain, BSD 2-Clause; pyannote-audio — MIT,
  with separate model terms. These are optional external adapters or future
  integration references, not bundled Python dependencies or model weights.
  CadensView only accepts user-provided segments when model, model version,
  source and author are supplied; it never invokes or redistributes a model.

MIT license text for ClipShots (included because its annotation/evaluation
interface informed the local validation format):

> Copyright (c) 2019 tangshitao
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
> THE SOFTWARE.
