# Third-Party Licenses

KinoMLX's own code is MIT-licensed (see [LICENSE](./LICENSE)). This file
lists the third-party projects whose code KinoMLX adapts or whose
published implementations it referenced, together with their license
texts. When code is borrowed verbatim, the originating file in
`kinomlx/` also carries a short header noting the source repository and
license.

Model weights are licensed separately and are not covered by any entry
here; see the "Attribution and license" section of the [README](./README.md).

---

## MLX

- **Source:** <https://github.com/ml-explore/mlx>
- **Used in:** runtime dependency declared in `pyproject.toml`, plus the STEEL
  attention primitives of the FP16 dense, Sol-Attn and neighborhood attention kernels
  under `kinomlx/models/ltx2/kernels/metal/`, and the STEEL GEMM fragment layout, block
  product and loader index scheme of the low-rank fusion kernel in
  `kinomlx/kernels/metal/low_rank_fusion.metal`. The attention resources use MLX source
  commit `2165dc08d7b33258260aa849d39f087d50e62962`; the fusion resource follows MLX
  v0.32.3, commit `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`.
- **License:** MIT

```text
MIT License

Copyright (c) 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## mlx-lm

- **Source:** <https://github.com/ml-explore/mlx-lm>
- **Used in:** Gemma 3 and Gemma 4 text backbones under
  `kinomlx/models/ltx2/text_encoder/`, adapted to expose the hidden states
  consumed by LTX, plus structural reference for the allocation-light Qwen
  text graph under `kinomlx/models/krea2/text_encoder/`. The inspected source
  revision was `77c33b14373ac70d7abd6f82af15962852adadbb`.
- **License:** MIT

```text
MIT License

Copyright (c) 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Transformers (Hugging Face)

- **Source:** <https://github.com/huggingface/transformers>
- **Copyright:** The Hugging Face team and contributors
- **Used in:** referenced implementation for the Gemma 3 and Gemma 4 text-encoder
  semantics (configuration schema, attention and RoPE behavior)
  implemented under `kinomlx/models/ltx2/text_encoder/`, plus the Qwen3-VL
  configuration, grouped-query attention, MRoPE, text-only tensor selection,
  and tokenizer semantics implemented under `kinomlx/models/krea2/`. The Krea
  Qwen behavior was inspected at the Transformers 4.57.1 source revision
  `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc`.
  The native Qwen3-VL vision graph, visual patch merging, multimodal positions,
  and DeepStack language integration additionally reference
  `ac3244569528944b9d5773cafea525cd8a8b63de`. The implementation uses native MLX
  operations and separately bound, allocation-light model components.
- **License:** Apache-2.0 (full text in the shared section below)

---

## Krea 2

- **Source:** <https://github.com/krea-ai/krea-2>
- **Used in:** referenced implementation for Krea 2 Raw/Turbo graph metadata,
  native checkpoint key semantics, MMDiT topology, grouped-query attention,
  three-axis RoPE, mask and sequence-padding behavior, modulation and residual
  order, component selection, recipe identity, prompt layout, and hidden-state
  selection implemented under `kinomlx/models/krea2/`. The inspected source revision was
  `db3984fbc6e13b34c0064990fc2d95ac64d00058`.
- **License:** Apache-2.0 (full text in the shared section below)

The Krea model weights are separate external artifacts under their own terms;
this source-code notice does not license or redistribute them.

---

## diffusers (Hugging Face)

- **Source:** <https://github.com/huggingface/diffusers>
- **Copyright:** the LTX-2 modules are marked "Copyright 2025 The
  Lightricks team and The HuggingFace Team"; the FLUX.2 transformer is marked
  "Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team.
  All rights reserved."
- **Used in:** referenced implementation for the LTX-2 model semantics
  (transformer, video and audio VAEs, vocoder, text connectors, and
  spatial latent upsampler) implemented under `kinomlx/models/ltx2/`, plus
  the RF ancestral Euler sampler implemented under `kinomlx/samplers/`, and
  the Krea 2 Diffusers component packaging, canonical transformer keys,
  equivalent MMDiT configuration and forward semantics, Qwen text selection,
  and Qwen Image decoder topology, causal still-frame specialization,
  normalization, attention, upsampling, and tiling behavior under
  `kinomlx/models/krea2/`. The inspected source revision was
  `c1bf18c92c6285334adcaac7e75ef8946a227f49`.
  The first-frame Qwen Image encoder, spatial downsampling, and posterior-mode
  encoding under `kinomlx/models/krea2/vae/` additionally reference
  `bda3386ddcaa31bc2853567a95e815629c8022b7`. The encoder is a native MLX
  implementation with allocation-light weights and a deterministic RGB-to-latent API.
  Klein component semantics under `kinomlx/models/flux2/` reference the same
  `bda3386ddcaa31bc2853567a95e815629c8022b7` revision: Qwen text selection,
  VAE packing and normalization, double/single transformer blocks, four-axis
  rotary positions, modulation, target-only reference editing, reference-only KV
  attention and fixed-reference modulation, and the empirical
  flow-matching Euler schedule. These are native MLX implementations; Diffusers
  is not a runtime dependency.
- **License:** Apache-2.0 (full text in the shared section below)

---

## PEFT (Hugging Face)

- **Source:** <https://github.com/huggingface/peft>
- **Copyright:** The Hugging Face team and contributors
- **Used in:** static adapter checkpoint schemas, scaling and inference equations
  implemented independently under `kinomlx/adapters/`. The inspected source
  revision was `004fd0369e770dc857b895050f65bb40f7bca89c`. PEFT is not a runtime dependency.
- **License:** Apache-2.0 (full text in the shared section below)

---

## PyWavelets coefficient data

- **Source:** <https://github.com/PyWavelets/pywt>, through PEFT's WaveFT coefficient table.
- **Used in:** reconstruction low-pass coefficient data in
  `kinomlx/adapters/wavelet_filters.py`. The MLX synthesis implementation
  does not include wavelet-toolbox code or add a PyWavelets dependency.
- **License:** MIT

```text
Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
Copyright (c) 2012- The PyWavelets Developers <https://github.com/PyWavelets/pywt>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## PyTorch

- **Source:** <https://github.com/pytorch/pytorch>
- **Used in:** the optional `torch-mps` normal-noise backend under
  `kinomlx/samplers/noise.py`, adapted from `c10/metal/random.h` and
  `aten/src/ATen/native/mps/kernels/Distributions.metal` at source commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- **License:** BSD-style license

```text
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## GMNet

- **Source:** <https://github.com/qtlark/GMNet>
- **Copyright:** (c) 2025 Yinuo Liao
- **Used in:** referenced implementation for the GMNet
  inverse-tone-mapping generator ("Learning Gain Map for Inverse Tone
  Mapping", Liao et al., ICLR 2025) implemented under
  `kinomlx/models/gmnet/`. No upstream code is bundled; the published
  checkpoints convert 1:1 into the local weights directory (see
  `kinomlx/models/gmnet/weights/Attribution.md`).
- **License:** MIT

```text
MIT License

Copyright (c) 2025 Yinuo Liao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## comfyui-krea2edit

- **Source:** <https://github.com/lbouaraba/comfyui-krea2edit>
- **Revision:** `86f886dac23013d88996e3a2e99093ba44d322fb`
- **Used in:** Krea image-edit reference geometry, centered reference positions,
  attention boosting, and grounded conversation template under
  `kinomlx/models/krea2/`. Source-mask cropping is a KinoMLX usability adaptation.
- **License:** Apache-2.0 (full text in the shared section below)

---

## Sana Sol-Engine (Sol-Attn)

- **Source:** <https://github.com/NVlabs/Sana> (`sol-engine` branch,
  `techniques/sparse_backends/sol_attn`, read at commit
  `62da34b14a53d7289e82fa56692f34e96e089bee`)
- **Used in:** algorithmic reference for the sparse video self-attention under
  `kinomlx/models/ltx2/kernels/sol_attention.py` and its Metal resources: 64-token
  block routing by mean query and mean key against a per-row mean plus tau standard
  deviations with a diagonal key variance, exact computation of routed and neighboring
  blocks, and the length-weighted mean-key approximation of the rest. The kernel is a
  new implementation on the MLX STEEL primitives noted above; no Sol-Attn source is
  copied.
- **License:** Apache-2.0 (full text in the shared section below)

---

## Apache License 2.0 (shared text)

The complete license text below applies to the Transformers, Krea 2,
diffusers, PEFT, comfyui-krea2edit, and Sana Sol-Engine entries above.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, Any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
