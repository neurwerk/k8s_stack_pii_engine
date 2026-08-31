# Third-Party Notices

This document highlights redistributed components that are central to the PII
Engine images. It is informational, is not legal advice, and does not replace
the license files and notices shipped by each dependency. The complete locked
dependency graph is in `uv.lock`; downstream redistributors should review the
licenses included in the installed distributions.

## Microsoft Presidio

The project installs `presidio-analyzer` and `presidio-anonymizer` 2.2.360.
Presidio 2.2.360 is copyright Microsoft Corporation and is distributed under
the MIT License.

- Source and release: <https://github.com/microsoft/presidio/tree/2.2.360>
- License: <https://github.com/microsoft/presidio/blob/2.2.360/LICENSE>
- Upstream third-party notice: <https://github.com/microsoft/presidio/blob/2.2.360/NOTICE>

## spaCy And Baseline Model Wheels

Presidio uses spaCy, which is copyright ExplosionAI GmbH, spaCy GmbH, Matthew
Honnibal, and contributors and is distributed under the MIT License. The exact
spaCy version is locked in `uv.lock`.

- spaCy source and license: <https://github.com/explosion/spaCy/blob/master/LICENSE>
- spaCy third-party licenses: <https://github.com/explosion/spaCy/blob/master/licenses/3rd_party_licenses.txt>

The release images install these Explosion model wheels directly from the
official spaCy model releases. Model licenses are distinct from the spaCy code
license and are stated in each release's metadata:

| Wheel | Version | Declared license | Attribution and source |
| --- | --- | --- | --- |
| `en_core_web_sm` | 3.8.0 | MIT | Explosion; <https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0> |
| `de_core_news_sm` | 3.8.0 | MIT | Explosion; <https://github.com/explosion/spacy-models/releases/tag/de_core_news_sm-3.8.0> |
| `nl_core_news_sm` | 3.8.0 | CC BY-SA 4.0 | Explosion; <https://github.com/explosion/spacy-models/releases/tag/nl_core_news_sm-3.8.0> |

The Dutch model metadata attributes UD Dutch LassySmall to Gosse Bouma and Gertjan
van Noord, Dutch NER annotations to NLP Town, and UD Dutch Alpino to Daniel
Zeman, Zdenek Zabokrtsky, Gosse Bouma, and Gertjan van Noord. Its declared
license is [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/legalcode).
The English and German release pages identify their training sources and
attributions; consult those pages before redistributing modified model assets.

## PyTorch And CUDA Image Variant

Both image variants install PyTorch 2.6.0 from an official PyTorch wheel index.
PyTorch is copyright its listed authors and contributors and is distributed
under a BSD-style license whose binary-redistribution conditions require the
copyright notice, conditions, and disclaimer in accompanying materials.

- PyTorch 2.6.0 license: <https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE>
- PyTorch 2.6.0 notice: <https://github.com/pytorch/pytorch/blob/v2.6.0/NOTICE>
- Official wheel indexes: <https://download.pytorch.org/whl/>

The `cu124` extra also installs CuPy and the CUDA-enabled PyTorch wheel. CuPy is
distributed under the MIT License:
<https://github.com/cupy/cupy/blob/main/LICENSE>.

The CUDA-enabled PyTorch dependency set redistributes NVIDIA CUDA runtime,
cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuSPARSELt, NCCL, NVRTC,
NVTX, and related binary packages identified in `uv.lock`. These components are
not licensed under this project's MIT License. Their use and redistribution are
subject to the applicable NVIDIA terms, including the CUDA Toolkit EULA and
the notices embedded in the NVIDIA packages:

- CUDA Toolkit EULA and redistribution terms: <https://docs.nvidia.com/cuda/eula/index.html>
- NVIDIA CUDA documentation: <https://docs.nvidia.com/cuda/>

Before publishing or redistributing the CUDA image, the publisher must review
the exact locked NVIDIA package versions, preserve their packaged license and
notice files, and confirm that the intended distribution complies with the
then-current NVIDIA terms. A compatible NVIDIA driver is supplied by the host,
not by this repository.

## External Transformer Bundles

Optional transformer bundles are synchronized at runtime and are not included
in this source repository or its release images. Their model cards, dataset
attributions, acceptable-use restrictions, and licenses are bundle-specific.
Providing credentials to fetch a bundle does not grant redistribution rights;
operators must review and retain the applicable notices independently.
