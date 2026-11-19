#!/usr/bin/env python3
"""Write the Kaggle notebook that runs the xcm-mmm binary, plus its metadata.

The notebook is generated rather than hand edited so the commands it shows stay
in step with the CLI and the dataset slugs. It is written to answer four
questions a reader would otherwise have to guess at:

  * can a dataset binary run at all, given that Kaggle drops the executable bit
    and mounts /kaggle/input read only;
  * does the static build depend on the image's glibc or on the network;
  * how do the fitted artefacts reach the CLI, given that both mounts are read
    only and every command defaults to a path under data/;
  * what does a real workload look like when the tool is the worker inside a
    map over shards, which is the pattern that makes a notebook advanced;
  * which binary and which artefacts produced the numbers, which is answered by
    a provenance record that travels with the file rather than by a hash in a
    card nobody recomputes.

The evaluation and packaging decisions each cite the source that forced them,
listed in full in README/DOCS/RESEARCH.md and repeated at the end of the
notebook.

The kernel is public: a reader can open it, run it and read the figures without
owning the datasets. Every figure is drawn from numbers the binary printed in
the same session, saved as a PNG under charts/ and displayed inline, so the
pictures cannot drift from the run that produced them.

    python3 scripts/kaggle-notebook.py            # stage dist/kaggle-kernel
    python3 scripts/kaggle-notebook.py --push     # stage, then push the kernel

The two dataset paths are read from XCM_INPUT and XCM_WORK, so the same cells
can be executed off Kaggle against local copies before they are pushed. The
cells themselves live in nbgen/setup.py and nbgen/text.py, in kernel order.
"""
from __future__ import annotations

import sys

from nbgen import main

if __name__ == "__main__":
    sys.exit(main())
