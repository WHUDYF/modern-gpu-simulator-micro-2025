# GCL ResNet-50 Reproduction Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a Word paper that follows the provided Chinese journal-style reference format while reporting the GCL ResNet-50 reproduction process and its current evidence boundaries.

**Architecture:** A small standard-library Python generator reads the existing GCL artifacts, assembles a Chinese academic paper, and writes a `.docx` file through OpenXML packaging. The generated document keeps measured results and claim limits tied to repository artifacts.

**Tech Stack:** Python standard library, JSON artifacts, WordprocessingML `.docx`.

---

### Task 1: Generate Paper Document

**Files:**
- Create: `scripts/generate_gcl_reproduction_paper_docx.py`
- Create: `docs/reports/gcl-resnet50-reproduction-paper.docx`

- [x] **Step 1: Read source material**

Read the journal-style PDF text, the GNN report excerpt, the K-means report excerpt, and the GCL acceptance artifacts under `artifacts/gcl_resnet50_full_trace_reproduction/`.

- [x] **Step 2: Create a deterministic `.docx` generator**

Create a Python script that writes a complete Word OpenXML package using only the standard library.

- [x] **Step 3: Generate the Word paper**

Run:

```bash
python scripts/generate_gcl_reproduction_paper_docx.py
```

Expected: `docs/reports/gcl-resnet50-reproduction-paper.docx` is created.

- [x] **Step 4: Verify the generated document**

Run:

```bash
unzip -t docs/reports/gcl-resnet50-reproduction-paper.docx
python - <<'PY'
import zipfile
from pathlib import Path
path = Path("docs/reports/gcl-resnet50-reproduction-paper.docx")
with zipfile.ZipFile(path) as z:
    xml = z.read("word/document.xml").decode("utf-8")
for needle in ["ResNet-50", "GCL", "265", "124876", "quantified_no_correctness_claim"]:
    assert needle in xml
print("verified", path, path.stat().st_size)
PY
```

Expected: zip integrity check passes and the key evidence strings are present.

- [x] **Step 5: Commit and push**

Run:

```bash
git add docs/superpowers/plans/2026-06-15-gcl-resnet50-reproduction-paper.md scripts/generate_gcl_reproduction_paper_docx.py docs/reports/gcl-resnet50-reproduction-paper.docx
git commit -m "docs: add gcl resnet50 reproduction paper"
git push origin trace_compressions_gcl
```

Expected: commit is created and pushed to the remote branch.
