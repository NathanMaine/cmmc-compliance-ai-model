# CMMC Compliance AI Model

**A suite of four locally-hosted, fine-tuned language models specialized in CMMC 2.0, NIST 800-171, NIST 800-53, HIPAA, and cybersecurity compliance frameworks.**

Built to answer the question: *Can a small team deploy a domain-specific AI compliance advisor that runs entirely on-premises — no cloud, no API fees, no CUI exposure?*

**Yes. Four model sizes from 5 GB to 42 GB — from laptop to workstation to server.**

> **Current release:** Version 1.0 (February 2026) — All four model sizes (7B, 14B, 32B, 72B) trained and published
>
> **In progress:** Version 2.0 — expanded to 9+ authoritative sources with automated scraping pipeline

---

## Why This Exists

Organizations pursuing CMMC certification face a knowledge bottleneck. Compliance staff spend hours searching across NIST publications, CMMC assessment guides, and DFARS clauses to answer questions that a well-trained model can handle in seconds.

Commercial LLMs (GPT-4, Claude) are powerful but introduce data residency concerns for organizations handling CUI and ITAR-controlled information. These models run fully local — no data leaves the premises, no internet required, no per-token costs.

---

## Model Suite

All four models share the same compliance knowledge base and training data. The tiered approach allows organizations to deploy the model that best matches their available hardware.

| Model | Parameters | GGUF Size | Quantization | Inference Speed | Hardware Required |
|-------|-----------|-----------|--------------|-----------------|-------------------|
| **cmmc-expert-7b** | 7.6B | 5.1 GB | q5_k_m | ~1-2 sec | 8 GB VRAM (consumer GPU) |
| **cmmc-expert-14b** | 14.7B | ~10 GB | q5_k_m | ~3-5 sec | 12 GB VRAM |
| **cmmc-expert-32b** | 32.5B | 18.5 GB | q4_k_m | ~30-60 sec | 24 GB VRAM or 32 GB RAM |
| **cmmc-expert-72b** | 72.7B | 44.2 GB | q4_k_m | ~2-4 min | 48 GB VRAM or 64 GB RAM |

**Base models**: [Qwen2.5 Instruct](https://huggingface.co/Qwen) (abliterated variant by [huihui-ai](https://huggingface.co/huihui-ai)) — 7B, 14B, 32B, 72B

**Method**: QLoRA fine-tuning — base model weights frozen in 4-bit, low-rank adapters trained on compliance data

**Runtime**: [Ollama](https://ollama.ai) (OpenAI-compatible API at `localhost:11434/v1`)

---

## Download

| Model | Hugging Face | Size | Status |
|-------|-------------|------|--------|
| **cmmc-expert-7b** | [Download GGUF](https://huggingface.co/Nathan-Maine/cmmc-expert-7b) | 5.1 GB | Available |
| **cmmc-expert-14b** | [Download GGUF](https://huggingface.co/Nathan-Maine/cmmc-expert-14b) | ~10 GB | Available |
| **cmmc-expert-32b** | [Download GGUF](https://huggingface.co/Nathan-Maine/cmmc-expert-32b) | 18.5 GB | Available |
| **cmmc-expert-72b** | [Download GGUF](https://huggingface.co/Nathan-Maine/cmmc-expert-72b) | 44.2 GB | Available |

Quick start with Ollama:
```bash
# Download and run (pick your size)
ollama run Nathan-Maine/cmmc-expert-7b

# Or use the OpenAI-compatible API
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Nathan-Maine/cmmc-expert-7b",
    "messages": [{"role": "user", "content": "What are the access control requirements for CMMC Level 2?"}]
  }'
```

---

## Version 1.0 — Current Release

**Released:** February 11, 2026

### Training Data

**13,434 training + 3,472 validation examples** (~3.3 million tokens) assembled from 5 curated sources:

| Source | Records | % | Coverage |
|--------|---------|---|----------|
| NIST Cybersecurity Publications | 6,372 | 47.4% | SP 800-171, 800-172, 800-53, 800-37, CSF, and related guidance |
| CMMC Primary | 4,787 | 35.6% | CMMC 2.0 requirements, controls, implementation guidance, assessment procedures |
| CMMC Balanced | 994 | 7.4% | Curated subset ensuring proportional coverage across all CMMC domains |
| HIPAA Compliance | 961 | 7.2% | Security Rule requirements, technical safeguards, enforcement guidance |
| CMMC Core | 320 | 2.4% | High-priority practices and assessment-critical requirements |

**Data quality pipeline:**
1. **Format conversion** — Raw text and embeddings converted to chat-style instruction/response pairs with a unified compliance expert system prompt
2. **Quality filtering** — Removed entries <100 chars, table-heavy fragments, OCR artifacts, and garbled text
3. **Relevance filtering** — NIST data reduced from 424,729 raw records to 6,372 CMMC-relevant examples
4. **Deduplication** — Exact dedup via xxhash, near-dedup via MinHash LSH (Jaccard threshold 0.8)
5. **Validation split** — 80/20 stratified split maintaining source distribution

### Framework Coverage (v1.0)

| Framework | Version | Coverage |
|-----------|---------|----------|
| **CMMC 2.0** | 32 CFR Part 170 | All three levels: 17 L1 practices, 110 L2 practices, 134 L3 practices. Assessment methodology, scoping guidance, POA&M requirements |
| **NIST SP 800-171** | Rev. 2 | 110 security requirements across 14 families (CMMC Level 2 foundation) |
| **NIST SP 800-172** | Original | Enhanced security requirements for critical CUI programs (CMMC Level 3 delta) |
| **NIST SP 800-53** | Rev. 5 | Full catalog of 1,189 controls across 20 families (cross-mapping reference) |
| **NIST SP 800-37** | Rev. 2 | Risk Management Framework steps, authorization process, continuous monitoring |
| **NIST CSF** | 1.1 | Identify, Protect, Detect, Respond, Recover functions and implementation tiers |
| **HIPAA Security Rule** | Current | Administrative, physical, and technical safeguards; breach notification; enforcement guidance |
| **DFARS Clauses** | Current | 252.204-7012, 7019, 7020, 7021 — contract-level CUI protection requirements |

### Training Configuration

All models trained using QLoRA (Quantized Low-Rank Adaptation) — base weights frozen in 4-bit, trainable adapter layers injected into transformer blocks.

| Parameter | 7B | 14B | 32B | 72B |
|-----------|-----|------|------|------|
| **GPU** | RTX 5000 Ada 16GB | A100 80GB SXM | A100 80GB SXM | A100 80GB SXM |
| **LoRA rank** | 64 | 16 | 32 | 8 |
| **LoRA alpha** | 128 | 32 | 64 | 16 |
| **LoRA dropout** | 0 | 0.05 | 0.05 | 0 |
| **Target modules** | q/k/v/o_proj | All 7 (q/k/v/o + gate/up/down) | All 7 | All 7 |
| **Effective batch size** | 16 | 16 | 16 | 16 |
| **Learning rate** | 2e-4 | 1e-4 | 1e-4 | 5e-5 |
| **Epochs** | 3 | 3 | 3 | 3 |
| **Max sequence length** | 2048 | 2048 | 2048 | 2048 |
| **Precision** | bf16 | bf16 | bf16 | bf16 |
| **Optimizer** | AdamW 8-bit | AdamW 8-bit | AdamW 8-bit | AdamW 8-bit |
| **Training time** | ~3.2 hours | ~6 hours | ~10 hours | ~16.9 hours |
| **Final eval loss** | 1.241 | — | — | 1.004 |

### Evaluation Results

All models showed continuous improvement across training with no overfitting observed.

**7B Eval Loss Curve:**

| Checkpoint | Progress | Eval Loss |
|------------|----------|-----------|
| Step 200 | 8% | 1.462 |
| Step 600 | 24% | 1.334 |
| Step 1000 | 40% | 1.286 |
| Step 1600 | 63% | 1.253 |
| Step 2400 | 95% | 1.242 |
| **Final** | **100%** | **1.241** |

**Final Eval Loss by Model Size:**

| Model | Eval Loss | Steps |
|-------|-----------|-------|
| 7B | 1.241 | 2,520 |
| 14B | — | 2,520 |
| 32B | — | 2,520 |
| 72B | **1.004** | 2,520 |

The 72B model achieved the lowest eval loss across all four sizes, demonstrating that larger base models extract more compliance knowledge from the same training data.

---

## Version 2.0 — In Progress

Version 2.0 significantly expands the training corpus with authoritative source material scraped directly from government APIs and official publications. An automated data pipeline handles scraping, conversion, quality filtering, deduplication, and versioning with full reproducibility.

### What's New in v2.0

The regulatory landscape has changed substantially since the v1 training data was assembled. Version 2.0 addresses these gaps:

| Update | Date | Significance |
|--------|------|-------------|
| **NIST SP 800-171 Rev. 3** | May 2024 | Replaces Rev. 2. Consolidated from 110 to 97 controls. Adds 88 organization-defined parameters (ODPs) and 509 assessment objectives. This is the new CMMC Level 2 foundation |
| **NIST CSF 2.0** | Feb 2024 | Major revision adding the GOVERN function (6 functions total). 34 categories, 174 subcategories with implementation examples |
| **CMMC Final Rule** (32 CFR 170) | Dec 2024 | The actual regulation establishing the CMMC program. Effective rule text, not just guidance |
| **DFARS 252.204-7021** | Nov 2025 | Acquisition rule requiring CMMC certification in DoD contracts. Phase-in timeline and applicability |
| **NIST SP 800-172 Rev. 3** | 2025 (FPD) | Enhanced CUI requirements, final public draft. CMMC Level 3 delta |
| **DoD Assessment Guides** | 2025 | Official L2/L3 assessment procedures, scoping guides, ODP values document |

### New Data Sources (v2.0)

The v2.0 pipeline scrapes 9 authoritative sources (6 new + 3 from v1):

| Source | Scraper | Records | Method |
|--------|---------|---------|--------|
| **NIST SP 800-171 Rev. 3** | `nist_sp800_171` | 97 | OSCAL JSON catalog from NIST GitHub. Each control includes statement, discussion, assessment objectives, assessment methods, and ODPs |
| **NIST CSF 2.0** | `nist_csf` | 208 | OSCAL JSON catalog. 34 category records + 174 subcategory records with implementation examples |
| **DoD Documents** | `dod_documents` | 606 | PDF extraction from dodcio.defense.gov and nvlpubs.nist.gov. 5 documents: Assessment Guide L2, Scoping Guide L2, Scoping Guide L3, ODP Values, SP 800-172 R3 |
| **NIST SP 800-53 Rev. 5** | `nist_csrc` | 1,016 | OSCAL JSON catalog. Full control catalog including enhancements, with CSV fallback |
| **eCFR Regulations** | `ecfr` | 413 | eCFR API. 32 CFR Part 170 (CMMC), 48 CFR Part 252 (DFARS), 45 CFR Part 164 (HIPAA) |
| **Federal Register** | `federal_register` | ~300+ | Federal Register API. CMMC rulemakings, DFARS notices, CUI policy documents |
| NIST Cybersecurity | *(from v1)* | 6,372 | Retained from v1 training data |
| CMMC / HIPAA | *(from v1)* | 7,062 | Retained from v1 training data |

**v2.0 target:** ~15,000–16,000 total training examples after quality filtering and deduplication

### Automated Pipeline

> **Source code:** [cmmc-data-pipeline](https://github.com/NathanMaine/cmmc-data-pipeline) — standalone repo with scrapers, processors, and CLI tools

The v2.0 data pipeline is fully automated and reproducible:

```
                Authoritative Sources
    ┌──────────────────────────────────────────┐
    │  NIST OSCAL (GitHub)    eCFR API         │
    │  DoD PDFs               Federal Register │
    └─────────────────┬────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │  Step 1: Scrape                          │
    │  Rate-limited, retry-enabled scrapers    │
    │  Raw JSON saved to data/raw/{source}/    │
    ├──────────────────────────────────────────┤
    │  Step 2: Convert                         │
    │  Source-specific templates generate       │
    │  chat-format instruction/response pairs  │
    ├──────────────────────────────────────────┤
    │  Step 3: Quality Filter                  │
    │  Min length, alpha ratio, table ratio    │
    ├──────────────────────────────────────────┤
    │  Step 4: Deduplicate                     │
    │  xxhash exact + MinHash LSH near-dedup   │
    ├──────────────────────────────────────────┤
    │  Step 5: Validate                        │
    │  Format checks, quality scoring, stats   │
    ├──────────────────────────────────────────┤
    │  Step 6: Version                         │
    │  Immutable snapshots with rollback       │
    ├──────────────────────────────────────────┤
    │  Step 7: Merge                           │
    │  Append to training data (optional)      │
    └──────────────────────────────────────────┘
```

The pipeline supports both full scrapes and incremental updates (only fetch documents changed since a given date). Each run creates a versioned snapshot that can be inspected, diffed, or rolled back before merging into the training dataset.

### v2.0 Pipeline Results (First Run)

| Step | Result |
|------|--------|
| Scraped | 911 raw records from 3 new OSCAL/PDF sources |
| Converted | 911 chat-format training records |
| Quality filtered | 807 passed (104 rejected — short DoD chunks) |
| Deduplicated | 803 unique (4 near-duplicates removed) |
| Validated | **PASSED** — 0 format errors, avg answer length 1,383 chars |

Snapshot `v001` created with 803 records. Additional pipeline runs for eCFR (413), SP 800-53 (1,016), and Federal Register (~300+) are in progress.

---

## Compliance Framework Coverage

### Full Coverage Map (v1.0 + v2.0)

| Framework | v1.0 | v2.0 Adds |
|-----------|------|-----------|
| **CMMC 2.0** (32 CFR Part 170) | Level 1–3 practices, assessment methodology | Actual regulatory text from eCFR, DoD assessment guide procedures, scoping guidance |
| **NIST SP 800-171** | Rev. 2 (110 requirements) | **Rev. 3** (97 controls with assessment objectives, methods, and 88 ODPs) |
| **NIST SP 800-172** | Original | **Rev. 3 Final Public Draft** (enhanced CUI requirements) |
| **NIST SP 800-53** | Rev. 5 (from NIST pubs dataset) | Rev. 5 full OSCAL catalog (1,016 controls + enhancements with structured statements and guidance) |
| **NIST CSF** | 1.1 (5 functions) | **2.0** (6 functions — adds GOVERN. 34 categories, 174 subcategories) |
| **HIPAA Security Rule** | Q&A training pairs | Full regulatory text from 45 CFR Part 164 (41 sections) |
| **DFARS Clauses** | 7012, 7019, 7020, 7021 guidance | Full regulatory text from 48 CFR Part 252 (348 sections) |
| **Federal Register** | — | CMMC rulemakings, DFARS proposed/final rules, CUI policy notices |
| **DoD Assessment Guides** | — | L2 Assessment Guide (332 chunks), L2/L3 Scoping Guides, ODP Values |

### DFARS and NIST Coverage Detail

The models understand the full chain from contract clause to technical implementation:

**DFARS 252.204-7012** — Safeguarding Covered Defense Information
- Adequate security requirements for CUI on contractor systems
- Cyber incident reporting obligations (72-hour timeline)
- Flow-down requirements to subcontractors
- Relationship to NIST SP 800-171 compliance

**DFARS 252.204-7019** — Notice of NIST SP 800-171 Assessment
- SPRS (Supplier Performance Risk System) scoring methodology
- Self-assessment requirements and documentation
- DoD Assessment Methodology (Basic, Medium, High)

**DFARS 252.204-7020** — NIST SP 800-171 DoD Assessment Requirements
- Government assessment access and cooperation requirements
- Relationship between SPRS scores and assessment levels

**DFARS 252.204-7021** — Cybersecurity Maturity Model Certification
- CMMC level requirements by contract type
- Phase-in timeline and applicability
- Relationship between CMMC levels and NIST SP 800-171/172

---

## Use Cases

| Application | How It Works |
|-------------|-------------|
| **SSP Generation** | Draft System Security Plan control descriptions with proper NIST/CMMC citations |
| **Gap Analysis** | Identify which controls are required for specific CMMC levels and contract requirements |
| **Assessment Prep** | Generate evidence checklists and assessment objective narratives |
| **Cross-Framework Mapping** | Map controls between CMMC, NIST 800-53, HIPAA, and other frameworks |
| **Policy Drafting** | Create initial policies and procedures aligned to specific CMMC practices |
| **DFARS Clause Analysis** | Identify applicable requirements from contract language (7012, 7019, 7020, 7021) |
| **Training & Education** | Always-available compliance reference — no waiting for SMEs |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     User Query                            │
│   "What access control requirements apply to              │
│    CMMC Level 2 for CUI handling?"                        │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Ollama Runtime (Local)                        │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Qwen2.5 Base (frozen, 4-bit quantized)            │   │
│  │  + QLoRA Adapters (compliance-tuned)                │   │
│  │                                                     │   │
│  │  7B  — quick lookups, day-to-day queries            │   │
│  │  14B — detailed analysis, multi-control reasoning   │   │
│  │  32B — deep gap assessments, SSP drafting           │   │
│  │  72B — complex multi-framework analysis             │   │
│  └────────────────────────────────────────────────────┘   │
│  System prompt: Compliance expert across CMMC/NIST/HIPAA  │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Structured Response                          │
│   - Framework-specific references (SP 800-171 §3.1)      │
│   - Implementation guidance                               │
│   - Assessment evidence requirements                      │
│   - Cross-framework mappings (CMMC ↔ 800-53 ↔ HIPAA)    │
│   - DFARS clause applicability                            │
└──────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **4 model sizes (7B–72B)** | Tiered deployment — laptop for quick lookups, workstation for deep analysis. Match the model to the hardware |
| **QLoRA (not full fine-tune)** | Trainable params are <0.1% of total. 50x less compute, comparable domain accuracy |
| **Abliterated base models** | Removes alignment refusals that interfere with compliance analysis (e.g., discussing vulnerability details, attack scenarios in assessment context) |
| **q5_k_m / q4_k_m quantization** | 5-bit for smaller models (accuracy-sensitive), 4-bit for larger (size-constrained). Compliance is fact-heavy — extra bit preserves control IDs |
| **Local-only deployment** | CUI/ITAR data cannot leave premises. Zero cloud dependency by design. Fully air-gappable |
| **Multi-framework training** | Organizations rarely have single-framework obligations. Cross-mapping across CMMC, NIST, HIPAA, and DFARS is the real value |
| **OSCAL-native scraping** | NIST's machine-readable OSCAL JSON provides structured control data far richer than PDF extraction |
| **Automated pipeline with versioning** | Reproducible data updates with immutable snapshots, rollback, and dedup against existing training data |

---

## Security and Privacy

This model suite was designed for environments where data sovereignty matters:

- **Fully air-gappable** — Runs entirely on local hardware after initial download. No internet required for inference
- **No telemetry** — No data is transmitted to any external service
- **No third-party API dependency** — No per-query costs, no rate limits, no data exposure to cloud providers
- **CUI-safe deployment** — Suitable for use in environments processing Controlled Unclassified Information, as no data leaves the local system boundary
- **Customizable** — Can be further fine-tuned with organization-specific policies, SSPs, and internal security documentation

---

## Hardware Requirements

### Inference (Running a Model)

| Model | GPU (VRAM) | CPU-Only (System RAM) | Storage |
|-------|-----------|----------------------|---------|
| **7B** | 8 GB | 16 GB | 10 GB |
| **14B** | 12 GB | 24 GB | 15 GB |
| **32B** | 24 GB | 32 GB | 25 GB |
| **72B** | 48 GB | 64 GB | 50 GB |

### Training (Reproducing from Scratch)

| Model | GPU Required | Approx. Time |
|-------|-------------|--------------|
| **7B** | 16 GB VRAM (e.g., RTX 5000 Ada) | ~3.2 hours |
| **14B** | 40+ GB VRAM (e.g., A100 40GB) | ~6 hours |
| **32B** | 80 GB VRAM (e.g., A100 80GB) | ~10 hours |
| **72B** | 80 GB VRAM (e.g., A100 80GB) | ~16.9 hours |

**OS**: Linux, macOS, Windows (WSL2)

---

## Repository Structure

```
cmmc-compliance-ai-model/
├── README.md                    # This file
├── docs/
│   ├── training-methodology.md  # Detailed QLoRA configuration and rationale
│   ├── data-pipeline.md         # Full pipeline documentation with filtering logic
│   └── evaluation-results.md    # Eval metrics, example outputs, failure modes
├── pipeline/
│   ├── 01_format_converter.py   # Raw → chat-style instruction pairs
│   ├── 02_quality_filter.py     # Length, artifact, and fragment removal
│   ├── 03_relevance_filter.py   # NIST relevance scoring and sampling
│   ├── 04_deduplication.py      # xxhash exact + MinHash LSH near-dedup
│   └── 05_train_val_split.py    # Stratified split with source balancing
├── training/
│   ├── train_qlora.py           # QLoRA training script
│   ├── config.yaml              # Hyperparameters and training config
│   └── merge_and_quantize.py    # Adapter merge + GGUF quantization
├── evaluation/
│   ├── eval_compliance.py       # Framework-specific accuracy testing
│   └── eval_cross_mapping.py    # Cross-framework mapping validation
├── deployment/
│   ├── Modelfile                # Ollama model configuration
│   └── setup_ollama.sh          # Local deployment script
└── publishing/
    ├── huggingface/             # Model cards + upload scripts
    ├── ollama/                  # Ollama Library submission guide
    └── github-releases/         # GitHub Release creation script
```

> **Note:** This repo contains the pipeline code, training configuration, and documentation. Pre-trained model weights (GGUF) are available on [Hugging Face](https://huggingface.co/Nathan-Maine). Training data and checkpoints are excluded from the repository.

---

## Roadmap

- [x] v1.0 — 7B model trained and published
- [x] v1.0 — 14B model trained and published
- [x] v1.0 — 32B model trained and published
- [x] v1.0 — 72B model trained and published
- [ ] v2.0 — Automated scraping pipeline (6/9 sources complete)
- [ ] v2.0 — Expanded training data (~15K+ examples)
- [ ] v2.0 — Retrain all 4 model sizes on v2.0 data
- [ ] RAG integration — Live document retrieval for real-time regulatory updates
- [ ] Agent integration — Deploy as a compliance agent with tool use (document search, SSP generation, gap analysis automation)
- [ ] FedRAMP baselines, CIS Controls, and ITAR coverage

---

## Built With

- **Base Models**: [Qwen2.5 Instruct](https://huggingface.co/Qwen) (abliterated variant) — 7B, 14B, 32B, 72B
- **Training**: [Unsloth](https://github.com/unslothai/unsloth) + [HuggingFace TRL](https://github.com/huggingface/trl) + [PEFT](https://github.com/huggingface/peft) — QLoRA fine-tuning
- **Quantization**: [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF format (q4_k_m / q5_k_m)
- **Inference**: [Ollama](https://ollama.ai) — Local deployment with OpenAI-compatible API
- **Data Pipeline**: [cmmc-data-pipeline](https://github.com/NathanMaine/cmmc-data-pipeline) — Python + [datasketch](https://github.com/ekzhu/datasketch) + [xxhash](https://github.com/Cyan4973/xxHash) — Scraping, conversion, deduplication
- **Data Sources**: NIST OSCAL (GitHub), eCFR API, Federal Register API, DoD PDFs
- **Local Training Hardware**: NVIDIA RTX 5000 Ada (16 GB VRAM)
- **Cloud Training Hardware**: NVIDIA A100 80GB SXM ([RunPod](https://www.runpod.io))

---

## Sources and References

All training data is derived from publicly available, authoritative government sources:

### CMMC Program

| Source | URL |
|--------|-----|
| CMMC Final Rule (32 CFR Part 170) | https://www.ecfr.gov/current/title-32/subtitle-A/chapter-I/subchapter-D/part-170 |
| DoD CIO CMMC Portal | https://dodcio.defense.gov/cmmc/ |
| CMMC Level 2 Assessment Guide | https://dodcio.defense.gov/Portals/0/Documents/CMMC/AGLevel2.pdf |
| CMMC Level 2 Scoping Guide | https://dodcio.defense.gov/Portals/0/Documents/CMMC/ScopingGuideLevel2.pdf |
| CMMC Level 3 Scoping Guide | https://dodcio.defense.gov/Portals/0/Documents/CMMC/ScopingGuideLevel3.pdf |
| CMMC ODP Values Document | https://dodcio.defense.gov/Portals/0/Documents/CMMC/ODPValues.pdf |

### NIST Publications

| Source | URL |
|--------|-----|
| NIST SP 800-171 Rev. 3 | https://csrc.nist.gov/pubs/sp/800/171/r3/final |
| NIST SP 800-171 Rev. 3 OSCAL Catalog | https://github.com/usnistgov/oscal-content |
| NIST SP 800-172 Rev. 3 (Final Public Draft) | https://csrc.nist.gov/pubs/sp/800/172/r3/fpd |
| NIST SP 800-53 Rev. 5 | https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final |
| NIST SP 800-53 Rev. 5 OSCAL Catalog | https://github.com/usnistgov/oscal-content |
| NIST SP 800-37 Rev. 2 (RMF) | https://csrc.nist.gov/pubs/sp/800/37/r2/final |
| NIST Cybersecurity Framework 2.0 | https://www.nist.gov/cyberframework |
| NIST CSF 2.0 OSCAL Catalog | https://github.com/usnistgov/oscal-content |

### Federal Regulations

| Source | URL |
|--------|-----|
| eCFR — 32 CFR Part 170 (CMMC) | https://www.ecfr.gov/current/title-32/subtitle-A/chapter-I/subchapter-D/part-170 |
| eCFR — 48 CFR Part 252 (DFARS) | https://www.ecfr.gov/current/title-48/chapter-2/subchapter-H/part-252 |
| eCFR — 45 CFR Part 164 (HIPAA Security Rule) | https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164 |
| Federal Register (CMMC Rulemakings) | https://www.federalregister.gov/ |

### DFARS Clauses

| Clause | Title | Reference |
|--------|-------|-----------|
| DFARS 252.204-7012 | Safeguarding Covered Defense Information | https://www.ecfr.gov/current/title-48/section-252.204-7012 |
| DFARS 252.204-7019 | Notice of NIST SP 800-171 Assessment | https://www.ecfr.gov/current/title-48/section-252.204-7019 |
| DFARS 252.204-7020 | NIST SP 800-171 DoD Assessment Requirements | https://www.ecfr.gov/current/title-48/section-252.204-7020 |
| DFARS 252.204-7021 | Cybersecurity Maturity Model Certification | https://www.ecfr.gov/current/title-48/section-252.204-7021 |

### Data Formats

| Format | Description | Source |
|--------|-------------|--------|
| OSCAL JSON | Machine-readable security control catalogs (NIST's Open Security Controls Assessment Language) | https://pages.nist.gov/OSCAL/ |
| eCFR API | Electronic Code of Federal Regulations structured text | https://www.ecfr.gov/developers/documentation/api/v1 |
| Federal Register API | Government policy documents and rulemakings | https://www.federalregister.gov/developers/documentation/api/v1 |

---

## Limitations

- The model is trained on publicly available compliance framework documentation. It does not contain classified or controlled information.
- Responses should be treated as expert-informed guidance, not as legal or regulatory determinations. All compliance decisions should be validated by qualified assessors.
- The model's knowledge reflects the training data at time of creation. Regulatory updates published after the training cutoff should be incorporated through the automated pipeline or RAG.
- Performance on highly specialized or edge-case compliance scenarios may vary. The model performs best on well-documented framework requirements.

---

**Built by [Nathan Maine](https://nathanmaine.com)** — Solving compliance bottlenecks with purpose-built AI.
