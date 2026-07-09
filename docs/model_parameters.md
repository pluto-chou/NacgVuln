# Model and Experiment Parameters

## Models

| Component | Model |
|---|---|
| Function-level classifier baseline | `microsoft/codebert-base` |
| Generative line-level localization model | `Salesforce/codet5-base` |

## Seed List

The experiments use 10 seed indices, mapped internally to predefined seed values:

```text
0,1,2,3,4,5,6,7,8,9
```

## NacgVuln Training Parameters

| Parameter | Value |
|---|---:|
| `INCLUDE_NEGATIVES` | `yes` |
| `NEGATIVE_RATIO` | `0.5` |
| `NO_VULN_TOKEN` | `<NO_VULN>` |
| `ROUGE_ON_VULN_ONLY` | `yes` |
| `MAX_INPUT_LEN` | `512` |
| `MAX_TARGET_LEN_CAP` | `128` |
| `BATCH_SIZE` | `2` |
| `GRAD_ACCUM_STEPS` | `4` |
| `NUM_EPOCHS` | `10` |
| `PATIENCE` | `5` |
| `LR` | `5e-5` |
| `CHUNK_TRAINING` | `yes` |
| `CHUNK_MAX_TOKENS` | `512` |
| `CHUNK_STRIDE_LINES` | `40` |
| `CHUNK_PREFIX_LINES` | `8` |
| `CHUNK_PREFIX_MAX_TOKENS` | `192` |
| `NEG_CHUNKS_PER_FUNC` | `1` |

## NacgVuln Evaluation Parameters

| Parameter | Value |
|---|---:|
| `MAX_INPUT_LEN` | `512` |
| `MAX_TARGET_LEN` | `256` for the archived main batch evaluation |
| `NUM_BEAMS` | `1` |
| `NUM_RETURN_SEQS` | `1` |
| `FALLBACK_SAMPLING` | `no` |
| `SAMPLE_RETURN_SEQS` | `1` |
| `CHUNK_LONG_FUNCS` | `yes` |
| `CHUNK_STRIDE_LINES` / `chunk_stride_lines_eval` | `20` |
| `SIMILARITY_REPLACEMENT` | `yes` |
| `DEDUP` | `yes` |
| `RERANK` | `no` |
| `EVAL_MODE` | `locvul` |
| `REMOVE_MISSING_LINE_LABELS` | `yes` |
| `EVAL_ONLY_VULN` | `yes` |
| `sort_by_lines` | `yes` |

## Hardware and Software Environment

The archived environment records report:

| Item | Value |
|---|---|
| Python | 3.9.19 |
| GPU | NVIDIA GeForce RTX 3090 Ti |
| GPU memory | 24564 MiB shown by `nvidia-smi` |
| Driver | 535.230.02 |
| CUDA shown by `nvidia-smi` | 12.2 |

Raw records are under:

```text
environment_records/
```
