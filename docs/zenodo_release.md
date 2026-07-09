# Zenodo Release Notes

## Is a Zenodo DOI Required?

A Zenodo DOI is strongly recommended for manuscript submission because it freezes the exact repository version used for the paper. It is not generated until after a public GitHub release is created and archived by Zenodo.

## Repository Metadata

| Field | Value |
|---|---|
| Title | NacgVuln: Negative-aware and Chunk-enhanced Generative Line-level Vulnerability Localization |
| Author | Shuai Huang |
| Affiliation | School of Computer Science and Engineering, Sichuan University of Science & Engineering, Yibin 644000, China |
| Repository | https://github.com/pluto-chou/NacgVuln |
| License | MIT |
| Version | v1.0.0 |

## Suggested Release Procedure

1. Push this repository to GitHub.
2. Confirm the repository is public.
3. Create a GitHub release with tag `v1.0.0`.
4. Connect the repository to Zenodo.
5. Let Zenodo archive the GitHub release and mint the DOI.
6. Update these files with the version DOI:

```text
README.md
CITATION.cff
.zenodo.json
docs/code_availability_statement.md
docs/zenodo_release.md
```

## Files Excluded from GitHub

Large checkpoints, raw datasets, Hugging Face caches, and logs are excluded by `.gitignore`. If needed, attach large checkpoints as Zenodo release files.
