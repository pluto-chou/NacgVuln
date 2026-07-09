# English text for manuscript

## Additional Analysis: CWE-level Performance

To further examine the behavior of NacgVuln across vulnerability types, we group vulnerable test functions by their CWE labels and report line-level localization results for frequent CWE categories. CWE provides a standardized taxonomy of software and hardware weaknesses. Samples without valid CWE labels were marked as UNKNOWN; since UNKNOWN does not correspond to a specific CWE category, these samples were excluded from the main table. To avoid unstable estimates caused by low-frequency categories, we only report CWE categories with at least 3 vulnerable functions in the main text, while the complete results are provided in the appendix.

Table 5-X presents the CWE-level localization results. The results show that NacgVuln achieves high Top-10 localization performance on frequent categories such as CWE-119, CWE-264, CWE-20, and CWE-125. However, rank-aware metrics and inspection-effort metrics still vary across CWE types. For some categories, A@10 remains high while MRR@10 or MAP@10 is relatively lower, indicating that the model can hit vulnerable lines within the Top-10 candidates but does not always rank them at the very top. This suggests that vulnerability-type-specific difficulty remains an important factor in generative line-level vulnerability localization.

It should be noted that the CWE-level results serve as complementary evidence on the behavior of the proposed method across vulnerability types, rather than a conclusive assessment of cross-CWE generalization. Since the CWE distribution in the test set is imbalanced and several CWE categories contain only a small number of samples, we further discuss this limitation in Threats to Validity.
