# Archived arithmetic component experiments

Small EML equations reproduce a localized scalar contribution inside Qwen3-1.7B and Qwen3-0.6B under held-out arithmetic interventions. The useful result is component fidelity; the rest of each language model remains active. Neural controls, range-shift failures, and uncertainty are included.

- [Measured results and limitations](REPORT.md)
- [Offline interactive explorer](explorer.html) — download and open in a browser
- [Protocol and follow-up disclosures](PROTOCOL.md)
- [Reproduction commands](REPRODUCE.md)
- [Historical standalone airfoil equation](https://github.com/AI-Safeter/eml-torch/blob/51cb441c80ef6eef5b2fe4480b1c6b45ec896cf5/research/airfoil-replay/airfoil_equation.py)
- [Dataset/model attribution](ATTRIBUTION.md)

Superseded Python, C and CUDA experiment executables have been removed from the current tree. Their complete source, original manifest, and reproduction instructions remain in [commit 51cb441](https://github.com/AI-Safeter/eml-torch/tree/51cb441c80ef6eef5b2fe4480b1c6b45ec896cf5/research). The historical manifest applies to that checkout. Measured results, protocols, checkpoints, raw observations, attribution and the explorer remain here. The active robustness study uses its own frozen source in `../robustness/src`.
