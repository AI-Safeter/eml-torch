# Arithmetic component replacement

Small EML equations reproduce a localized scalar contribution inside Qwen3-1.7B and Qwen3-0.6B under held-out arithmetic interventions. The useful result is component fidelity; the rest of each language model remains active. Neural controls, range-shift failures, and uncertainty are included.

- [Measured results and limitations](REPORT.md)
- [Offline interactive explorer](explorer.html) — download and open in a browser
- [Protocol and follow-up disclosures](PROTOCOL.md)
- [Reproduction commands](REPRODUCE.md)
- [Standalone airfoil equation](airfoil-replay/airfoil_equation.py)
- [Dataset/model attribution](ATTRIBUTION.md)

The core `EMLHead` implementation is on the `trainable-head` branch. This `arithmetic-components` branch adds the separate research bundle. Raw observations are compressed JSON; the reproduction command expands them into a new directory. A SHA-256 manifest covers the released files.
