# Artifacts

This folder contains generated outputs from the active methodology.

Use this layout:

```text
artifacts/stage0/     frozen teacher, export manifests, Joseph packages
artifacts/stage0_1/   stabilized/instrumented PPO experiments
artifacts/stage1/     primitive discovery and preflight outputs
artifacts/stage2/     action fidelity and portfolio diagnostics
artifacts/stage3/     labels and market mechanism scores
artifacts/stage4/     outcome and PPO-mechanism diagnostics
artifacts/stage5/     one-step causal audit
artifacts/stage55/    sequential response audit
artifacts/stage6/     primitive-aware adapter experiments
```

Large generated files should stay here, not in `src/`.
