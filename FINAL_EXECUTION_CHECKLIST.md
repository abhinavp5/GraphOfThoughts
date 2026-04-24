# Final Execution Checklist (BFS)

## A) Frozen Protocol

- [ ] Use `BFS_FINAL_PROTOCOL.md` unchanged for all final runs.
- [ ] Pin one commit hash for final numbers.
- [ ] Keep seed/settings fixed across model comparisons.

## B) Required Runs

- [ ] Baseline (`Qwen/Qwen2.5-0.5B-Instruct`, no adapter)
- [ ] Best SFT adapter
- [ ] Optional stage-2 adapter (DAgger or RL variant)
- [ ] For each: main + NLGraph + GLBench metrics and failures

## C) Metrics to Produce

- [ ] operation accuracy JSON
- [ ] failure analysis JSON
- [ ] state consistency JSON
- [ ] structural generalization JSON

## D) Figures to Produce

- [ ] step-accuracy comparison
- [ ] first-error distribution
- [ ] failure-by-operation-type
- [ ] structural generalization (n vs accuracy)

## E) Stage-2 (DAgger) Minimal Loop

- [ ] collect recovery examples from free-running rollouts
- [ ] run one stage-2 fine-tune pass
- [ ] re-evaluate with frozen protocol
- [ ] report delta vs baseline/SFT

## F) Reproducibility / Handoff

- [ ] pipeline + benchmark flags documented in README
- [ ] final reproduction commands documented
- [ ] transient logs/artifacts curated (only keep what you need)
- [ ] commit/tag used for report numbers recorded

## G) Writing Owners

- [ ] Methods owner
- [ ] Results owner
- [ ] Evaluation owner
- [ ] Final editor

