# Nomenclature

Where every artifact lives and what names it by.

**Harbor supplies the format, never the machinery.** Nothing imports it, it is not
a dependency, and it is not installed. Tasks and results are written in its shape
so the output is readable by anything in the ecosystem, and by nothing else that
we had to build.

One layout, written directly by `eval/run_batch.py`: a **bundle** per task, holding
that task's inputs and every attempt ever made against it. Path segments name the
dimensions — **task → model → agent → attempt** — so a comparison is a directory
listing rather than a crawl through `config.json`.

`~/jobs/` is the previous layout. `script/export_bundle.py` migrates it; nothing
writes it any more.

## At-a-glance map

Write `<run>` for `bundles/<task-id>/trajectories/<model>/<agent>/run_N`.

| What you want | Where to find it | Field |
|---|---|---|
| The agent's patch | `<run>/agent/<slug>.diff` | — |
| Binary verdict | `<run>/result.json` | `verifier_result.rewards.reward` |
| MigrationBench tiers | same file | `rewards.minimal`, `rewards.maximal` |
| A gate's verdict | same file | `rewards.<gate_name>` — `true` / `false` / `null` |
| Why a trial was not graded | same file | `rewards.grading_error` |
| Scalar reward, as a Harbor reader expects | `<run>/verifier/reward.txt` | scalar in `[0,1]` |
| Test report | `<run>/verifier/ctrf.json` | CTRF |
| Which agent and model ran | `<run>/config.json` | `strands_agent.name`, `strands_agent.model_name` |
| Wall-clock of each phase | `<run>/result.json` | `environment_setup`, `agent_execution`, `verifier` |
| Collected files | `<run>/artifacts/manifest.json` | — |
| Weighted score for one attempt | `<run>/verifier/test_function_outputs.json` | `weighted_score` |
| Mean and pass@k for one model on one task | `bundles/<task-id>/trajectories/<model>/pass_summary.json` | `average_reward`, `pass_at_k` |
| η across every task | `python script/aggregate_runs.py` | `eta_minimal_pct`, `eta_maximal_pct` |

## `null` is not `false`

A gate reports three states. `true` passed, `false` failed, **`null` could not be
measured** — the base commit has no deprecated API for `deprecation_count`. A `null` never counts against a
migration and is never reported as a pass. Absence of evidence is not evidence.

`reward` is `null` when grading itself failed. That trial is excluded from η
rather than counted as an agent failure; the reason is in `rewards.grading_error`.
`reward.txt` stays numeric because Harbor requires a scalar.

## Bundle layout

```
bundles/<task-id>/
    task.toml                  metadata, resources, timeouts
    instruction.md             the prompt
    TRUTH.md                   what a correct migration had to do, generated from task.toml
    rubric.md                  written criteria, [auto] / [review]
    environment/Dockerfile     base image, clone at base_commit, history hardening
    solution/solve.sh          harvested from a scoring run, never hand-written
    tests/test.sh              writes /logs/verifier/reward.txt
    tests/test_outputs.py      MigrationBench r1-r5
    tests/rubrics.json         every criterion, machine-readable
    tests/test_weights.json    signed weight per check; negatives are guardrails
    trajectories/
        <model>/
            <agent>/
                run_1/  trial.log · config.json · lock.json · result.json
                        agent/<agent_name>.txt · agent/exit-code.txt · agent/<slug>.diff
                        verifier/reward.txt · verifier/ctrf.json
                        artifacts/manifest.json · artifacts/*
                run_2/
            pass_summary.json
```

A task's inputs are copied in when its first attempt is claimed, so a bundle whose
run was killed halfway is still replayable, and `bundles/` can be shipped as-is.

`<agent>` is the scaffold variant (`rag`, `baseline`, `pe`, `hybrid`, `oracle`)
with the `javamigration-` prefix stripped. It sits between model and attempt
because it is a condition on a fixed model, not a separate subject.

`run_N` is a sequential attempt index, unpadded: `run_1`, `run_2`, … `run_10`.

`--run-label` tags the artifacts one invocation collects. It is **not** a
directory and never has to be unique — a second run of the same task, model and
agent becomes `run_2`, whatever the label says.

### How `run_N` is assigned

`mkdir(exist_ok=False)` on `run_N`, incrementing on `FileExistsError`. The
collision *is* the increment, so parallel workers cannot claim the same directory
and no shared counter is needed. Because the index is read from the filesystem
rather than from memory, a fresh process continues the sequence — which is also
why `--resume` is a directory test rather than bookkeeping.

## Task and trial names

| Name | Rule | Example |
|---|---|---|
| Task directory | `repo_id` with `/` → `__` | `284885166__spring-boot-hashids` |
| `task.toml` name | `javamigration/<slug>` | `javamigration/284885166__spring-boot-hashids` |
| Attempt path | `<model>/<agent>/run_N` | `claude-sonnet-4-6/rag/run_1` |
| Agent | `javamigration-<agent_type>` | `javamigration-rag` |
| Eval key | `<agent>__<source>` | `javamigration-rag__tasks` |

`284885166` is a GitHub owner name, not an id — that account's username is numeric.

## `pass_summary.json`

Written per `<model>/` directory, aggregating that model's runs on that task
across agent variants. Rebuilt from the tree on every run, so it counts every
attempt ever recorded rather than only the current invocation's.

```json
{
  "task": "284885166__spring-boot-hashids",
  "model": "claude-sonnet-4-6",
  "runs": [
    {"agent": "rag", "run": 1, "reward": 1.0, "minimal": true, "maximal": true,
     "elapsed_sec": 439.1}
  ],
  "run_count": 1,
  "graded_count": 1,
  "average_reward": 1.0,
  "pass_at_k": 1.0,
  "k": 1
}
```

`average_reward` rewards consistency — every run counts, including the bad ones.
`pass_at_k` rewards capability — the best run only. A model scoring `0,0,0,1` has
mean 0.25 and pass@4 1.0. Both are reported; neither replaces the other.

Ungraded trials (`reward: null`) are excluded from both and counted in
`run_count - graded_count`.

## Where the format comes from

Measured across the 16 repositories in the reference corpus that carry task
bundles, counting repositories rather than tasks so a 148-task repository does not
outvote a 5-task one:

| Entry | Harness repos | Sample repos |
|---|---|---|
| `task.toml` · `instruction.md` · `environment/` · `tests/` | 6/6 | 9/9 |
| `tests/test.sh` | 16/16 | |
| `tests/test_outputs.py` | 14/16 (7 spell it `test_output.py`) | |
| `tests/rubrics.json` | 7/16 | |
| `solution/solve.sh` | 12/16 | |
| `environment/Dockerfile` | 14/16 | |
| `TRUTH.md` | 0/6 | 8/9 |
| `trajectories/` | 0/6 | 8/9 |

`TRUTH.md` and `trajectories/` are *delivery* artifacts: no harness repository
carries either, and almost every sample repository carries both. That split is why
`tasks/` holds neither and `bundles/` holds both -- `tasks/` is what this
repository maintains, `bundles/` is what it ships.

Within `run_N/`, `agent/` appears in 7 repositories, `verifier/` in 5 (2 more spell
it `verifiers/`), `config.json` and `result.json` in 5, `artifacts/` in 3.
Everything past that is per-benchmark, so there is no deeper standard to conform
to.

## Extensions

Deliberate additions, none of which appear in the corpus:

- `rubric.md` -- the reviewer's companion to `tests/rubrics.json`. The machine-
  readable form carries every criterion; this carries what it cannot, namely the
  difficulty banding and the hand-written notes on whether this repository is a
  fair test at all.
- `trajectories/<model>/pass_summary.json` -- mean and pass@k. No repository in
  the reference corpus writes one, though pass@k is the number the corpus reports.
- `verifier/ctrf.json` -- the gate outcomes in the format Harbor's own pytest
  verifier emits.
- `agent/<slug>.diff` -- the agent's patch. The corpus does not record one, and its
  own reader reconstructs edits from tool calls and names the result
  `reconstructed_edits` to keep the caveat attached.
- `lock.json` -- the task digest, so a task edited after a run is detectable.

## Conventions this repository does not follow

- `run_01` zero-padding. We use `run_1`.
- `trajectories/<task>/<model>/run_N` — trajectories above the task id. That
  ordering is the older one; the current convention puts the task first.
- Harbor's `jobs/<job>/<trial>/` runtime layout. It was ours until the bundle
  layout replaced it; `script/export_bundle.py` migrates what is left.
