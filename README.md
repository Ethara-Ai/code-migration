# code-migration

Containerized evaluation of Java 8 → 17 migration agents.

One container per repository. The agent runs inside it, isolated from the
network except an allowlist; the grader runs afterwards in a separate container
and never sees the agent's filesystem. Tasks and results are written in
[Harbor](https://github.com/harbor-framework/harbor)'s on-disk format, so they
are portable — but Harbor never drives anything.

Built on two upstream projects, both used as released:

| | Role | Source |
|---|---|---|
| **JavaMigration** | the agent under evaluation — Strands agents (`baseline`, `pe`, `rag`, `hybrid`) | this repository's history derives from [amazon-science/JavaMigration](https://github.com/amazon-science/JavaMigration) |
| **MigrationBench** | the grader — criteria r1–r5, and the JDK 17 + Maven image | pinned dependency, [amazon-science/MigrationBench](https://github.com/amazon-science/MigrationBench)@`d705e9b` |
| **Harbor** | task and job *format* only — parsed and validated, never executed | [harbor-framework/harbor](https://github.com/harbor-framework/harbor) |

---

## Install

Docker must be running. Everything else installs into a virtualenv.

```bash
uv venv --python 3.12 && uv pip install -e .
```

A model endpoint is needed only to *run an agent*. Building and validating
tasks needs none. Configure it the way your provider expects; see
`--model-provider` and `--model-base-url` below.

---

## End to end, in four commands

The short version. Each step is explained below.

```bash
# 1. find repositories whose maintainers already migrated, and build tasks
python script/build_tasks.py all --limit 20

# 2. prove one is solvable: both sides build and test green
python script/build_tasks.py validate --repos gbif/name-parser

# 3. run the reference solution through the real grading path -- must score 1.0
python eval/run_batch.py --agent-type oracle --repos gbif/name-parser

# 4. run an agent against it
python eval/run_batch.py --agent-type rag --repos gbif/name-parser
```

---

## 1. Build tasks

A task needs two things: a repository as it was before migrating, and a
migration known to be correct. `build_tasks.py` finds both.

Many of the dataset's repositories were migrated off Java 8 by their own
maintainers, after the base commit. That patch is human-authored, reviewed,
merged and shipped, and no benchmark certified it — which is what makes it
usable as a reference.

Seven stages, cheapest first, so the expensive ones only run on survivors:

| Stage | What it asks | Cost |
|---|---|---|
| `candidate` | every dataset row | free |
| `filter` | did this project leave Java 8? | 1 API call per repo |
| `locate` | which commit carries it to 17? | ~log₂(n) calls |
| `isolate` | is the migration separable from unrelated work? | 1 compare |
| `generate` | render the task bundle | local |
| `emit` | write `fix.patch` and `test.patch` | 1 diff fetch |
| `validate` | do both sides build and test green? | 2 image builds, 2 suite runs |

Run them together or one at a time:

```bash
python script/build_tasks.py all --limit 20      # every stage
python script/build_tasks.py filter              # just one
python script/build_tasks.py report              # what survived, and why the rest did not
```

State lives in `data/migrations.jsonl`, one row per repository. A run that dies
resumes where it stopped and nothing is fetched twice. Every rejection records
its reason, so the repositories that did *not* become tasks are accountable
rather than silently absent.

```
  300 repositories
    isolated     11
    validated     3
    rejected    286

  rejected, by reason:
     150  tip still on Java 8
     120  no Java version declared in the root pom at tip
       9  reached only Java 11, never 17
       3  not separable
```

## 2. Validate

A task is not usable until it is proven solvable. `validate` runs the suite
twice:

```
base commit,  JDK 8   →  must build and pass
base + golden, JDK 17 →  must build and pass
```

A task that fails either is not a hard task, it is a broken one — and grading an
agent on it produces a zero that reads as agent failure.

```bash
python script/build_tasks.py validate --repos gbif/name-parser
```

```
  gbif/name-parser VALIDATED: base 186 tests -> golden 193 tests
```

Transient faults retry rather than reject: a clone that drops mid-transfer costs
a retry, never a task.

## 3. Run the reference solution

Before measuring an agent, confirm the whole grading path works by putting the
reference solution in the agent's seat. Same containers, same grading, same
reward — the only difference is what produces the patch.

```bash
python eval/run_batch.py --agent-type oracle --repos gbif/name-parser
```

It must score **1.0**. Anything less means something in the harness is wrong,
not the migration — the patch is one the maintainers merged. The per-check
breakdown says which part.

## 4. Run an agent

```bash
python eval/run_batch.py \
    --agent-type rag \
    --repos gbif/name-parser \
    --airgap
```

`--airgap` puts the agent on an internal network whose only exit is an
allowlisting proxy, so the upstream repository — and therefore the answer —
cannot be re-fetched. `--resume` skips repositories that already finished;
resume is a directory test, not bookkeeping, because a trial directory is
claimed atomically.

Agent types: `baseline`, `pe`, `rag`, `hybrid`, and `oracle`.

### Models

`--model-provider` selects the provider and `--model-id` the model.
`--model-base-url` points at an alternative endpoint; a loopback address is
rewritten so the container can reach a service bound to the host.

---

## Where the output goes

```
bundles/<owner>__<repo>/
    task.toml · instruction.md · TRUTH.md · rubric.json · rubric.md
    environment/Dockerfile
    solution/{fix.patch, test.patch, solve.sh}
    tests/{test.sh, test_outputs.py, test_weights.json}
    trajectories/<model>/<agent>/run_N/
        result.json          verdicts and timings
        agent/               the patch, the exit code
        verifier/            reward.txt · ctrf.json · test_function_outputs.json
        artifacts/           trajectory, logs
```

The two files worth reading after a run:

```bash
cat bundles/<task>/trajectories/<model>/<agent>/run_1/result.json
cat bundles/<task>/trajectories/<model>/<agent>/run_1/verifier/test_function_outputs.json
```

`result.json` carries `minimal` and `maximal` — MigrationBench's own verdicts,
comparable to the paper — alongside each check's outcome.
`test_function_outputs.json` carries the score breakdown.

Aggregate across runs:

```bash
python script/aggregate_runs.py
```

See [NOMENCLATURE.md](NOMENCLATURE.md) for where every artifact lives and what
names it.

---

## Layout

```
eval/run_batch.py           the single driver
script/build_tasks.py       find migrations upstream, build and validate tasks
script/generate_tasks.py    render a task bundle from a dataset row
script/aggregate_runs.py    mean reward and pass@k across runs
src/harness/utils/          docker_utils · run · tasks · gates · grading · airgap
src/strands_agent/          the agent (upstream, adapted to run headless)
templates/                  task.toml · Dockerfile · instruction · verifier
docker/Dockerfile.agent     agent image, layered on MigrationBench's
data/migrations.jsonl       the task-construction ledger
```

`tasks/` and `bundles/` are generated and not tracked. `build_tasks.py` rebuilds
tasks from the ledger, which carries the frozen base and golden commit hashes.

---

## Licence

Apache 2.0. Derived from
[amazon-science/JavaMigration](https://github.com/amazon-science/JavaMigration);
see `LICENSE` and `NOTICE`.
