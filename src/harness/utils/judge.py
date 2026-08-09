"""
Apply a task's reviewer-written rubric to one attempt, with a judge council.

Criteria already decided by executing code are excluded; a judge only sees what
no gate can measure -- scope, idiom, behaviour no test covers::

    weighted       = sum of weight for each criterion resolved SATISFIED
    total_positive = sum of weight for each criterion whose weight > 0
    overall_score  = weighted / total_positive

Positive-only denominator: summing all weights overshoots 1 whenever a penalty
exists. A negative-weight criterion resolving SATISFIED contributes -|w|.

Resolution per criterion: unanimous, else the tiebreak member, else abstain.
An abstained criterion contributes to neither the score nor the failure count,
and `total == passed + failed + abstained` holds.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: How much of the agent's patch a judge sees. A criterion is written to be
#: decidable from a diff, so the diff is what it gets -- not the agent's
#: reasoning, which would let a confident explanation stand in for the change.
MAX_DIFF_CHARS = 60_000

_SYSTEM = """You are reviewing one attempt at a Java 8 -> 17 migration against a
written rubric. You did not write the rubric and you are not asked whether it is
a good rubric. Decide, for each numbered criterion, whether the diff below
satisfies it.

Rules:
- Judge only what the diff shows. If the diff does not contain the evidence a
  criterion asks about, the answer is No -- absence is not satisfaction.
- Do not reward intent, comments, or plausible-looking structure. A criterion
  naming a specific API, package or file is satisfied only if that specific
  change is present.
- Do not penalise a criterion for things another criterion covers.

Emit exactly {n} verdicts, in order, inside a single <judgment> block:

<judgment>
1. [[SATISFIED: Yes]] <one short reason>
2. [[SATISFIED: No]] <one short reason>
...
</judgment>"""

_USER = """RUBRIC

{rubric}

THE ATTEMPT

```diff
{diff}
```"""

_VERDICT = re.compile(r"^\s*(\d+)\s*[.)]?\s*\[\[SATISFIED:\s*(Yes|No)\]\]\s*(.*)$",
                      re.I | re.M)


@dataclass
class Verdict:
    """One criterion's outcome, and how it was reached."""

    number: str = ""
    criterion: str = ""
    weight: float = 0.0
    satisfied: Optional[bool] = None          # None == abstained
    resolved_by: str = "abstain"              # unanimous | tiebreak | abstain
    by_judge: Dict[str, Optional[bool]] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class RubricScore:
    """The channel's result for one attempt."""

    overall_score: Optional[float] = None
    verdicts: List[Verdict] = field(default_factory=list)
    council: List[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        graded = [v for v in self.verdicts if v.weight > 0]
        passed = sum(1 for v in graded if v.satisfied is True)
        failed = sum(1 for v in graded if v.satisfied is False)
        return {
            "overall_score": None if self.overall_score is None
                             else round(self.overall_score, 4),
            "rubric_weights_percentage": None if self.overall_score is None
                                         else round(self.overall_score * 100.0, 2),
            "criteria_total": len(graded),
            "criteria_passed": passed,
            "criteria_failed": failed,
            "criteria_abstained": len(graded) - passed - failed,
            "council": self.council,
            "criteria": [{
                "number": v.number, "criterion": v.criterion, "weight": v.weight,
                "satisfied": v.satisfied, "resolved_by": v.resolved_by,
                "by_judge": v.by_judge, "rationale": v.rationale,
            } for v in self.verdicts],
            "error": self.error,
        }


def judged_criteria(rubric: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    The criteria a judge decides: everything not already decided by executing code.

    A criterion carrying `enforced_by` is settled by a gate or a pytest check;
    letting a judge re-decide it would put a model over a measurement.
    """
    return [c for c in rubric
            if c.get("mode") != "automated" and not c.get("enforced_by")]


def _render(criteria: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{n}. [{c.get('importance', 'important')}, weight {c.get('score', 3)}] "
        f"{c['criterion']}"
        for n, c in enumerate(criteria, 1))


def _parse(reply: str, expected: int) -> List[Optional[bool]]:
    """
    Read one judge's verdicts, positionally.

    A judge emitting fewer verdicts than there are criteria abstains at the
    missing positions rather than being read as No.
    """
    block = re.search(r"<judgment>(.*?)</judgment>", reply, re.S | re.I)
    text = block.group(1) if block else reply
    out: List[Optional[bool]] = [None] * expected
    for number, answer, _reason in _VERDICT.findall(text):
        index = int(number) - 1
        if 0 <= index < expected:
            out[index] = answer.lower() == "yes"
    return out


def _reasons(reply: str, expected: int) -> List[str]:
    out = [""] * expected
    block = re.search(r"<judgment>(.*?)</judgment>", reply, re.S | re.I)
    for number, _answer, reason in _VERDICT.findall(block.group(1) if block else reply):
        index = int(number) - 1
        if 0 <= index < expected:
            out[index] = reason.strip()[:200]
    return out


def grade_rubric(
    rubric: Sequence[Dict[str, Any]],
    diff: str,
    llm: Callable[..., str],
    council: Optional[Sequence[str]] = None,
    tiebreak: Optional[str] = None,
) -> RubricScore:
    """
    Apply the reviewer-written criteria to one attempt.

    Args:
        rubric: the task's rubric.json, already parsed
        diff: the agent's patch
        llm: called as llm(prompt, model=...) and returning the reply text
        council: judge model ids. One member is the single-judge case.
        tiebreak: the member deciding a split. Defaults to the first; use the
            largest-context member, since a split is usually truncation.

    Returns:
        RubricScore, with `overall_score` None when nothing could be resolved --
        distinct from a resolved score of zero.
    """
    criteria = judged_criteria(rubric)
    if not criteria:
        return RubricScore(error="no judged criteria in this rubric")

    members = list(council or ["default"])
    tiebreak = tiebreak or members[0]
    n = len(criteria)
    prompt = _SYSTEM.format(n=n) + "\n\n" + _USER.format(
        rubric=_render(criteria), diff=diff[:MAX_DIFF_CHARS])

    votes: Dict[str, List[Optional[bool]]] = {}
    reasons: Dict[str, List[str]] = {}
    for member in members:
        try:
            reply = llm(prompt) if member == "default" else llm(prompt, model=member)
        except Exception as exc:                       # a dead member abstains
            logger.warning("judge %s failed: %s", member, exc)
            votes[member] = [None] * n
            reasons[member] = [""] * n
            continue
        votes[member] = _parse(reply, n)
        reasons[member] = _reasons(reply, n)

    verdicts: List[Verdict] = []
    for i, criterion in enumerate(criteria):
        cast = {m: votes[m][i] for m in members}
        voted = [v for v in cast.values() if v is not None]
        if len(voted) == len(members) and len(set(voted)) == 1:
            resolved, how = voted[0], "unanimous"
        elif cast.get(tiebreak) is not None:
            resolved, how = cast[tiebreak], "tiebreak"
        else:
            resolved, how = None, "abstain"
        verdicts.append(Verdict(
            number=criterion.get("number", f"R{i + 1}"),
            criterion=criterion["criterion"],
            weight=float(criterion.get("score", 3)),
            satisfied=resolved, resolved_by=how, by_judge=cast,
            rationale=next((reasons[m][i] for m in members if reasons[m][i]), ""),
        ))

    total_positive = sum(v.weight for v in verdicts if v.weight > 0)
    if not total_positive or all(v.satisfied is None for v in verdicts):
        return RubricScore(verdicts=verdicts, council=members,
                           error="no criterion could be resolved")

    weighted = sum(v.weight for v in verdicts if v.satisfied is True)
    return RubricScore(overall_score=weighted / total_positive,
                       verdicts=verdicts, council=members)


def grade_task_rubric(task_dir: Path, diff_path: Path, llm: Callable[..., str],
                      council: Optional[Sequence[str]] = None) -> RubricScore:
    """Load a task's rubric and grade one attempt's patch against it."""
    rubric_path = task_dir / "rubric.json"
    if not rubric_path.is_file():
        return RubricScore(error="task has no rubric.json")
    if not diff_path.is_file() or diff_path.stat().st_size == 0:
        return RubricScore(error="empty or missing diff")
    try:
        rubric = json.loads(rubric_path.read_text())
    except ValueError as exc:
        return RubricScore(error=f"unreadable rubric.json: {exc}")
    return grade_rubric(rubric, diff_path.read_text(errors="replace"), llm,
                        council=council)
