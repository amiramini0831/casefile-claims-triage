"""Choose the evaluation cases and write eval/cases.yaml.

    python -m casefile.eval.select_cases

Cases are drawn with a fixed seed from the claims that qualify for each group,
and the expected result of every case is computed by casefile.rules. Nothing
is picked or typed in by hand, so the easy cases cannot be cherry-picked.

The whole selection is rebuilt from the raw files and needs no database.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import get_args

import yaml

from casefile.config import RAW_DATA_DIR
from casefile.data.prior_claims import (
    ORACLE_FILE_NAME,
    generate_prior_claims,
    read_past_claim_bins,
)
from casefile.data.records import (
    ClaimRecord,
    IncidentType,
    PolicyRecord,
    PriorClaimRecord,
    RepairCostReferenceRecord,
)
from casefile.data.repair_cost_reference import build_reference
from casefile.data.transform import (
    RAW_FILE_NAME,
    RawDataError,
    read_raw_claims,
    transform,
)
from casefile.eval.cases import (
    CASES_PATH,
    CaseGroup,
    DefectType,
    EscalationReason,
    EvalCase,
    EvalSuite,
    ExpectedResult,
    InjectionType,
    RuleThresholds,
)
from casefile.rules import (
    FREQUENT_CLAIMS_COUNT,
    FREQUENT_CLAIMS_WINDOW_MONTHS,
    Decision,
    DecisionReason,
    DocumentIssue,
    count_recent_prior_claims,
    decide,
)

SUITE_VERSION = 1
SEED = 20151009

CLEAN_CASES_PER_INCIDENT_TYPE = 2
# Claims that reach these results on their own, with untouched documents.
NATURAL_QUOTAS: tuple[tuple[DecisionReason, int], ...] = (
    ("amount_above_reference", 2),
    ("frequent_prior_claims", 3),
    ("below_deductible", 1),
)
CASES_PER_DEFECT = 2
INJECTION_QUOTAS: tuple[tuple[InjectionType, int], ...] = (
    ("extractor_error_first_attempt", 2),
    ("reviewer_never_approves", 1),
    ("repeated_schema_errors", 1),
    ("budget_too_small", 1),
    ("step_limit_too_small", 1),
)

DOCUMENT_ISSUE_BY_DEFECT: dict[DefectType, DocumentIssue] = {
    "amount_mismatch": "amount_mismatch",
    "date_mismatch": "date_mismatch",
    "missing_document": "missing_document",
}
# The system must stop itself with this reason when the fault is injected.
ESCALATION_BY_INJECTION: dict[InjectionType, EscalationReason] = {
    "reviewer_never_approves": "max_revisions",
    "repeated_schema_errors": "schema_errors",
    "budget_too_small": "budget_exceeded",
    "step_limit_too_small": "max_steps",
}
SEND_BACK_INJECTIONS: frozenset[InjectionType] = frozenset(
    {"extractor_error_first_attempt", "reviewer_never_approves"}
)
# How far before the policy start a planted out-of-coverage date lies. Only
# used to compute the expected result; the document generator picks and
# records the date it actually plants.
OUTSIDE_COVERAGE_DAYS = 30

DEFECT_NOTES: dict[DefectType, str] = {
    "amount_mismatch": "repair estimate and claim form state different amounts",
    "date_mismatch": "police report and claim form state different incident dates",
    "missing_document": "the repair estimate is missing from the file",
    "wrong_policy_number": "documents carry a policy number that does not exist",
    "incident_outside_coverage": "documents date the incident before the policy start",
}
INJECTION_NOTES: dict[InjectionType, str] = {
    "extractor_error_first_attempt": (
        "extractor returns a wrong amount once; the reviewer must send it back"
    ),
    "reviewer_never_approves": "reviewer keeps sending the work back",
    "repeated_schema_errors": "an agent returns invalid output twice in a row",
    "budget_too_small": "the case budget cannot cover the next model call",
    "step_limit_too_small": "the step limit is reached before the work is done",
}


class SelectionError(RuntimeError):
    """The data does not hold enough claims for the planned cases."""


@dataclass(frozen=True)
class Portfolio:
    """Everything the rules need to know about the loaded claims."""

    policies: Mapping[int, PolicyRecord]
    claims: tuple[ClaimRecord, ...]
    prior_claims: Mapping[int, tuple[PriorClaimRecord, ...]]
    reference: Mapping[tuple[str, str], RepairCostReferenceRecord]

    def find_policy(self, policy_number: int) -> PolicyRecord | None:
        """Return the policy, or ``None`` if the number matches no policy."""
        return self.policies.get(policy_number)

    def find_prior_claims(self, policy_number: int) -> tuple[PriorClaimRecord, ...]:
        """Return the earlier claims of a policy."""
        return self.prior_claims.get(policy_number, ())

    def find_reference(
        self, incident_type: str, incident_severity: str
    ) -> RepairCostReferenceRecord | None:
        """Return the repair cost range for this kind of claim, if any."""
        return self.reference.get((incident_type, incident_severity))

    def decide(self, claim: ClaimRecord, defect: DefectType | None = None) -> Decision:
        """Return the decision for ``claim`` once ``defect`` is planted."""
        policy: PolicyRecord | None = self.policies[claim.policy_number]
        issues: tuple[DocumentIssue, ...] = ()
        if defect == "wrong_policy_number":
            policy = None
        elif defect == "incident_outside_coverage":
            planted = policy.policy_start_date - timedelta(days=OUTSIDE_COVERAGE_DAYS)
            claim = claim.model_copy(update={"incident_date": planted})
        elif defect is not None:
            issues = (DOCUMENT_ISSUE_BY_DEFECT[defect],)
        return decide(
            claim=claim,
            policy=policy,
            prior_claims=self.prior_claims.get(claim.policy_number, ()),
            reference=self.reference.get(
                (claim.incident_type, claim.incident_severity)
            ),
            document_issues=issues,
        )


def build_portfolio(
    claims_path: Path, oracle_path: Path, *, verify_checksum: bool = True
) -> Portfolio:
    """Rebuild the domain data from the raw files, exactly as the loaders do."""
    tables = transform(read_raw_claims(claims_path, verify_checksum=verify_checksum))
    bins = read_past_claim_bins(oracle_path, verify_checksum=verify_checksum)
    prior_by_policy: dict[int, list[PriorClaimRecord]] = defaultdict(list)
    for prior in generate_prior_claims(tables.policies, tables.claims, bins):
        prior_by_policy[prior.policy_number].append(prior)
    return Portfolio(
        policies={policy.policy_number: policy for policy in tables.policies},
        claims=tables.claims,
        prior_claims={
            number: tuple(priors) for number, priors in prior_by_policy.items()
        },
        reference={
            (row.incident_type, row.incident_severity): row
            for row in build_reference(tables.claims)
        },
    )


def select_cases(
    portfolio: Portfolio,
    seed: int = SEED,
    natural_quotas: Sequence[tuple[DecisionReason, int]] = NATURAL_QUOTAS,
) -> EvalSuite:
    """Draw the cases and compute what each one is expected to end in."""
    rng = random.Random(seed)  # noqa: S311 - sampling, not cryptography
    pools: dict[DecisionReason, list[ClaimRecord]] = defaultdict(list)
    for claim in sorted(portfolio.claims, key=lambda claim: claim.claim_id):
        pools[portfolio.decide(claim).reason].append(claim)
    covered = pools["covered"]

    def draw(pool: list[ClaimRecord], count: int, purpose: str) -> list[ClaimRecord]:
        """Take ``count`` claims out of ``pool`` so no claim is used twice."""
        if len(pool) < count:
            raise SelectionError(
                f"{purpose}: need {count} claims, only {len(pool)} qualify"
            )
        chosen = rng.sample(pool, count)
        for claim in chosen:
            pool.remove(claim)
        return sorted(chosen, key=lambda claim: claim.claim_id)

    drafts: list[tuple[CaseGroup, ClaimRecord, DefectType | None, InjectionType | None]]
    drafts = []
    for incident_type in get_args(IncidentType):
        of_type = [claim for claim in covered if claim.incident_type == incident_type]
        for claim in draw(of_type, CLEAN_CASES_PER_INCIDENT_TYPE, incident_type):
            covered.remove(claim)
            drafts.append(("clean", claim, None, None))
    for reason, count in natural_quotas:
        for claim in draw(pools[reason], count, reason):
            drafts.append(("natural", claim, None, None))
    for defect in get_args(DefectType):
        # A date mismatch needs a police report to disagree with the claim form.
        eligible = [
            claim
            for claim in covered
            if defect != "date_mismatch" or claim.police_report_available
        ]
        for claim in draw(eligible, CASES_PER_DEFECT, defect):
            covered.remove(claim)
            drafts.append(("document_defect", claim, defect, None))
    for injection, count in INJECTION_QUOTAS:
        for claim in draw(covered, count, injection):
            drafts.append(("system_stress", claim, None, injection))

    cases = tuple(
        EvalCase(
            case_id=f"C{number:02d}",
            group=group,
            claim_id=claim.claim_id,
            policy_number=claim.policy_number,
            document_defect=defect,
            fault_injection=injection,
            note=_note(portfolio, claim, defect, injection),
            expected=_expected(portfolio.decide(claim, defect), injection),
        )
        for number, (group, claim, defect, injection) in enumerate(drafts, start=1)
    )
    return EvalSuite(
        version=SUITE_VERSION,
        seed=seed,
        rules=RuleThresholds(
            frequent_claims_count=FREQUENT_CLAIMS_COUNT,
            frequent_claims_window_months=FREQUENT_CLAIMS_WINDOW_MONTHS,
        ),
        cases=cases,
    )


def _expected(decision: Decision, injection: InjectionType | None) -> ExpectedResult:
    send_back = injection in SEND_BACK_INJECTIONS
    if injection in ESCALATION_BY_INJECTION:
        return ExpectedResult(
            outcome="escalated",
            reason=ESCALATION_BY_INJECTION[injection],
            reviewer_send_back=send_back,
        )
    return ExpectedResult(
        outcome=decision.outcome,
        reason=decision.reason,
        payable_amount=decision.payable_amount,
        reviewer_send_back=send_back,
    )


def _note(
    portfolio: Portfolio,
    claim: ClaimRecord,
    defect: DefectType | None,
    injection: InjectionType | None,
) -> str:
    """Explain in one line why the case ends the way it does."""
    if defect is not None:
        return DEFECT_NOTES[defect]
    if injection is not None:
        return INJECTION_NOTES[injection]
    policy = portfolio.policies[claim.policy_number]
    reason = portfolio.decide(claim).reason
    if reason == "amount_above_reference":
        high = portfolio.reference[claim.incident_type, claim.incident_severity]
        return (
            f"vehicle_claim {claim.vehicle_claim} is above the reference "
            f"high of {high.high_amount}"
        )
    if reason == "frequent_prior_claims":
        priors = portfolio.prior_claims.get(claim.policy_number, ())
        return (
            f"{count_recent_prior_claims(claim, priors)} prior claims in the "
            f"last {FREQUENT_CLAIMS_WINDOW_MONTHS} months"
        )
    if reason == "below_deductible":
        return (
            f"total_claim_amount {claim.total_claim_amount} does not exceed "
            f"the deductible of {policy.policy_deductible}"
        )
    return f"{claim.incident_type}, {claim.incident_severity}, nothing unusual"


FILE_HEADER = """\
# Pre-registered evaluation cases.
#
# Written by `python -m casefile.eval.select_cases` before the agents were
# built. Do not edit by hand: the expected results are computed by
# casefile.rules, and tests/test_eval_cases.py checks that this file is
# exactly what that script produces.
#
# outcome            what the run must end in
# reason             the single rule, or the stopping guard, behind it
# payable_amount     total_claim_amount minus the deductible, when paying
# reviewer_send_back the reviewer must return the work at least once
"""


def render(suite: EvalSuite) -> str:
    """Return the YAML text of ``suite``."""
    body = yaml.safe_dump(
        suite.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=100
    )
    return f"{FILE_HEADER}\n{body}"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--claims-path", type=Path, default=RAW_DATA_DIR / RAW_FILE_NAME
    )
    parser.add_argument(
        "--oracle-path", type=Path, default=RAW_DATA_DIR / ORACLE_FILE_NAME
    )
    parser.add_argument("--output", type=Path, default=CASES_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Select the cases and write them. Returns an exit code."""
    args = parse_args(argv)
    try:
        portfolio = build_portfolio(args.claims_path, args.oracle_path)
        suite = select_cases(portfolio)
    except (RawDataError, FileNotFoundError, SelectionError) as error:
        print(f"Cannot select cases: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(suite), encoding="utf-8", newline="\n")
    print(f"Wrote {len(suite.cases)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
