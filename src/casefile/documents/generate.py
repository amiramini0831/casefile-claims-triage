"""Write the documents of every evaluation case and the answer key.

    python -m casefile.documents.generate

For each case in eval/cases.yaml this writes data/cases/<case id>/ with a
claim form, a repair estimate and, when one exists, a police report. Every
planted difference is listed in eval/defects.json.

eval/defects.json and eval/cases.yaml are the answer key. The agents are given
the case folders and must never read those two files.

Like the case selection, this needs the raw files but no database.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from casefile.config import PROJECT_ROOT, RAW_DATA_DIR
from casefile.data.prior_claims import ORACLE_FILE_NAME
from casefile.data.transform import RAW_FILE_NAME, RawDataError
from casefile.documents.content import SEED, CaseDocuments, build_case_documents
from casefile.documents.render import render_case
from casefile.eval.cases import CASES_PATH, EvalSuite, load_cases
from casefile.eval.select_cases import Portfolio, build_portfolio

CASES_DIR = PROJECT_ROOT / "data" / "cases"
DEFECTS_PATH = PROJECT_ROOT / "eval" / "defects.json"


def build_all_documents(
    suite: EvalSuite, portfolio: Portfolio, seed: int = SEED
) -> list[CaseDocuments]:
    """Build the document contents of every case in ``suite``."""
    claims = {claim.claim_id: claim for claim in portfolio.claims}
    return [
        build_case_documents(
            case,
            portfolio.policies[case.policy_number],
            claims[case.claim_id],
            portfolio.policies.keys(),
            seed,
        )
        for case in suite.cases
    ]


def write_case_folders(all_documents: Sequence[CaseDocuments], cases_dir: Path) -> int:
    """Write the PDFs of every case under ``cases_dir``. Returns the file count."""
    written = 0
    for documents in all_documents:
        folder = cases_dir / documents.case_id
        folder.mkdir(parents=True, exist_ok=True)
        # A document that is no longer generated must not linger from an
        # earlier run, or a "missing document" case would not be missing it.
        for stale in folder.glob("*.pdf"):
            stale.unlink()
        for file_name, content in render_case(documents).items():
            (folder / file_name).write_bytes(content)
            written += 1
    return written


def render_defects(all_documents: Sequence[CaseDocuments], seed: int = SEED) -> str:
    """Return the JSON text of the answer key."""
    defects = [
        defect.model_dump()
        for documents in all_documents
        for defect in documents.defects
    ]
    return json.dumps({"seed": seed, "defects": defects}, indent=2) + "\n"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--claims-path", type=Path, default=RAW_DATA_DIR / RAW_FILE_NAME
    )
    parser.add_argument(
        "--oracle-path", type=Path, default=RAW_DATA_DIR / ORACLE_FILE_NAME
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the documents and the answer key. Returns an exit code."""
    args = parse_args(argv)
    try:
        suite = load_cases(CASES_PATH)
        portfolio = build_portfolio(args.claims_path, args.oracle_path)
    except (RawDataError, FileNotFoundError) as error:
        print(f"Cannot generate documents: {error}", file=sys.stderr)
        return 1

    all_documents = build_all_documents(suite, portfolio)
    written = write_case_folders(all_documents, CASES_DIR)
    DEFECTS_PATH.write_text(
        render_defects(all_documents), encoding="utf-8", newline="\n"
    )
    planted = sum(len(documents.defects) for documents in all_documents)
    print(f"Wrote {written} PDFs for {len(all_documents)} cases to {CASES_DIR}")
    print(f"Recorded {planted} planted defects in {DEFECTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
