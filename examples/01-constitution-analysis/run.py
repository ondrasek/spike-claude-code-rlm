#!/usr/bin/env python3
"""US Constitution analysis — self-contained RLM example.

Demonstrates the full RLM pipeline (context -> LLM -> REPL -> answer)
using smart callbacks that generate real analysis code.

    python run.py           # No API key needed — uses smart callbacks
    python run.py anthropic # Use Anthropic API for richer analysis
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rlm.backends import CallbackBackend  # noqa: E402
from rlm.rlm import RLM  # noqa: E402

# ---------------------------------------------------------------------------
# Analysis code executed inside the RLM REPL.
# The REPL provides: CONTEXT, re, json, math, collections, itertools,
#                     print(), FINAL(), llm_query()
# ---------------------------------------------------------------------------

AMENDMENTS_CODE = """\
text = str(CONTEXT)
results = []

# --- Bill of Rights: Amendments 1-10, ratified December 15, 1791 ---
bor_start = text.find("The United States Bill of Rights")
amend_xi_pos = text.find("Amendment XI")
if bor_start >= 0 and amend_xi_pos > bor_start:
    bor_text = text[bor_start:amend_xi_pos]
else:
    bor_text = ""

roman_map = {"I":1,"II":2,"III":3,"IV":4,"V":5,
             "VI":6,"VII":7,"VIII":8,"IX":9,"X":10}

# Find standalone Roman numeral lines (amendment headers)
headers = [(m.start(), m.end(), m.group(1))
           for m in re.finditer(r'\\n([IVX]+)\\n', bor_text)
           if m.group(1) in roman_map]

for i, (start, end, numeral) in enumerate(headers):
    num = roman_map[numeral]
    body_start = end
    body_end = headers[i + 1][0] if i + 1 < len(headers) else len(bor_text)
    body = bor_text[body_start:body_end].strip().replace("\\n", " ")
    dot = body.find(". ")
    first = (body[:dot + 1] if dot > 0 else body.rstrip(".") + ".")
    results.append((num, f"  {num:>2}. (Ratified December 15, 1791) {first}"))

# --- Amendments 11-27 ---
def to_int(roman):
    vals = {"I":1,"V":5,"X":10,"L":50,"C":100}
    n = 0
    for j in range(len(roman)):
        v = vals.get(roman[j], 0)
        nxt = vals.get(roman[j + 1], 0) if j + 1 < len(roman) else 0
        n += -v if v < nxt else v
    return n

for m in re.finditer(
    r'Amendment ([IVXLC]+)\\nRatified ([^\\n]+)\\n\\n(.*?)(?=\\nAmendment [IVXLC]+\\n|\\Z)',
    text, re.DOTALL,
):
    numeral, date, body = m.group(1), m.group(2), m.group(3)
    num = to_int(numeral)
    body = body.strip().replace("\\n", " ")
    if body.startswith("Section 1. "):
        body = body[len("Section 1. "):]
    dot = body.find(". ")
    first = (body[:dot + 1] if dot > 0 else body.rstrip(".") + ".")
    results.append((num, f"  {num:>2}. (Ratified {date}) {first}"))

results.sort(key=lambda x: x[0])
header = f"THE 27 AMENDMENTS TO THE US CONSTITUTION\\n{'=' * 46}\\n\\n"
body_text = "\\n".join(line for _, line in results)
FINAL(header + body_text + f"\\n\\nTotal: {len(results)} amendments found.")
"""

VOTING_CODE = """\
text = str(CONTEXT)
voting_amendments = []

# Keywords that signal voting rights
voting_kw = ["vote", "voting", "suffrage", "election", "elect", "ballot",
             "right of citizens", "right to vote", "poll tax"]

# --- Bill of Rights (I-X) ---
bor_start = text.find("The United States Bill of Rights")
amend_xi_pos = text.find("Amendment XI")
bor_text = text[bor_start:amend_xi_pos] if bor_start >= 0 and amend_xi_pos > 0 else ""

roman_map = {"I":1,"II":2,"III":3,"IV":4,"V":5,
             "VI":6,"VII":7,"VIII":8,"IX":9,"X":10}
headers = [(m.start(), m.end(), m.group(1))
           for m in re.finditer(r'\\n([IVX]+)\\n', bor_text)
           if m.group(1) in roman_map]
for i, (start, end, numeral) in enumerate(headers):
    num = roman_map[numeral]
    body_end = headers[i + 1][0] if i + 1 < len(headers) else len(bor_text)
    body = bor_text[end:body_end].strip().lower()
    if any(kw in body for kw in voting_kw):
        voting_amendments.append((num, bor_text[end:body_end].strip()))

# --- Amendments 11-27 ---
def to_int(roman):
    vals = {"I":1,"V":5,"X":10,"L":50,"C":100}
    n = 0
    for j in range(len(roman)):
        v = vals.get(roman[j], 0)
        nxt = vals.get(roman[j + 1], 0) if j + 1 < len(roman) else 0
        n += -v if v < nxt else v
    return n

for m in re.finditer(
    r'Amendment ([IVXLC]+)\\nRatified ([^\\n]+)\\n\\n(.*?)(?=\\nAmendment [IVXLC]+\\n|\\Z)',
    text, re.DOTALL,
):
    numeral, date, body = m.group(1), m.group(2), m.group(3)
    num = to_int(numeral)
    body_clean = body.strip().lower()
    if any(kw in body_clean for kw in voting_kw):
        voting_amendments.append((num, f"Amendment {numeral} (Ratified {date})\\n{body.strip()}"))

voting_amendments.sort(key=lambda x: x[0])

output = f"VOTING RIGHTS AMENDMENTS\\n{'=' * 46}\\n\\n"
output += f"Found {len(voting_amendments)} amendments related to voting rights:\\n\\n"
for num, text_block in voting_amendments:
    lines = text_block.replace("\\n", " ")[:300]
    output += f"Amendment {num}:\\n  {lines}\\n\\n"

output += "EXPANSION OF SUFFRAGE OVER TIME:\\n"
output += "  - 15th (1870): Cannot deny vote based on race or color\\n"
output += "  - 19th (1920): Cannot deny vote based on sex\\n"
output += "  - 24th (1964): Cannot require poll tax to vote\\n"
output += "  - 26th (1971): Voting age lowered to 18\\n"
FINAL(output)
"""

CHECKS_CODE = """\
text = str(CONTEXT)

# Find Articles I, II, III
art1_start = text.find("Article 1")
if art1_start < 0:
    art1_start = text.find("ARTICLE 1")
art2_start = text.find("ARTICLE 2")
if art2_start < 0:
    art2_start = text.find("Article 2")
art3_start = text.find("ARTICLE THREE")
if art3_start < 0:
    art3_start = text.find("Article 3")
art4_start = text.find("ARTICLE FOUR")
if art4_start < 0:
    art4_start = text.find("Article 4")

art1 = text[art1_start:art2_start] if art1_start >= 0 and art2_start > art1_start else ""
art2 = text[art2_start:art3_start] if art2_start >= 0 and art3_start > art2_start else ""
art3 = text[art3_start:art4_start] if art3_start >= 0 and art4_start > art3_start else ""

output = f"CHECKS AND BALANCES IN ARTICLES I-III\\n{'=' * 46}\\n\\n"

# Article I - Legislative Powers
output += "ARTICLE I — THE LEGISLATURE (Congress)\\n" + "-" * 40 + "\\n"
checks_1 = []
if "sole Power of Impeachment" in art1:
    checks_1.append("House has sole power of impeachment (check on Executive & Judiciary)")
if "sole Power to try all Impeachments" in art1:
    checks_1.append("Senate tries impeachments; 2/3 vote to convict")
if "presented to the President" in art1:
    checks_1.append("Bills must be presented to President for signature (Executive check)")
if "two thirds" in art1 and "pass the Bill" in art1:
    checks_1.append("Congress can override presidential veto with 2/3 vote in both houses")
if "Advice and Consent" in art1 or "advice and consent" in art1:
    checks_1.append("Senate confirms appointments (check on Executive)")
if "declare War" in art1:
    checks_1.append("Only Congress can declare war (check on Executive war power)")
if "raise and support Armies" in art1:
    checks_1.append("Congress controls military funding (purse power)")
if "Habeas Corpus" in art1:
    checks_1.append("Limits on suspending habeas corpus protect individual liberty")
for c in checks_1:
    output += f"  * {c}\\n"

# Article II - Executive Powers
output += "\\nARTICLE II — THE EXECUTIVE (President)\\n" + "-" * 40 + "\\n"
checks_2 = []
if "sign it" in art2 or "approve" in art2:
    checks_2.append("President can sign or veto legislation (check on Legislature)")
if "Commander in Chief" in art2:
    checks_2.append("President is Commander in Chief (military authority)")
if "Advice and Consent of the Senate" in art2:
    checks_2.append("Treaty-making requires 2/3 Senate consent (Legislative check)")
if "nominate" in art2 and "appoint" in art2:
    checks_2.append("President nominates judges and officers; Senate must confirm")
if "Reprieves and Pardons" in art2:
    checks_2.append("Pardon power (check on Judiciary), except in impeachment cases")
if "Impeachment" in art2:
    checks_2.append("President removable by impeachment (Legislative check)")
if "State of the Union" in art2:
    checks_2.append("Must report to Congress on State of the Union")
for c in checks_2:
    output += f"  * {c}\\n"

# Article III - Judicial Powers
output += "\\nARTICLE III — THE JUDICIARY (Supreme Court)\\n" + "-" * 40 + "\\n"
checks_3 = []
if "good behavior" in art3 or "good Behavior" in art3 or "good behaviour" in art3:
    checks_3.append("Judges serve during good behavior (life tenure = judicial independence)")
if "shall not be diminished" in art3:
    checks_3.append("Judicial compensation cannot be reduced (protects independence)")
if "judicial Power shall extend" in art3:
    checks_3.append("Judicial power covers constitutional questions (implied judicial review)")
if "Trial of all Crimes" in art3 and "Jury" in art3:
    checks_3.append("Criminal trials require jury (protects citizens from government)")
if "Treason" in art3:
    checks_3.append("Treason narrowly defined with high proof standard (prevents political abuse)")
for c in checks_3:
    output += f"  * {c}\\n"

output += "\\nSUMMARY:\\n"
output += "  The Constitution distributes power across three branches, each with\\n"
output += "  mechanisms to limit the others: Congress legislates but the President\\n"
output += "  can veto; the President executes but needs Senate confirmation; the\\n"
output += "  Judiciary interprets but judges are nominated by the President and\\n"
output += "  confirmed by the Senate. Impeachment gives Congress ultimate oversight.\\n"
FINAL(output)
"""


# ---------------------------------------------------------------------------
# Callbacks: inspect the query, return REPL code that does real analysis
# ---------------------------------------------------------------------------

QUERIES = [
    (
        "List all 27 amendments with ratification years and one-sentence summaries.",
        AMENDMENTS_CODE,
    ),
    (
        "Which amendments deal with voting rights? How has the right to vote expanded over time?",
        VOTING_CODE,
    ),
    (
        "Identify the checks and balances described in Articles I, II, and III.",
        CHECKS_CODE,
    ),
]


def _make_callback(analysis_code: str):
    """Create a callback that returns real analysis code for the REPL."""

    def callback(messages: list[dict[str, str]], model: str) -> str:
        last = messages[-1]["content"] if messages else ""
        if "Output:" in last:
            # Fallback: analysis ran but didn't call FINAL — wrap output
            return '```python\nFINAL("See analysis output above.")\n```'
        return "I'll analyze the document.\n\n```python\n" + analysis_code + "```\n"

    return callback


def _run_with_real_backend(backend: str) -> None:
    """Run via CLI with a real LLM backend."""
    rlm_cmd = ["uv", "run", "--directory", str(REPO_ROOT), "rlm"]
    context = str(SCRIPT_DIR / "constitution.txt")

    for i, (query, _) in enumerate(QUERIES, 1):
        print(f"\n--- Query {i} ---")
        subprocess.run(
            [
                *rlm_cmd,
                "--backend",
                backend,
                "--context-file",
                context,
                "--query",
                query,
                "--verbose",
            ],
            check=True,
        )


def main() -> None:
    """Run all three Constitution analysis queries."""
    backend = sys.argv[1] if len(sys.argv) > 1 else "callback"

    print("=== Example 1: US Constitution Analysis ===")
    print(f"Backend: {backend}\n")

    if backend != "callback":
        _run_with_real_backend(backend)
        return

    context_path = SCRIPT_DIR / "constitution.txt"
    context = context_path.read_text(encoding="utf-8")

    for i, (query, code) in enumerate(QUERIES, 1):
        print(f"\n{'=' * 70}")
        print(f"Query {i}: {query}")
        print("=" * 70)

        cb = CallbackBackend(_make_callback(code))
        rlm = RLM(backend=cb, verbose=True)
        result = rlm.completion(context=context, query=query)

        if result.success:
            print(f"\n{'~' * 70}")
            print("ANSWER:")
            print("~" * 70)
            print(result.answer)
        else:
            print(f"ERROR: {result.error}")

        print(f"\nStats: {result.stats.iterations} iterations, {result.stats.llm_calls} LLM calls")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
