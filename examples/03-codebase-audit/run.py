#!/usr/bin/env python3
"""Codebase architecture audit — self-contained RLM example.

Points RLM at its own source code for a dogfooding architecture review.

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
from rlm.context import CompositeContext  # noqa: E402
from rlm.rlm import RLM  # noqa: E402

# ---------------------------------------------------------------------------
# Analysis code executed inside the RLM REPL.
# CompositeContext str() renders files separated by "--- filename ---"
# ---------------------------------------------------------------------------

AUDIT_CODE = """\
text = str(CONTEXT)
defaultdict = collections.defaultdict

# Parse composite context: files separated by "--- filename ---"
files = {}
current_file = None
current_lines = []
for line in text.split("\\n"):
    m = re.match(r'^--- (.+?) ---$', line)
    if m:
        if current_file:
            files[current_file] = "\\n".join(current_lines)
        current_file = m.group(1)
        current_lines = []
    else:
        current_lines.append(line)
if current_file:
    files[current_file] = "\\n".join(current_lines)

# Analyze each file
imports_by_file = defaultdict(list)
classes_by_file = defaultdict(list)
functions_by_file = defaultdict(list)
total_lines = 0

for fname, content in files.items():
    flines = content.strip().split("\\n")
    total_lines += len(flines)
    for line in flines:
        # Imports
        imp = re.match(r'^(?:from\\s+(\\S+)\\s+)?import\\s+(.+)', line)
        if imp:
            src = imp.group(1) or imp.group(2).split(",")[0].strip()
            imports_by_file[fname].append(src)
        # Classes
        cls = re.match(r'^class\\s+(\\w+)(?:\\(([^)]*)\\))?:', line)
        if cls:
            name = cls.group(1)
            bases = cls.group(2) or ""
            classes_by_file[fname].append((name, bases))
        # Top-level and method functions
        func = re.match(r'^(    )?(async\\s+)?def\\s+(\\w+)\\(', line)
        if func:
            indent = func.group(1)
            is_method = indent is not None
            fn_name = func.group(3)
            functions_by_file[fname].append((fn_name, is_method))

# Internal dependency graph
internal_deps = defaultdict(set)
module_names = {f.rsplit(".", 1)[0] for f in files}
for fname, imps in imports_by_file.items():
    for imp in imps:
        dep = imp.lstrip(".")
        if dep in module_names or any(dep.startswith(m) for m in module_names):
            internal_deps[fname].add(dep)

# Detect design patterns
patterns = []
for fname, cls_list in classes_by_file.items():
    for name, bases in cls_list:
        if "ABC" in bases or "Abstract" in name:
            patterns.append(f"Abstract Base Class: {name} in {fname}")
        if "Backend" in name:
            patterns.append(f"Strategy pattern: {name} in {fname}")
        if "Env" in name:
            patterns.append(f"Environment pattern: {name} in {fname}")
for fname, fn_list in functions_by_file.items():
    for fn_name, _ in fn_list:
        if "create" in fn_name.lower() or "factory" in fn_name.lower():
            patterns.append(f"Factory method: {fn_name}() in {fname}")

# Public API
public_api = []
for fname, cls_list in classes_by_file.items():
    for name, bases in cls_list:
        if not name.startswith("_"):
            label = f"{name}({bases})" if bases else name
            public_api.append(f"  class {label}  [{fname}]")
for fname, fn_list in functions_by_file.items():
    for fn_name, is_method in fn_list:
        if not fn_name.startswith("_") and not is_method:
            public_api.append(f"  def {fn_name}()  [{fname}]")

# Build report
report = "CODEBASE ARCHITECTURE AUDIT\\n"
report += "=" * 60 + "\\n\\n"
report += f"Files analyzed: {len(files)}\\n"
report += f"Total lines: {total_lines:,}\\n\\n"

report += "MODULE STRUCTURE\\n" + "-" * 60 + "\\n"
for fname in sorted(files.keys()):
    nc = len(classes_by_file.get(fname, []))
    nf = len(functions_by_file.get(fname, []))
    nl = len(files[fname].strip().split("\\n"))
    report += f"  {fname}: {nl} lines, {nc} classes, {nf} functions\\n"

report += "\\nINTERNAL DEPENDENCIES\\n" + "-" * 60 + "\\n"
for fname in sorted(files.keys()):
    deps = internal_deps.get(fname, set())
    if deps:
        report += f"  {fname} -> {', '.join(sorted(deps))}\\n"

if patterns:
    report += "\\nDESIGN PATTERNS DETECTED\\n" + "-" * 60 + "\\n"
    for p in patterns:
        report += f"  * {p}\\n"

report += "\\nPUBLIC API SURFACE\\n" + "-" * 60 + "\\n"
for item in sorted(public_api):
    report += item + "\\n"

report += "\\nPOTENTIAL IMPROVEMENTS\\n" + "-" * 60 + "\\n"
# Check for large files
for fname, content in files.items():
    nl = len(content.strip().split("\\n"))
    if nl > 200:
        report += f"  * {fname} is {nl} lines — consider splitting\\n"
# Check for missing __all__
for fname in files:
    if fname == "__init__.py" and "__all__" not in files[fname]:
        report += f"  * {fname} missing __all__ export list\\n"

FINAL(report)
"""


def _make_callback(analysis_code: str):
    """Create a callback that returns real analysis code for the REPL."""

    def callback(messages: list[dict[str, str]], model: str) -> str:
        last = messages[-1]["content"] if messages else ""
        if "Output:" in last:
            return '```python\nFINAL("See analysis output above.")\n```'
        return "I'll audit the codebase architecture.\n\n```python\n" + analysis_code + "```\n"

    return callback


def main() -> None:
    """Run codebase architecture audit."""
    backend = sys.argv[1] if len(sys.argv) > 1 else "callback"
    rlm_dir = REPO_ROOT / "rlm"
    query = (
        "Perform an architecture review of this Python package: "
        "module dependency graph, design patterns used, public API surface, "
        "and potential improvements."
    )

    print("=== Example 3: Codebase Architecture Audit ===")
    print(f"Backend: {backend}")
    print(f"Target: {rlm_dir}\n")

    if backend != "callback":
        rlm_cmd = ["uv", "run", "--directory", str(REPO_ROOT), "rlm"]
        subprocess.run(
            [
                *rlm_cmd,
                "--backend",
                backend,
                "--context-dir",
                str(rlm_dir),
                "--context-glob",
                "**/*.py",
                "--query",
                query,
                "--verbose",
            ],
            check=True,
        )
        return

    context = CompositeContext.from_directory(rlm_dir, glob="**/*.py")
    print(f"Loaded {len(context.files)} Python files ({len(context):,} bytes)\n")

    print("=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    cb = CallbackBackend(_make_callback(AUDIT_CODE))
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
