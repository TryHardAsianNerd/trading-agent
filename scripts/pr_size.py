"""Count the changed CODE lines between a base ref and HEAD (the PR size limit).

Blank lines, comment lines and Python docstrings don't count, and docs and data
files are skipped. Usage: python scripts/pr_size.py <base>. Prints the count;
over WARN_AT it emits a GitHub Actions warning, over FAIL_AT it exits 1.
"""

import ast
import subprocess
import sys

WARN_AT = 800
FAIL_AT = 1200
SKIPPED_SUFFIXES = (".md", ".lock", ".json", ".txt")


def git(*args: str) -> str:
    """Run a git command in the current repository.

    Args:
        args: git's arguments.

    Returns:
        Its standard output.
    """
    result = subprocess.run(
        ["git", *args], capture_output=True, check=True, encoding="utf-8", errors="replace"
    )
    return result.stdout


def docstring_lines(source: str) -> set[int]:
    """Find the lines of every module, class and function docstring.

    Args:
        source: Python source code.

    Returns:
        1-based line numbers covered by docstrings.
    """
    lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def free_lines(rev: str, path: str | None) -> set[int]:
    """Lines of one file version that are docstrings, so they don't count.

    Args:
        rev: the commit holding this version.
        path: the file's path, or None when the file doesn't exist on that side.

    Returns:
        Docstring line numbers for a Python file, else an empty set.
    """
    if path is None or not path.endswith(".py"):
        return set()
    return docstring_lines(git("show", f"{rev}:{path}"))


def is_code(text: str, lineno: int, docstrings: set[int]) -> bool:
    """Decide whether one changed line counts toward the PR size.

    Args:
        text: the line without its +/- marker.
        lineno: its line number in that file version.
        docstrings: docstring lines of that file version.

    Returns:
        True for a code line; False for a blank, comment or docstring line.
    """
    stripped = text.strip()
    return bool(stripped) and not stripped.startswith("#") and lineno not in docstrings


def count_code_lines(base: str) -> int:
    """Count the changed code lines from the merge base of base and HEAD to HEAD.

    Args:
        base: the ref the PR merges into, e.g. origin/main.

    Returns:
        Added plus removed code lines.
    """
    old_rev = git("merge-base", base, "HEAD").strip()
    diff = git("diff", "-U0", "--no-color", "--no-ext-diff", old_rev, "HEAD")
    total = 0
    old_path = new_path = None
    old_free: set[int] = set()
    new_free: set[int] = set()
    in_header = skip = True
    old_no = new_no = 0
    # Split on newlines only: splitlines() would also break a line at a form feed.
    for line in diff.split("\n"):
        if line.startswith("diff --git"):
            in_header = skip = True
        elif line.startswith("@@"):
            in_header = False
            old_range, new_range = line.split()[1:3]
            old_no = int(old_range[1:].split(",")[0])
            new_no = int(new_range[1:].split(",")[0])
        elif in_header:
            # Paths are read only here: inside a hunk, a removed "-- x" line reads "--- x".
            if not line.startswith(("--- ", "+++ ")):
                continue
            # Git ends a path containing a space with a tab.
            path = None if line[4:] == "/dev/null" else line[6:].rstrip("\t")
            if line.startswith("--- "):
                old_path = path
                continue
            new_path = path
            skip = (new_path or old_path).endswith(SKIPPED_SUFFIXES)
            if not skip:
                old_free = free_lines(old_rev, old_path)
                new_free = free_lines("HEAD", new_path)
        elif skip:
            continue
        elif line.startswith("-"):
            total += is_code(line[1:], old_no, old_free)
            old_no += 1
        elif line.startswith("+"):
            total += is_code(line[1:], new_no, new_free)
            new_no += 1
    return total


def main(argv: list[str]) -> int:
    """Print the PR's code-line count and judge it against the limits.

    Args:
        argv: the command line; argv[1] is the base ref.

    Returns:
        The exit code: 1 over FAIL_AT, else 0.
    """
    count = count_code_lines(argv[1])
    print(f"{count} changed code lines (warn over {WARN_AT}, fail over {FAIL_AT})")
    if count > FAIL_AT:
        print(f"::error::{count} changed code lines is well over the ~1000 limit; split the PR")
        return 1
    if count > WARN_AT:
        print(f"::warning::{count} changed code lines is over {WARN_AT}; reviewers judge the size")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
