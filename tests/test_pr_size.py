"""scripts/pr_size.py: only code lines count toward the PR size limit."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pr_size.py"

BASE_PY = '''"""A module."""


def greet(name):
    """Say hello.

    Args:
        name: who.
    """
    return f"hi {name}"


x = 1
'''


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo: main holds BASE_PY as a.py, and HEAD is a new branch off it."""
    # Isolate from the user's git config (signing, autocrlf, hooks).
    (tmp_path / "gitconfig").write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "t")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "t@example.com")
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    commit(path, {"a.py": BASE_PY})
    git(path, "checkout", "-q", "-b", "pr")
    return path


def git(path: Path, *args: str) -> None:
    """Run git in the test repo."""
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)


def commit(path: Path, files: dict[str, str | None]) -> None:
    """Write (or, for None, delete) files and commit them."""
    for name, content in files.items():
        if content is None:
            (path / name).unlink()
        else:
            (path / name).write_text(content, newline="\n")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "change")


def size(path: Path) -> tuple[int, subprocess.CompletedProcess[str]]:
    """Run the script against main; return the reported count and the finished process."""
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "main"],
        cwd=path,
        capture_output=True,
        text=True,
    )
    match = re.match(r"(\d+) changed code lines", run.stdout)
    assert match, run.stdout + run.stderr
    return int(match.group(1)), run


def test_comment_and_blank_edits_count_zero(repo):
    commit(repo, {"a.py": BASE_PY.replace("x = 1\n", "# a note\n\n\n    # indented note\nx = 1\n")})
    assert size(repo)[0] == 0


NEW_CLASS = '''

class Thing:
    """One line."""

    def go(self):
        """Go.

        Returns:
            Nothing.
        """
'''


def test_docstring_edits_count_zero(repo):
    edited = BASE_PY.replace('"""A module."""', '"""A module.\n\nNow longer.\n"""')
    edited = edited.replace("name: who.", "name: who to greet.")
    commit(repo, {"a.py": edited + NEW_CLASS})
    # Only the class and def lines are code; every docstring line, old and new, is free.
    assert size(repo)[0] == 2


def test_added_and_removed_code_lines_both_count(repo):
    commit(repo, {"a.py": BASE_PY.replace("x = 1\n", "x = 2\ny = 3\n")})
    assert size(repo)[0] == 3


def test_a_deleted_file_counts_its_code_but_not_its_docstrings(repo):
    commit(repo, {"a.py": None})
    assert size(repo)[0] == 3  # def, return, x = 1


def test_a_string_assigned_to_a_variable_is_code(repo):
    commit(repo, {"b.py": 'MESSAGE = """\nhello\n"""\n'})
    assert size(repo)[0] == 3


def test_a_removed_line_starting_with_dashes_is_not_a_header(repo):
    git(repo, "checkout", "-q", "main")
    commit(repo, {"c.sql": "-- note\nselect 1;\n-- other\n"})
    git(repo, "checkout", "-q", "-b", "pr2")
    commit(repo, {"c.sql": "select 1;\n"})
    assert size(repo)[0] == 2  # "-- note" and "-- other" are code in SQL


def test_docs_and_lock_files_count_zero(repo):
    commit(repo, {"README.md": "# Title\nlots of text\n", "requirements.lock": "x==1\ny==2\n"})
    assert size(repo)[0] == 0


def test_other_code_files_skip_hash_comments(repo):
    commit(repo, {"ci.yml": "# job\non: push\n\njobs: {}\n"})
    assert size(repo)[0] == 2


def test_over_800_warns_but_passes(repo):
    commit(repo, {"big.py": "".join(f"v{i} = {i}\n" for i in range(801))})
    count, run = size(repo)
    assert count == 801
    assert run.returncode == 0
    assert "::warning::" in run.stdout


def test_at_800_says_nothing(repo):
    commit(repo, {"big.py": "".join(f"v{i} = {i}\n" for i in range(800))})
    _, run = size(repo)
    assert run.returncode == 0
    assert "::" not in run.stdout


def test_over_1200_fails(repo):
    commit(repo, {"big.py": "".join(f"v{i} = {i}\n" for i in range(1201))})
    _, run = size(repo)
    assert run.returncode == 1
    assert "::error::" in run.stdout
