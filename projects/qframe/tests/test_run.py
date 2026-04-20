"""
Unit tests for the pipeline CLI entry point (qframe.pipeline.run).

Focuses on the dirty-worktree guard introduced 2026-04-20.
"""
import os
import sys
import pytest


# ---------------------------------------------------------------------------
# _get_dirty_files
# ---------------------------------------------------------------------------

class TestGetDirtyFiles:
    def test_returns_list(self):
        """_get_dirty_files should always return a list (never raises)."""
        from qframe.pipeline.run import _get_dirty_files
        result = _get_dirty_files()
        assert isinstance(result, list)

    def test_ignores_logs_prefix(self, tmp_path, monkeypatch):
        """Files under logs/ must be excluded from the dirty list."""
        import subprocess
        from qframe.pipeline.run import _REPO_ROOT

        # Patch subprocess to return a fake porcelain output with logs/ entry
        def fake_check_output(cmd, **kwargs):
            return "?? logs/run_20260420_090513.log\nM  src/qframe/pipeline/run.py\n"

        monkeypatch.setattr("qframe.pipeline.run.subprocess.check_output", fake_check_output)
        from qframe.pipeline.run import _get_dirty_files
        dirty = _get_dirty_files()
        # logs/... should be filtered out; src/... should be included
        assert not any("logs/" in f for f in dirty)
        assert any("src/qframe/pipeline/run.py" in f for f in dirty)

    def test_ignores_pycache_prefix(self, monkeypatch):
        """__pycache__ files must be excluded from the dirty list."""
        def fake_check_output(cmd, **kwargs):
            return "?? __pycache__/loop.cpython-311.pyc\n"

        monkeypatch.setattr("qframe.pipeline.run.subprocess.check_output", fake_check_output)
        from qframe.pipeline.run import _get_dirty_files
        dirty = _get_dirty_files()
        assert dirty == []

    def test_returns_empty_on_subprocess_error(self, monkeypatch):
        """If git is unavailable or raises, return an empty list (don't crash)."""
        def fake_check_output(cmd, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr("qframe.pipeline.run.subprocess.check_output", fake_check_output)
        from qframe.pipeline.run import _get_dirty_files
        assert _get_dirty_files() == []


# ---------------------------------------------------------------------------
# _enforce_clean_worktree
# ---------------------------------------------------------------------------

class TestEnforceCleanWorktree:
    def test_passes_on_clean_worktree(self, monkeypatch):
        """Should return without error when no dirty files are found."""
        monkeypatch.setattr("qframe.pipeline.run._get_dirty_files", lambda: [])
        monkeypatch.delenv("QFRAME_ALLOW_DIRTY", raising=False)
        from qframe.pipeline.run import _enforce_clean_worktree
        _enforce_clean_worktree()  # must not raise

    def test_exits_on_dirty_worktree(self, monkeypatch, capsys):
        """Should call sys.exit(1) when uncommitted files are present."""
        monkeypatch.setattr(
            "qframe.pipeline.run._get_dirty_files",
            lambda: ["src/qframe/pipeline/run.py", "CLAUDE.md"],
        )
        monkeypatch.delenv("QFRAME_ALLOW_DIRTY", raising=False)
        from qframe.pipeline.run import _enforce_clean_worktree

        with pytest.raises(SystemExit) as exc_info:
            _enforce_clean_worktree()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "DIRTY WORKTREE" in captured.out
        assert "QFRAME_ALLOW_DIRTY" in captured.out  # escape-hatch is shown

    def test_allow_dirty_env_var_bypasses_guard(self, monkeypatch, capsys):
        """QFRAME_ALLOW_DIRTY=1 must bypass the check even with dirty files."""
        monkeypatch.setattr(
            "qframe.pipeline.run._get_dirty_files",
            lambda: ["src/qframe/pipeline/run.py"],
        )
        monkeypatch.setenv("QFRAME_ALLOW_DIRTY", "1")
        from qframe.pipeline.run import _enforce_clean_worktree
        _enforce_clean_worktree()  # must not raise or exit
        captured = capsys.readouterr()
        assert "QFRAME_ALLOW_DIRTY is set" in captured.out

    def test_dirty_files_listed_in_output(self, monkeypatch, capsys):
        """Error output must list the dirty files to help the user act."""
        dirty = ["CLAUDE.md", "src/qframe/pipeline/loop.py"]
        monkeypatch.setattr("qframe.pipeline.run._get_dirty_files", lambda: dirty)
        monkeypatch.delenv("QFRAME_ALLOW_DIRTY", raising=False)
        from qframe.pipeline.run import _enforce_clean_worktree

        with pytest.raises(SystemExit):
            _enforce_clean_worktree()
        captured = capsys.readouterr()
        assert "CLAUDE.md" in captured.out
        assert "src/qframe/pipeline/loop.py" in captured.out
