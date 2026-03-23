"""Tests for conftest.py worktree path resolution robustness."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import _resolve_project_root


class TestResolveProjectRoot:
    """Tests for _resolve_project_root() helper."""

    def test_valid_absolute_gitdir(self, tmp_path: Path) -> None:
        """When .git file has valid 'gitdir: /abs/path/.git/worktrees/45',
        returns correct project root (3 parents up from gitdir)."""
        # Create a fake worktree structure
        fake_repo = tmp_path / "main_repo"
        fake_repo.mkdir()
        worktree_dir = tmp_path / "worktree_45"
        worktree_dir.mkdir()
        gitdir_path = fake_repo / ".git" / "worktrees" / "45"
        gitdir_path.mkdir(parents=True)

        # Write .git file in worktree pointing to the gitdir
        dot_git = worktree_dir / ".git"
        dot_git.write_text(f"gitdir: {gitdir_path}\n")

        result = _resolve_project_root(worktree_dir)
        assert result == fake_repo.resolve()

    def test_oserror_falls_back_to_worktree_root(self, tmp_path: Path) -> None:
        """When .git file exists but read_text() raises OSError,
        falls back to worktree_root without raising."""
        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir()
        dot_git = worktree_dir / ".git"
        dot_git.write_text("gitdir: /some/path\n")

        # Make the file unreadable
        dot_git.chmod(0o000)
        try:
            result = _resolve_project_root(worktree_dir)
            assert result == worktree_dir
        finally:
            # Restore permissions for cleanup
            dot_git.chmod(0o644)

    def test_no_gitdir_prefix_falls_back(self, tmp_path: Path) -> None:
        """When .git file exists but content doesn't start with 'gitdir:',
        falls back to worktree_root."""
        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir()
        dot_git = worktree_dir / ".git"
        dot_git.write_text("some random content\n")

        result = _resolve_project_root(worktree_dir)
        assert result == worktree_dir

    def test_relative_gitdir_resolved_against_worktree_root(
        self, tmp_path: Path
    ) -> None:
        """When .git file contains a relative gitdir path like
        'gitdir: ../.git/worktrees/45', the relative path is resolved
        against worktree_root (not CWD)."""
        # Create structure: tmp_path/main_repo/.git/worktrees/45
        #                   tmp_path/worktrees/45/.git (file)
        main_repo = tmp_path / "main_repo"
        main_repo.mkdir()
        gitdir_path = main_repo / ".git" / "worktrees" / "45"
        gitdir_path.mkdir(parents=True)

        worktree_dir = tmp_path / "worktrees" / "45"
        worktree_dir.mkdir(parents=True)

        # Write relative gitdir path
        dot_git = worktree_dir / ".git"
        dot_git.write_text("gitdir: ../../main_repo/.git/worktrees/45\n")

        # Run with CWD set to /tmp to prove it doesn't depend on CWD
        original_cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            result = _resolve_project_root(worktree_dir)
            assert result == main_repo.resolve()
        finally:
            os.chdir(original_cwd)

    def test_dot_git_is_directory_returns_worktree_root(
        self, tmp_path: Path
    ) -> None:
        """When .git is a directory (not a file), returns worktree_root
        unchanged (normal non-worktree repo)."""
        worktree_dir = tmp_path / "normal_repo"
        worktree_dir.mkdir()
        dot_git_dir = worktree_dir / ".git"
        dot_git_dir.mkdir()

        result = _resolve_project_root(worktree_dir)
        assert result == worktree_dir

    def test_absolute_gitdir_ignores_worktree_prefix(
        self, tmp_path: Path
    ) -> None:
        """When gitdir path is absolute, _WORKTREE_ROOT prefix is
        effectively ignored (Path('/') / Path('/abs') == Path('/abs'))."""
        fake_repo = tmp_path / "repo"
        fake_repo.mkdir()
        gitdir_path = fake_repo / ".git" / "worktrees" / "99"
        gitdir_path.mkdir(parents=True)

        worktree_dir = tmp_path / "worktree_99"
        worktree_dir.mkdir()
        dot_git = worktree_dir / ".git"
        dot_git.write_text(f"gitdir: {gitdir_path}\n")

        result = _resolve_project_root(worktree_dir)
        assert result == fake_repo.resolve()
