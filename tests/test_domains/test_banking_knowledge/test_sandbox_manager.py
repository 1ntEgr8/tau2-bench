"""Tests for SandboxManager startup dep verification.

These tests cover the eager dependency check added to ``SandboxManager.__init__``.
The motivating bug: when ``srt`` was installed via npm but the system tools it
shells out to (``rg``, ``bwrap``, ``socat``) were missing, every shell invocation
silently returned ``"Sandbox dependencies are not available on this system"`` as
a tool result. The agent treated that as a normal tool output and learned to
abandon the shell tool, while every simulation looked superficially "successful".

The fix raises ``SandboxRuntimeError`` at construction time so misconfigured
environments are loud at the start of a run.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tau2.knowledge import sandbox_manager
from tau2.knowledge.sandbox_manager import (
    _SECRET_ENV_NAME_RE,
    SandboxManager,
    SandboxRuntimeError,
    _check_sandbox_dependencies,
)

_FAKE_SECRET = "fake-secret-value-0123456789"


@pytest.fixture(autouse=True)
def _reset_dep_cache():
    """Make every test independent of the global cache."""
    sandbox_manager._DEPS_VERIFIED = False
    yield
    sandbox_manager._DEPS_VERIFIED = False


def _which_returning(present: set[str]):
    """Build a fake ``shutil.which`` that returns a path only for ``present``."""

    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in present else None

    return fake_which


class TestRequiredBinaries:
    def test_linux_requires_srt_rg_bwrap_socat(self):
        with patch.object(sandbox_manager.sys, "platform", "linux"):
            assert sandbox_manager._required_binaries() == (
                "srt",
                "rg",
                "bwrap",
                "socat",
            )

    def test_macos_requires_srt_rg(self):
        with patch.object(sandbox_manager.sys, "platform", "darwin"):
            assert sandbox_manager._required_binaries() == ("srt", "rg")

    def test_unsupported_platform_raises(self):
        with patch.object(sandbox_manager.sys, "platform", "win32"):
            with pytest.raises(SandboxRuntimeError, match="not supported"):
                sandbox_manager._required_binaries()


class TestCheckSandboxDependencies:
    def test_all_present_passes(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "rg", "bwrap", "socat"}),
            ),
        ):
            _check_sandbox_dependencies()

    def test_missing_rg_raises_with_install_hint(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "bwrap", "socat"}),
            ),
        ):
            with pytest.raises(SandboxRuntimeError) as exc_info:
                _check_sandbox_dependencies()
        msg = str(exc_info.value)
        assert "rg" in msg
        assert "ripgrep" in msg
        assert "apt install" in msg

    def test_missing_bwrap_raises_on_linux(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "rg", "socat"}),
            ),
        ):
            with pytest.raises(SandboxRuntimeError, match="bwrap"):
                _check_sandbox_dependencies()

    def test_missing_socat_raises_on_linux(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "rg", "bwrap"}),
            ),
        ):
            with pytest.raises(SandboxRuntimeError, match="socat"):
                _check_sandbox_dependencies()

    def test_missing_srt_raises(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"rg", "bwrap", "socat"}),
            ),
        ):
            with pytest.raises(SandboxRuntimeError) as exc_info:
                _check_sandbox_dependencies()
        msg = str(exc_info.value)
        assert "srt" in msg
        assert "@anthropic-ai/sandbox-runtime" in msg

    def test_macos_does_not_require_bwrap_or_socat(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "darwin"),
            patch.object(
                sandbox_manager.shutil, "which", _which_returning({"srt", "rg"})
            ),
        ):
            _check_sandbox_dependencies()

    def test_macos_install_hint_uses_brew(self):
        with (
            patch.object(sandbox_manager.sys, "platform", "darwin"),
            patch.object(sandbox_manager.shutil, "which", _which_returning({"srt"})),
        ):
            with pytest.raises(SandboxRuntimeError) as exc_info:
                _check_sandbox_dependencies()
        msg = str(exc_info.value)
        assert "brew install" in msg
        assert "rg" in msg

    def test_check_is_cached(self):
        """First call hits ``shutil.which``; subsequent calls return immediately."""
        mock_which = MagicMock(
            side_effect=_which_returning({"srt", "rg", "bwrap", "socat"})
        )
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(sandbox_manager.shutil, "which", mock_which),
        ):
            _check_sandbox_dependencies()
            calls_after_first = mock_which.call_count
            _check_sandbox_dependencies()
            _check_sandbox_dependencies()
            assert mock_which.call_count == calls_after_first

    def test_force_bypasses_cache(self):
        """``force=True`` re-runs the check even after a successful one."""
        mock_which = MagicMock(
            side_effect=_which_returning({"srt", "rg", "bwrap", "socat"})
        )
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(sandbox_manager.shutil, "which", mock_which),
        ):
            _check_sandbox_dependencies()
            calls_after_first = mock_which.call_count
            _check_sandbox_dependencies(force=True)
            assert mock_which.call_count > calls_after_first


class TestSandboxManagerInit:
    def test_init_raises_when_deps_missing(self, tmp_path):
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(sandbox_manager.shutil, "which", _which_returning({"srt"})),
        ):
            with pytest.raises(SandboxRuntimeError):
                SandboxManager(base_temp_dir=str(tmp_path))

    def test_init_succeeds_when_deps_present(self, tmp_path):
        # Skip if we can't actually create a kb dir on this filesystem
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "rg", "bwrap", "socat"}),
            ),
        ):
            sm = SandboxManager(base_temp_dir=str(tmp_path))
            try:
                assert sm.kb_dir.exists()
            finally:
                sm.cleanup()

    def test_init_does_not_run_subprocess(self, tmp_path):
        """The dep check is purely PATH-based; it must not shell out to ``srt``.

        Running an actual srt invocation from __init__ would slow simulation
        startup and could fail for unrelated reasons (e.g., bwrap mount races).
        """
        with (
            patch.object(sandbox_manager.sys, "platform", "linux"),
            patch.object(
                sandbox_manager.shutil,
                "which",
                _which_returning({"srt", "rg", "bwrap", "socat"}),
            ),
            patch.object(sandbox_manager.subprocess, "run") as mock_run,
        ):
            SandboxManager(base_temp_dir=str(tmp_path))
            mock_run.assert_not_called()


@pytest.mark.skipif(
    shutil.which("srt") is None
    or (
        sys.platform.startswith("linux")
        and (shutil.which("bwrap") is None or shutil.which("socat") is None)
    )
    or shutil.which("rg") is None,
    reason="sandbox-runtime + system deps not all installed locally",
)
class TestSandboxManagerLive:
    """End-to-end check that real construction works on a properly-configured machine."""

    def test_can_construct_real_sandbox(self, tmp_path):
        sm = SandboxManager(base_temp_dir=str(tmp_path))
        try:
            assert sm.kb_dir.exists()
        finally:
            sm.cleanup()

    def test_live_env_does_not_leak_secrets(self, tmp_path, monkeypatch):
        """``env`` inside the real sandbox must not show parent API keys."""
        monkeypatch.setenv("FAKE_TEST_API_KEY", _FAKE_SECRET)
        sm = SandboxManager(base_temp_dir=str(tmp_path))
        try:
            rc, stdout, stderr = sm.run_command("env")
            assert rc == 0, stderr
            assert "FAKE_TEST_API_KEY" not in stdout
            assert _FAKE_SECRET not in stdout
            names = {line.split("=", 1)[0] for line in stdout.splitlines()}
            assert not any(_SECRET_ENV_NAME_RE.search(n) for n in names)
        finally:
            sm.cleanup()


# --- Minimal sandbox environment and output redaction -------------------------
#
# Motivating bug: ``run_command`` called ``subprocess.run`` without ``env=``, so
# an agent running ``env`` through the shell tool received every API key loaded
# from ``.env`` in its tool result.


@pytest.fixture
def mocked_sandbox(tmp_path):
    """A SandboxManager constructed with dependency checks mocked out."""
    with (
        patch.object(sandbox_manager.sys, "platform", "linux"),
        patch.object(
            sandbox_manager.shutil,
            "which",
            _which_returning({"srt", "rg", "bwrap", "socat"}),
        ),
    ):
        sm = SandboxManager(base_temp_dir=str(tmp_path))
    try:
        yield sm
    finally:
        sm.cleanup()


def _completed(stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=0, stdout=stdout, stderr=stderr)


class TestSandboxEnvironment:
    def test_run_command_passes_minimal_env(self, mocked_sandbox, monkeypatch):
        monkeypatch.setenv("FAKE_TEST_API_KEY", _FAKE_SECRET)
        monkeypatch.setenv("FAKE_SERVICE_TOKEN", _FAKE_SECRET)
        monkeypatch.setenv("SOME_UNRELATED_VAR", "launcher-value")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setenv("LC_ALL", "en_US.UTF-8")

        with patch.object(
            sandbox_manager.subprocess, "run", return_value=_completed()
        ) as mock_run:
            mocked_sandbox.run_command("env")

        env = mock_run.call_args.kwargs["env"]
        assert env is not None
        assert "FAKE_TEST_API_KEY" not in env
        assert "FAKE_SERVICE_TOKEN" not in env
        assert "SOME_UNRELATED_VAR" not in env
        assert _FAKE_SECRET not in env.values()
        assert not any(_SECRET_ENV_NAME_RE.search(name) for name in env)
        assert env["PATH"] == "/usr/bin:/bin"
        assert env["LC_ALL"] == "en_US.UTF-8"
        # HOME points at the sandbox, not the real user's home directory.
        assert env["HOME"] == str(mocked_sandbox.sandbox_dir)
        # Working directory behavior is unchanged.
        assert mock_run.call_args.kwargs["cwd"] == str(mocked_sandbox.kb_dir)

    def test_env_keys_are_allowlisted(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", _FAKE_SECRET)
        monkeypatch.setenv("USER", "someone")
        env = sandbox_manager._build_sandbox_env(home="/sandbox")
        for name in env:
            assert (
                name == "HOME"
                or name in sandbox_manager._SANDBOX_ENV_ALLOWLIST
                or name.startswith(sandbox_manager._SANDBOX_ENV_ALLOWED_PREFIXES)
            )

    def test_secret_like_name_dropped_even_with_allowed_prefix(self, monkeypatch):
        monkeypatch.setenv("LC_FAKE_TOKEN", _FAKE_SECRET)
        env = sandbox_manager._build_sandbox_env(home="/sandbox")
        assert "LC_FAKE_TOKEN" not in env

    def test_deny_read_paths_do_not_depend_on_sandbox_home(self, mocked_sandbox):
        """srt expands ``~`` via HOME, which now points at the sandbox dir."""
        settings = json.loads(mocked_sandbox.settings_path.read_text())
        deny_read = settings["filesystem"]["denyRead"]
        assert not any(p.startswith("~") for p in deny_read)
        assert str(Path.home() / ".ssh") in deny_read


class TestSecretRedaction:
    def test_sanitize_output_redacts_secret_env_values(
        self, mocked_sandbox, monkeypatch
    ):
        monkeypatch.setenv("FAKE_TEST_API_KEY", _FAKE_SECRET)
        out = mocked_sandbox._sanitize_output(
            f"FAKE_TEST_API_KEY={_FAKE_SECRET}\nother line\n", "env"
        )
        assert _FAKE_SECRET not in out
        assert "FAKE_TEST_API_KEY=[REDACTED]" in out
        assert "other line" in out

    def test_short_and_non_secret_values_are_not_redacted(
        self, mocked_sandbox, monkeypatch
    ):
        monkeypatch.setenv("FAKE_DEBUG_KEY", "1")
        monkeypatch.setenv("FAKE_PLAIN_VAR", "plain-but-long-value")
        out = mocked_sandbox._sanitize_output("1 plain-but-long-value\n", "cat")
        assert out == "1 plain-but-long-value\n"

    def test_run_command_redacts_stdout_and_stderr(self, mocked_sandbox, monkeypatch):
        monkeypatch.setenv("FAKE_TEST_API_KEY", _FAKE_SECRET)
        with patch.object(
            sandbox_manager.subprocess,
            "run",
            return_value=_completed(
                stdout=f"leaked {_FAKE_SECRET}", stderr=f"err {_FAKE_SECRET}"
            ),
        ):
            rc, stdout, stderr = mocked_sandbox.run_command("cat something.txt")
        assert rc == 0
        assert stdout == "leaked [REDACTED]"
        assert stderr == "err [REDACTED]"
