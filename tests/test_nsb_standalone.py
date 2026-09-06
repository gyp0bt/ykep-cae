"""nsb が xkep_cae_fluid から独立していること（スナップショット、同期なし）を検証する."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_FILES: tuple[str, ...] = ("data.py", "assembly.py")


class TestNSBStandaloneAPI:
    def test_snapshot_files_do_not_import_xkep(self):
        for name in SNAPSHOT_FILES:
            text = (ROOT / "nsb" / name).read_text(encoding="utf-8")
            assert "from xkep_cae_fluid" not in text, name
            assert "import xkep_cae_fluid" not in text, name

    def test_snapshot_files_are_marked(self):
        """スナップショットである旨（切り離し済み・同期しない）が先頭 docstring に書いてある."""
        for name in SNAPSHOT_FILES:
            head = (ROOT / "nsb" / name).read_text(encoding="utf-8")[:600]
            assert "スナップショット" in head, name

    def test_nsb_imports_without_xkep(self):
        """xkep_cae_fluid を import 不能にした別プロセスで nsb 全モジュールが読み込める."""
        pytest.importorskip("pypardiso")
        code = (
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path, target=None):\n"
            "        if name == 'xkep_cae_fluid' or name.startswith('xkep_cae_fluid.'):\n"
            "            raise ImportError('blocked: ' + name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "import nsb, nsb.adjoint, nsb.assembly, nsb.core, nsb.data, nsb.geo, nsb.solver, nsb.utils\n"
            "from nsb import make_case, solve_steady, NSBSettings\n"
            "inp = make_case('flat', 1, 1.0, settings=NSBSettings(velocity_floor=0.1, init_field='stokes', alpha_u=1.0))\n"
            "assert not any(m.startswith('xkep') for m in sys.modules), 'xkep が読み込まれた'\n"
            "print('OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "OK"
