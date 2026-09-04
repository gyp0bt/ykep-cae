"""nsb が xkep_cae_fluid から独立していること、およびコピー元との一致を検証する."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from sync_nsb_from_xkep import FILES, diverged_files, xkep_to_nsb  # noqa: E402


class TestNSBStandaloneAPI:
    def test_copies_match_xkep_modulo_imports(self):
        """nsb/{data,assembly}.py は xkep 側と import 行以外で一致する（乖離したら sync スクリプトで同期）."""
        assert diverged_files() == [], (
            "python scripts/sync_nsb_from_xkep.py で同期してください: "
            + ", ".join(diverged_files())
        )

    def test_copies_do_not_import_xkep(self):
        for name in FILES:
            text = (ROOT / "nsb" / name).read_text(encoding="utf-8")
            assert "xkep_cae_fluid" not in text, name

    def test_import_rewrite_is_pure(self):
        src = "from xkep_cae_fluid.brinkman_flow.data import X\nfrom scipy import sparse\n"
        assert xkep_to_nsb(src) == "from nsb.data import X\nfrom scipy import sparse\n"

    def test_nsb_imports_without_xkep(self):
        """xkep_cae_fluid を import 不能にした別プロセスで nsb 全モジュールが読み込める."""
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
