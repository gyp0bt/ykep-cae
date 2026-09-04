"""xkep_cae_fluid/brinkman_flow/{data,assembly}.py を nsb/ へコピー同期する（import 行のみ差し替え）.

nsb は xkep_cae_fluid に依存せず単体で持ち出せるよう、共有離散化を**コピー**で持つ
（[nsb/README.md](../nsb/README.md)）。xkep 側を変更したら本スクリプトで nsb 側へ反映し、
`pytest tests/test_nsb_standalone.py` で乖離が無いことを確認する。

    python scripts/sync_nsb_from_xkep.py            # xkep -> nsb へコピー
    python scripts/sync_nsb_from_xkep.py --reverse  # nsb -> xkep へコピー（nsb 側で先に直した場合）
    python scripts/sync_nsb_from_xkep.py --check    # 差分の有無だけ確認（終了コード 1 で乖離あり）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XKEP_PKG = "xkep_cae_fluid.brinkman_flow"
NSB_PKG = "nsb"
FILES: tuple[str, ...] = ("data.py", "assembly.py")


def xkep_to_nsb(text: str) -> str:
    """xkep 側ソースを nsb 側の import パスに書き換える."""
    return text.replace(f"from {XKEP_PKG}.", f"from {NSB_PKG}.")


def nsb_to_xkep(text: str) -> str:
    """nsb 側ソースを xkep 側の import パスに書き換える."""
    return text.replace(f"from {NSB_PKG}.", f"from {XKEP_PKG}.")


def diverged_files() -> list[str]:
    """import 行を正規化しても内容が一致しないファイル名の一覧."""
    out: list[str] = []
    for name in FILES:
        src = (ROOT / "xkep_cae_fluid" / "brinkman_flow" / name).read_text(encoding="utf-8")
        dst = (ROOT / "nsb" / name).read_text(encoding="utf-8")
        if xkep_to_nsb(src) != dst:
            out.append(name)
    return out


def sync(reverse: bool) -> list[str]:
    """コピー同期を実行し、書き換えたファイル名を返す."""
    written: list[str] = []
    for name in FILES:
        xkep_path = ROOT / "xkep_cae_fluid" / "brinkman_flow" / name
        nsb_path = ROOT / "nsb" / name
        if reverse:
            new = nsb_to_xkep(nsb_path.read_text(encoding="utf-8"))
            target = xkep_path
        else:
            new = xkep_to_nsb(xkep_path.read_text(encoding="utf-8"))
            target = nsb_path
        if not target.exists() or target.read_text(encoding="utf-8") != new:
            target.write_text(new, encoding="utf-8")
            written.append(str(target.relative_to(ROOT)))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reverse", action="store_true", help="nsb -> xkep_cae_fluid の向きでコピー")
    ap.add_argument("--check", action="store_true", help="差分の有無のみ確認（書き換えない）")
    args = ap.parse_args(argv)
    if args.check:
        bad = diverged_files()
        if bad:
            print("乖離あり:", ", ".join(bad))
            return 1
        print("一致: " + ", ".join(FILES))
        return 0
    written = sync(args.reverse)
    print("更新:", ", ".join(written) if written else "なし（既に一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
