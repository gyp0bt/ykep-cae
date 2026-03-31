"""プロセス契約違反検出スクリプト.

xkep-cae と共通の契約検証フレームワーク。
FDM/FVM 流体ソルバー向けに適応。

法律（C: Contract）:
- C3: テスト未紐付けのプロセス（_test_class is None）
- C5: process() 内の未宣言依存（AST解析）
- C7: process() のメタクラスラップ漏れ
- C9: frozen dataclass numpy 配列変更検出
- C15: ProcessMeta.document_path で指定されたドキュメントが実在するか検証

使用方法:
    python contracts/validate_process_contracts.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
import textwrap
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from xkep_cae_fluid.core.base import AbstractProcess  # noqa: E402
from xkep_cae_fluid.core.registry import ProcessRegistry  # noqa: E402


def _ast_fallback_binds_to(py_file: Path, registry: dict | None = None) -> None:
    """pytest 未インストール環境用: AST で @binds_to(XxxProcess) を検出し紐付け."""
    try:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return

    reg = registry if registry is not None else ProcessRegistry.default()

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for deco in node.decorator_list:
            if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Name):
                if deco.func.id == "binds_to" and deco.args:
                    arg = deco.args[0]
                    if isinstance(arg, ast.Name):
                        process_name = arg.id
                        if process_name in reg and reg[process_name]._test_class is None:
                            mod_path = py_file.relative_to(_project_root).with_suffix("")
                            mod_name = str(mod_path).replace("/", ".")
                            test_path = f"{mod_name}::{node.name}"
                            reg[process_name]._test_class = test_path


def _import_all_modules() -> None:
    """全プロセスモジュール + テストモジュールをインポートしてレジストリを構築."""
    xkep_root = _project_root / "xkep_cae_fluid"
    scan_roots = [d for d in sorted(xkep_root.iterdir()) if d.is_dir() and d.name != "__pycache__"]
    top_tests = _project_root / "tests"
    if top_tests.exists():
        scan_roots.append(top_tests)

    _SKIP_NAMES = {"__init__", "base", "categories", "data", "slots", "tree", "runner"}

    process_modules = []
    test_files = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for py_file in sorted(scan_root.rglob("*.py")):
            if py_file.parent.name == "__pycache__":
                continue
            if py_file.parent.name == "tests" or py_file.name.startswith("test_"):
                test_files.append(py_file)
                continue
            if py_file.stem in _SKIP_NAMES:
                continue
            if not py_file.stem.isidentifier():
                continue
            mod_path = py_file.relative_to(_project_root).with_suffix("")
            mod_name = str(mod_path).replace("/", ".")
            process_modules.append(mod_name)

    for mod_name in process_modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            print(f"  警告: {mod_name} のインポートに失敗: {e}")

    for py_file in test_files:
        mod_path = py_file.relative_to(_project_root).with_suffix("")
        mod_name = str(mod_path).replace("/", ".")
        try:
            importlib.import_module(mod_name)
        except Exception:
            _ast_fallback_binds_to(py_file, registry=None)


def _is_test_fixture(cls: type) -> bool:
    module = getattr(cls, "__module__", "")
    return ".tests." in module or module.startswith("tests.")


def check_c3_test_binding(registry: dict[str, type]) -> list[str]:
    """C3: テスト未紐付けのプロセスを検出."""
    errors = []
    for name, cls in sorted(registry.items()):
        if hasattr(cls, "meta") and cls.meta.deprecated:
            continue
        if _is_test_fixture(cls):
            continue
        if cls._test_class is None:
            errors.append(f"C3: {name} にテストが紐付けられていない (@binds_to 未使用)")
    return errors


def check_c5_undeclared_deps(registry: dict[str, type]) -> list[str]:
    """C5: process() 内の未宣言依存をAST解析で検出."""
    errors = []
    registry_names = set(registry.keys())

    for name, cls in sorted(registry.items()):
        meta = getattr(cls, "meta", None)
        if meta is not None and getattr(meta, "deprecated", False):
            continue
        method = getattr(cls, "process", None)
        if method is None:
            continue
        if hasattr(method, "__wrapped__"):
            method = method.__wrapped__

        try:
            source = inspect.getsource(method)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            continue

        used_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in registry_names
        }
        declared = {dep.__name__ for dep in cls.uses}
        undeclared = used_names - declared - {name}

        for u in sorted(undeclared):
            errors.append(f"C5: {name}.process() が uses 未宣言の {u} を参照")

    return errors


def check_c7_metaclass_wrap(registry: dict[str, type]) -> list[str]:
    """C7: process() のメタクラスラップ漏れを検出."""
    errors = []
    for name, cls in sorted(registry.items()):
        method = getattr(cls, "process", None)
        if method is None:
            continue
        if not hasattr(method, "__wrapped__"):
            errors.append(f"C7: {name}.process() がメタクラスでラップされていない")
    return errors


def check_c9_frozen_immutability(registry: dict[str, type]) -> list[str]:
    """C9: execute() に入力データ不変性チェックが実装されているか検証."""
    errors = []
    try:
        source = inspect.getsource(AbstractProcess.execute)
        if "checksum" not in source and "hash" not in source:
            errors.append(
                "C9: AbstractProcess.execute() に入力データ不変性チェックが未実装"
            )
    except (OSError, TypeError):
        errors.append("C9: AbstractProcess.execute() のソースを取得できない")
    return errors


def check_c15_strategy_docs(registry: dict[str, type]) -> list[str]:
    """C15: ProcessMeta.document_path で指定されたドキュメントが実在するか検証."""
    errors = []
    for name, cls in sorted(registry.items()):
        meta = getattr(cls, "meta", None)
        if meta is None:
            continue
        doc_path = getattr(meta, "document_path", None)
        if not doc_path:
            continue
        try:
            src_file = Path(inspect.getfile(cls))
            doc_full = (src_file.parent / doc_path).resolve()
            if not doc_full.exists():
                errors.append(
                    f"C15: {name} の document_path '{doc_path}' が存在しない"
                    f" (期待: {doc_full.relative_to(_project_root)})"
                )
        except (TypeError, OSError):
            errors.append(f"C15: {name} のソースファイルを取得できない")
    return errors


def main() -> int:
    """全チェックを実行し、結果を表示."""
    print("=" * 60)
    print("プロセス契約違反検出スクリプト（C3-C15）")
    print("=" * 60)

    print("\nモジュールインポート中...")
    _import_all_modules()

    registry = ProcessRegistry.default()
    print(f"レジストリ登録プロセス数: {len(registry)}")
    for name in sorted(registry.keys()):
        cls = registry[name]
        test = cls._test_class or "(未紐付け)"
        print(f"  {name}: test={test}")

    all_errors: list[str] = []
    checks = [
        ("C3: テスト紐付け", check_c3_test_binding),
        ("C5: 未宣言依存（AST）", check_c5_undeclared_deps),
        ("C7: メタクラスラップ", check_c7_metaclass_wrap),
        ("C9: frozen不変性", check_c9_frozen_immutability),
        ("C15: ドキュメント存在", check_c15_strategy_docs),
    ]

    for label, check_fn in checks:
        print(f"\n--- {label} ---")
        errors = check_fn(registry)
        all_errors.extend(errors)
        if errors:
            for e in errors:
                print(f"  NG: {e}")
        else:
            print("  OK")

    print("\n" + "=" * 60)
    if all_errors:
        print(f"契約違反: {len(all_errors)} 件")
        return 1
    else:
        print("契約違反なし")
        return 0


if __name__ == "__main__":
    sys.exit(main())
