"""Audit all calls to src/ functions in kaggle/ execution scripts using inspect.signature and AST."""

import ast
import importlib
import inspect
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

scripts_to_check = [
    "kaggle/run_exp1_resnet50.py",
    "kaggle/run_exp2c_panderm_uniform.py",
    "kaggle/run_exp3b_panderm_logit_adj.py",
]


class CallAuditor(ast.NodeVisitor):
    def __init__(self, imported_src_objects, script_name):
        self.imported = imported_src_objects  # name -> (obj, module, qualname)
        self.script_name = script_name
        self.mismatches = []
        self.valid_calls = []

    def visit_Call(self, node):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # could be module.func or obj.method
            if isinstance(node.func.value, ast.Name):
                func_name = f"{node.func.value.id}.{node.func.attr}"
                if node.func.attr in self.imported:
                    func_name = node.func.attr

        if func_name in self.imported:
            obj, mod, qual = self.imported[func_name]
            pos_args_count = len(node.args)
            kw_names = [kw.arg for kw in node.keywords if kw.arg is not None]

            # Try to inspect signature
            try:
                sig = inspect.signature(obj)
                # Create fake args for positional, and kwargs with None
                fake_args = [None] * pos_args_count
                fake_kwargs = {name: None for name in kw_names}
                try:
                    sig.bind(*fake_args, **fake_kwargs)
                    self.valid_calls.append({
                        "script": self.script_name,
                        "line": node.lineno,
                        "func": func_name,
                        "signature": str(sig),
                        "call_args": pos_args_count,
                        "call_kwargs": kw_names,
                    })
                except TypeError as err:
                    self.mismatches.append({
                        "script": self.script_name,
                        "line": node.lineno,
                        "func": func_name,
                        "signature": str(sig),
                        "call_args": pos_args_count,
                        "call_kwargs": kw_names,
                        "error": str(err),
                    })
            except Exception as ex:
                print(f"Could not inspect signature of {func_name}: {ex}")

        self.generic_visit(node)


def audit_script(rel_path: str):
    abs_path = PROJECT_ROOT / rel_path
    with open(abs_path, "r", encoding="utf-8") as f:
        code = f.read()

    tree = ast.parse(code, filename=str(abs_path))

    # Collect imported src objects
    imported_src = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and (node.module.startswith("src.") or node.module == "src"):
                try:
                    mod = importlib.import_module(node.module)
                    for alias in node.names:
                        name = alias.asname or alias.name
                        if hasattr(mod, alias.name):
                            obj = getattr(mod, alias.name)
                            imported_src[name] = (obj, node.module, alias.name)
                except Exception as e:
                    print(f"Failed to import {node.module}: {e}")

    auditor = CallAuditor(imported_src, rel_path)
    auditor.visit(tree)
    return auditor.valid_calls, auditor.mismatches


def main():
    all_mismatches = []
    all_valid = []

    print("=" * 80)
    print("AUDITING ALL SRC/ FUNCTION CALLS ACROSS KAGGLE EXECUTION SCRIPTS")
    print("=" * 80)

    for script in scripts_to_check:
        print(f"\nAnalyzing: {script}")
        valid, mismatches = audit_script(script)
        all_valid.extend(valid)
        all_mismatches.extend(mismatches)
        print(f"  Valid src calls: {len(valid)}")
        print(f"  Signature mismatches: {len(mismatches)}")
        for m in mismatches:
            print(f"    [MISMATCH] Line {m['line']}: {m['func']}")
            print(f"      Signature: {m['signature']}")
            print(f"      Provided kwargs: {m['call_kwargs']}")
            print(f"      Error: {m['error']}")

    print("\n" + "=" * 80)
    print(f"TOTAL AUDITED CALLS: {len(all_valid) + len(all_mismatches)}")
    print(f"TOTAL VALID CALLS:   {len(all_valid)}")
    print(f"TOTAL MISMATCHES:    {len(all_mismatches)}")
    print("=" * 80)

    if all_mismatches:
        print("\nSUMMARY OF ALL MISMATCHES DETECTED:")
        for idx, m in enumerate(all_mismatches, 1):
            print(f"{idx}. {m['script']}:{m['line']} -> {m['func']}()")
            print(f"   Signature: {m['signature']}")
            print(f"   Passed kwargs: {m['call_kwargs']}")
            print(f"   Exception: {m['error']}\n")
    else:
        print("\nZERO MISMATCHES DETECTED.")


if __name__ == "__main__":
    main()
