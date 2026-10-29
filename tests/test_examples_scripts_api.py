"""Fail when shipped examples or scripts drift from the current API."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any

from metalsurfer.campaign_schema import load_campaign_yaml
from metalsurfer.campaigns import (
    run_adsorption,
    run_adsorption_bo,
    run_campaign,
    run_saturation,
    run_saturation_bo,
)
from metalsurfer.config import AdsorptionConfig, BOConfig, BOTransferConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.ml.bayesian import (
    build_transfer_surrogate,
    score_and_select,
    select_initial_bo_indices,
    splice_exploration_picks,
    train_surrogate,
)
from metalsurfer.ml.features import extract_features_from_dataset
from metalsurfer.models import windowed_bo_step_memories
from metalsurfer.placement.generators import (
    enumerate_placement_specs,
    generate_placement_from_spec_with_reason,
)
from metalsurfer.placement.site_enumeration import get_unified_sites
from metalsurfer.surface_prep.prep import (
    finalize_substrate,
    prepare_substrate,
    resize_substrate_for_molecule,
)

_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIRS = (
    _ROOT / "examples",
    _ROOT / "scripts",
    _ROOT / ".github" / "scripts",
)
_YAML_DIRS = (_ROOT / "examples", _ROOT / "scripts")
_CONFIG_CALLS = frozenset({"AdsorptionConfig", "BOConfig", "BOTransferConfig"})
_TRACKED: dict[str, Any] = {
    "AdsorptionConfig": AdsorptionConfig,
    "BOConfig": BOConfig,
    "BOTransferConfig": BOTransferConfig,
    "prepare_substrate": prepare_substrate,
    "finalize_substrate": finalize_substrate,
    "resize_substrate_for_molecule": resize_substrate_for_molecule,
    "run_adsorption": run_adsorption,
    "run_adsorption_bo": run_adsorption_bo,
    "run_saturation": run_saturation,
    "run_saturation_bo": run_saturation_bo,
    "run_campaign": run_campaign,
    "enumerate_placement_specs": enumerate_placement_specs,
    "generate_placement_from_spec_with_reason": (
        generate_placement_from_spec_with_reason
    ),
    "get_unified_sites": get_unified_sites,
    "create_conformers_from_smiles": create_conformers_from_smiles,
    "train_surrogate": train_surrogate,
    "build_transfer_surrogate": build_transfer_surrogate,
    "score_and_select": score_and_select,
    "select_initial_bo_indices": select_initial_bo_indices,
    "splice_exploration_picks": splice_exploration_picks,
    "extract_features_from_dataset": extract_features_from_dataset,
    "windowed_bo_step_memories": windowed_bo_step_memories,
}
_KEYWORD_KINDS = frozenset(
    {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
)


class _Scope:
    def __init__(self, tree: ast.Module) -> None:
        self.symbols: dict[str, str] = {}
        self.modules: dict[str, str] = {}
        self.local_defs = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in ast.walk(tree):
            self._bind(node)

    def _bind(self, node: ast.AST) -> None:
        if isinstance(node, ast.ImportFrom):
            self._bind_from(node)
            return
        if isinstance(node, ast.Import):
            self._bind_import(node)

    def _bind_from(self, node: ast.ImportFrom) -> None:
        if node.level != 0 or not _is_metalsurfer_module(node.module):
            return
        for alias in node.names:
            if alias.name in _TRACKED:
                self.symbols[alias.asname or alias.name] = alias.name

    def _bind_import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _is_metalsurfer_module(alias.name):
                continue
            local = alias.asname or alias.name.split(".")[-1]
            self.modules[local] = alias.name


class _NotLiteral(Exception):
    pass


def _rel(path: Path) -> str:
    return path.relative_to(_ROOT).as_posix()


def _python_files() -> list[Path]:
    return sorted(
        path
        for directory in _PYTHON_DIRS
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _yaml_files() -> list[Path]:
    return sorted(
        path
        for directory in _YAML_DIRS
        for pattern in ("*.yaml", "*.yml")
        for path in directory.rglob(pattern)
    )


def _is_metalsurfer_module(name: str | None) -> bool:
    return name is not None and (
        name == "metalsurfer" or name.startswith("metalsurfer.")
    )


def _root_name(expr: ast.expr) -> str | None:
    current = expr
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _call_target(func: ast.expr, scope: _Scope) -> str | None:
    if isinstance(func, ast.Name):
        if func.id in scope.local_defs:
            return None
        return scope.symbols.get(func.id)
    if not isinstance(func, ast.Attribute) or func.attr not in _TRACKED:
        return None
    root = _root_name(func.value)
    module_name = None if root is None else scope.modules.get(root)
    if _is_metalsurfer_module(module_name):
        return func.attr
    return None


def _allowed_keywords(fn: Any) -> set[str] | None:
    signature = inspect.signature(fn)
    parameters = tuple(signature.parameters.values())
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters):
        return None
    return {
        name
        for name, param in signature.parameters.items()
        if param.kind in _KEYWORD_KINDS
    }


def _sequence_literal(node: ast.List | ast.Tuple | ast.Set, scope: _Scope) -> Any:
    if any(isinstance(elt, ast.Starred) for elt in node.elts):
        raise _NotLiteral
    values = [_eval_literal(elt, scope) for elt in node.elts]
    if isinstance(node, ast.List):
        return values
    if isinstance(node, ast.Tuple):
        return tuple(values)
    return set(values)


def _dict_literal(node: ast.Dict, scope: _Scope) -> dict[Any, Any]:
    items: list[tuple[Any, Any]] = []
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None or isinstance(value, ast.Starred):
            raise _NotLiteral
        items.append((_eval_literal(key, scope), _eval_literal(value, scope)))
    return dict(items)


def _unary_literal(node: ast.UnaryOp, scope: _Scope) -> Any:
    if not isinstance(node.op, (ast.UAdd, ast.USub)):
        raise _NotLiteral
    value = _eval_literal(node.operand, scope)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _NotLiteral
    if isinstance(node.op, ast.USub):
        return -value
    return value


def _eval_literal(node: ast.expr, scope: _Scope) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        return _unary_literal(node, scope)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return _sequence_literal(node, scope)
    if isinstance(node, ast.Dict):
        return _dict_literal(node, scope)
    if isinstance(node, ast.Call):
        return _construct_literal_call(node, scope)
    raise _NotLiteral


def _construct_literal_call(node: ast.Call, scope: _Scope) -> Any:
    target = _call_target(node.func, scope)
    if target not in _CONFIG_CALLS or node.args:
        raise _NotLiteral
    kwargs: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            raise _NotLiteral
        kwargs[kw.arg] = _eval_literal(kw.value, scope)
    return _TRACKED[target](**kwargs)


def _parsed_modules() -> list[tuple[Path, ast.Module]]:
    return [
        (
            path,
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
        )
        for path in _python_files()
    ]


def _assert_clean(errors: list[str]) -> None:
    assert not errors, "Stale scripts or examples:\n" + "\n".join(errors)


def test_shipped_python_compiles() -> None:
    errors: list[str] = []
    for path in _python_files():
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{_rel(path)}:{exc.lineno or 0}: {exc.msg}")
    _assert_clean(errors)


def _import_errors(path: Path, tree: ast.Module, loaded: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [
                (alias.name, None)
                for alias in node.names
                if _is_metalsurfer_module(alias.name)
            ]
        elif isinstance(node, ast.ImportFrom) and _is_metalsurfer_module(node.module):
            if node.level != 0 or node.module is None:
                continue
            modules = [(node.module, node.names)]
        else:
            continue
        for module_name, aliases in modules:
            module = _load_module(path, node.lineno, module_name, loaded, errors)
            if module is None or aliases is None:
                continue
            errors.extend(
                _missing_names(path, node.lineno, module_name, module, aliases)
            )
    return errors


def _load_module(
    path: Path,
    lineno: int,
    module_name: str,
    loaded: dict[str, Any],
    errors: list[str],
) -> Any:
    if module_name in loaded:
        return loaded[module_name]
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        loaded[module_name] = None
        errors.append(
            f"{_rel(path)}:{lineno}: import {module_name} "
            f"failed: {type(exc).__name__}: {exc}"
        )
        return None
    loaded[module_name] = module
    return module


def _missing_names(
    path: Path,
    lineno: int,
    module_name: str,
    module: Any,
    aliases: list[ast.alias],
) -> list[str]:
    return [
        f"{_rel(path)}:{lineno}: {module_name}.{alias.name} is missing"
        for alias in aliases
        if alias.name != "*" and not hasattr(module, alias.name)
    ]


def test_shipped_metalsurfer_imports_exist() -> None:
    errors: list[str] = []
    loaded: dict[str, Any] = {}
    for path, tree in _parsed_modules():
        errors.extend(_import_errors(path, tree, loaded))
    _assert_clean(errors)


def test_shipped_call_keywords_match_signatures() -> None:
    errors: list[str] = []
    allowed_by_name = {name: _allowed_keywords(fn) for name, fn in _TRACKED.items()}
    for path, tree in _parsed_modules():
        scope = _Scope(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _call_target(node.func, scope)
            allowed = None if target is None else allowed_by_name[target]
            if allowed is None:
                continue
            errors.extend(
                f"{_rel(path)}:{node.lineno}: {target}() has unknown keyword {kw.arg!r}"
                for kw in node.keywords
                if kw.arg is not None and kw.arg not in allowed
            )
    _assert_clean(errors)


def test_shipped_literal_configs_construct() -> None:
    errors: list[str] = []
    for path, tree in _parsed_modules():
        scope = _Scope(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _call_target(node.func, scope)
            if target not in _CONFIG_CALLS:
                continue
            try:
                _construct_literal_call(node, scope)
            except _NotLiteral:
                continue
            except Exception as exc:
                errors.append(
                    f"{_rel(path)}:{node.lineno}: {target}() "
                    f"{type(exc).__name__}: {exc}"
                )
    _assert_clean(errors)


def test_shipped_campaign_yaml_loads() -> None:
    errors: list[str] = []
    for path in _yaml_files():
        try:
            load_campaign_yaml(path)
        except Exception as exc:
            errors.append(f"{_rel(path)}: {type(exc).__name__}: {exc}")
    _assert_clean(errors)
