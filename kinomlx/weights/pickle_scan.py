"""Exact static pickle-global scanning for tensor checkpoint gates."""

from __future__ import annotations

import pickletools

from kinomlx.errors import KinoMLXError


class RestrictedCheckpointError(KinoMLXError, RuntimeError):
    """A checkpoint is malformed or outside the tensor-only allowlist."""


_STRING_OPCODES = {
    "SHORT_BINUNICODE",
    "BINUNICODE",
    "BINUNICODE8",
    "UNICODE",
    "SHORT_BINSTRING",
    "BINSTRING",
}
_UNRESOLVED_STACK_GLOBAL = "<unresolved> STACK_GLOBAL"
_UNRESOLVED_EXTENSION = "<unresolved> pickle-extension"
_UNKNOWN = object()
_MARK = pickletools.markobject
_STORAGE_TOKENS = frozenset(
    {
        "FloatStorage",
        "HalfStorage",
        "BFloat16Storage",
        "LongStorage",
        "IntStorage",
        "ShortStorage",
        "CharStorage",
        "ByteStorage",
        "BoolStorage",
    }
)
_ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch", "Size"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch._utils", "_rebuild_tensor_v2"),
    *(("torch", token) for token in _STORAGE_TOKENS),
}


def _is_allowed_global(module: str, name: str) -> bool:
    return (module, name) in _ALLOWED_GLOBALS


def _global_reference(module: object, name: object) -> str:
    if not isinstance(module, str) or not isinstance(name, str):
        return _UNRESOLVED_STACK_GLOBAL
    return f"{module} {name}"


def _split_global_reference(reference: str) -> tuple[str, str] | None:
    module, separator, name = reference.partition(" ")
    if not separator or not module or not name or " " in name:
        return None
    return module, name


def _pop_stack(stack: list[object], opcode: str) -> object:
    if not stack:
        raise RestrictedCheckpointError(
            f"cannot statically scan checkpoint pickle: {opcode} underflowed the stack"
        )
    return stack.pop()


def _memo_index(argument: object, opcode: str) -> int:
    if isinstance(argument, bool) or not isinstance(argument, (int, str)):
        raise RestrictedCheckpointError(
            f"cannot statically scan checkpoint pickle: {opcode} has an invalid memo index"
        )
    try:
        return int(argument)
    except ValueError as exc:
        raise RestrictedCheckpointError(
            f"cannot statically scan checkpoint pickle: {opcode} has an invalid memo index"
        ) from exc


def _apply_stack_effect(
    stack: list[object],
    opcode: pickletools.OpcodeInfo,
) -> None:
    before = opcode.stack_before
    if pickletools.stackslice in before:
        try:
            marker = len(stack) - 1 - stack[::-1].index(_MARK)
        except ValueError as exc:
            raise RestrictedCheckpointError(
                f"cannot statically scan checkpoint pickle: {opcode.name} has no MARK"
            ) from exc
        fixed = before.index(pickletools.markobject)
        del stack[marker:]
        for _ in range(fixed):
            _pop_stack(stack, opcode.name)
    else:
        for _ in before:
            _pop_stack(stack, opcode.name)
    stack.extend(_UNKNOWN for _ in opcode.stack_after)


def scan_pickle_globals(data: bytes) -> set[str]:
    """List globals one pickle stream references without unpickling it.

    The symbolic stack and memo are intentionally narrow: strings are retained
    only so protocol-4 ``STACK_GLOBAL`` references can be resolved exactly.
    Every other produced value is represented as unknown.
    """
    references: set[str] = set()
    stack: list[object] = []
    memo: dict[int, object] = {}
    saw_stop = False
    try:
        for opcode, argument, _position in pickletools.genops(data):
            name = opcode.name
            if name in _STRING_OPCODES:
                stack.append(str(argument))
            elif name == "GLOBAL":
                module, separator, global_name = str(argument).partition(" ")
                references.add(
                    _global_reference(module, global_name)
                    if separator
                    else _UNRESOLVED_STACK_GLOBAL
                )
                stack.append(_UNKNOWN)
            elif name == "STACK_GLOBAL":
                stack_global_name = _pop_stack(stack, name)
                stack_module = _pop_stack(stack, name)
                references.add(_global_reference(stack_module, stack_global_name))
                stack.append(_UNKNOWN)
            elif name in {"BINPUT", "LONG_BINPUT", "PUT"}:
                if not stack:
                    _pop_stack(stack, name)
                memo[_memo_index(argument, name)] = stack[-1]
            elif name == "MEMOIZE":
                if not stack:
                    _pop_stack(stack, name)
                memo[len(memo)] = stack[-1]
            elif name in {"BINGET", "LONG_BINGET", "GET"}:
                index = _memo_index(argument, name)
                if index not in memo:
                    raise RestrictedCheckpointError(
                        "cannot statically scan checkpoint pickle: "
                        f"{name} references missing memo entry {index}"
                    )
                stack.append(memo[index])
            elif name == "MARK":
                stack.append(_MARK)
            elif name == "POP":
                _pop_stack(stack, name)
            elif name == "POP_MARK":
                try:
                    marker = len(stack) - 1 - stack[::-1].index(_MARK)
                except ValueError as exc:
                    raise RestrictedCheckpointError(
                        "cannot statically scan checkpoint pickle: POP_MARK has no MARK"
                    ) from exc
                del stack[marker:]
            elif name == "DUP":
                if not stack:
                    _pop_stack(stack, name)
                stack.append(stack[-1])
            elif name == "INST":
                module, separator, global_name = str(argument).partition(" ")
                references.add(
                    _global_reference(module, global_name)
                    if separator
                    else _UNRESOLVED_STACK_GLOBAL
                )
                _apply_stack_effect(stack, opcode)
            elif name in {"EXT1", "EXT2", "EXT4"}:
                references.add(_UNRESOLVED_EXTENSION)
                _apply_stack_effect(stack, opcode)
            else:
                _apply_stack_effect(stack, opcode)
            if name == "STOP":
                saw_stop = True
                break
    except RestrictedCheckpointError:
        raise
    except Exception as exc:
        raise RestrictedCheckpointError(f"cannot statically scan checkpoint pickle: {exc}") from exc
    if not saw_stop:
        raise RestrictedCheckpointError(
            "cannot statically scan checkpoint pickle: missing STOP opcode"
        )
    return references


def suspicious_globals(references: set[str]) -> list[str]:
    """Return globals outside the restricted reader's exact allowlist."""
    return sorted(
        reference
        for reference in references
        if (
            (parsed := _split_global_reference(reference)) is None
            or not _is_allowed_global(*parsed)
        )
    )


__all__ = [
    "RestrictedCheckpointError",
    "scan_pickle_globals",
    "suspicious_globals",
]
