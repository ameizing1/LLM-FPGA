from __future__ import annotations

from .designs.cphybrid import CPHybridDesign

_DESIGNS = {
    'balanced': CPHybridDesign('balanced','Balanced',0b011,'39 LUT6_2 + 7 CARRY4'),
    'quality': CPHybridDesign('quality','Quality',0b001,'40 LUT6_2 + 8 CARRY4'),
}
_ALIASES = {}


def canonical_name(name: str) -> str:
    key=str(name).strip().lower().replace('-','_')
    key=_ALIASES.get(key,key)
    if key not in _DESIGNS:
        raise ValueError(f'unknown design {name!r}; valid={sorted(list(_DESIGNS)+list(_ALIASES))}')
    return key


def get_design(name: str):
    return _DESIGNS[canonical_name(name)]


def choices(include_aliases: bool=True):
    out=list(_DESIGNS)
    if include_aliases: out.extend(_ALIASES)
    return tuple(sorted(out))
