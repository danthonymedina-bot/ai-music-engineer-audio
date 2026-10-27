"""Undo/redo: immutable Project snapshots + an edit-op log. Plus a few pure edit operations.

Every op returns a *new* Project (models are frozen). `commit` records the op with before/after JSON for each
target id, drops any redo branch, and advances the cursor. `undo`/`redo` only move the cursor.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from .model import EditHistory, EditOp, Lane, LaneItem, Processor, Project, Range

MAX_HISTORY = 200


def find(project: Project, target_id: str) -> Optional[Any]:
    """Look up any addressable object by id. 'project' and 'snap' are reserved ids."""
    if target_id == "project":
        return project.model_copy(update={"lanes": (), "clips": ()})
    if target_id == "snap":
        return project.snap
    for c in project.clips:
        if c.id == target_id:
            return c
    for lane in project.lanes:
        if lane.id == target_id:
            return lane.model_copy(update={"items": (), "processorChain": ()})  # lane-level fields only
        for obj in (*lane.items, *lane.processorChain):
            if obj.id == target_id:
                return obj
    return None


def _dump(obj) -> Optional[dict]:
    return None if obj is None else obj.model_dump(mode="json")


def new_history(project: Project) -> EditHistory:
    return EditHistory(snapshots=(project,), ops=(), cursor=0)


def current(h: EditHistory) -> Project:
    return h.snapshots[h.cursor]


def commit(h: EditHistory, new_project: Project, op_type: str, target_ids: list[str] | tuple[str, ...],
           op_id: Optional[str] = None) -> EditHistory:
    old = current(h)
    op = EditOp(opId=op_id or uuid.uuid4().hex[:12], type=op_type, targetIds=tuple(target_ids),
                before={t: _dump(find(old, t)) for t in target_ids},
                after={t: _dump(find(new_project, t)) for t in target_ids})
    snaps = h.snapshots[: h.cursor + 1] + (new_project,)
    ops = h.ops[: h.cursor] + (op,)
    if len(ops) > MAX_HISTORY:  # drop the oldest
        snaps, ops = snaps[-(MAX_HISTORY + 1):], ops[-MAX_HISTORY:]
    return EditHistory(snapshots=snaps, ops=ops, cursor=len(snaps) - 1)


def can_undo(h: EditHistory) -> bool:
    return h.cursor > 0


def can_redo(h: EditHistory) -> bool:
    return h.cursor < len(h.snapshots) - 1


def undo(h: EditHistory) -> EditHistory:
    return h.model_copy(update={"cursor": h.cursor - 1}) if can_undo(h) else h


def redo(h: EditHistory) -> EditHistory:
    return h.model_copy(update={"cursor": h.cursor + 1}) if can_redo(h) else h


# ---------- pure edit operations (mirror ClipOps in Arrangement.kt) ----------

def _replace_lane(p: Project, lane: Lane) -> Project:
    return p.model_copy(update={"lanes": tuple(lane if l.id == lane.id else l for l in p.lanes)})


def _lane_of_item(p: Project, item_id: str) -> Lane:
    return next(l for l in p.lanes if any(i.id == item_id for i in l.items))


def add_lane(p: Project, lane: Lane) -> Project:
    return Project.model_validate({**p.model_dump(), "lanes": [*p.model_dump()["lanes"], lane.model_dump()]})


def set_lane(p: Project, lane_id: str, **fields) -> Project:
    lane = p.lane(lane_id)
    return _revalidate(_replace_lane(p, lane.model_copy(update=fields)))


def add_item(p: Project, lane_id: str, item: LaneItem) -> Project:
    lane = p.lane(lane_id)
    return _revalidate(_replace_lane(p, lane.model_copy(update={"items": lane.items + (item,)})))


def set_item(p: Project, item_id: str, **fields) -> Project:
    lane = _lane_of_item(p, item_id)
    items = tuple(i.model_copy(update=fields) if i.id == item_id else i for i in lane.items)
    return _revalidate(_replace_lane(p, lane.model_copy(update={"items": items})))


def move_item(p: Project, item_id: str, lane_start_sample: int, to_lane_id: Optional[str] = None) -> Project:
    src = _lane_of_item(p, item_id)
    item = next(i for i in src.items if i.id == item_id).model_copy(update={"laneStartSample": max(0, lane_start_sample)})
    if to_lane_id is None or to_lane_id == src.id:
        return set_item(p, item_id, laneStartSample=item.laneStartSample)
    p = _replace_lane(p, src.model_copy(update={"items": tuple(i for i in src.items if i.id != item_id)}))
    return add_item(p, to_lane_id, item)


def delete_item(p: Project, item_id: str) -> Project:
    lane = _lane_of_item(p, item_id)
    return _replace_lane(p, lane.model_copy(update={"items": tuple(i for i in lane.items if i.id != item_id)}))


def split_item(p: Project, item_id: str, at_lane_sample: int, new_id: str) -> Project:
    """Split at a timeline sample: left keeps id, right gets new_id. Sample-exact (left.len + right.len == old.len)."""
    lane = _lane_of_item(p, item_id)
    it = next(i for i in lane.items if i.id == item_id)
    k = at_lane_sample - it.laneStartSample
    if k <= 0 or k >= it.range.length:
        raise ValueError("split point must be strictly inside the item")
    cut = it.range.startSample + k
    left = it.model_copy(update={"range": Range(startSample=it.range.startSample, endSample=cut), "fadeOutMs": 0.0})
    right = it.model_copy(update={"id": new_id, "range": Range(startSample=cut, endSample=it.range.endSample),
                                  "laneStartSample": at_lane_sample, "fadeInMs": 0.0})
    items = []
    for i in lane.items:
        items.extend([left, right] if i.id == item_id else [i])
    return _revalidate(_replace_lane(p, lane.model_copy(update={"items": tuple(items)})))


def add_processor(p: Project, lane_id: str, proc: Processor) -> Project:
    lane = p.lane(lane_id)
    return _revalidate(_replace_lane(p, lane.model_copy(update={"processorChain": lane.processorChain + (proc,)})))


def set_processor(p: Project, proc_id: str, **fields) -> Project:
    lane = next(l for l in p.lanes if any(x.id == proc_id for x in l.processorChain))
    chain = tuple(x.model_copy(update=fields) if x.id == proc_id else x for x in lane.processorChain)
    return _revalidate(_replace_lane(p, lane.model_copy(update={"processorChain": chain})))


def _revalidate(p: Project) -> Project:
    # model_copy skips validation; round-trip so ids/refs/ranges are checked on every edit.
    return Project.model_validate(p.model_dump())
