# coding=utf-8
"""Tests for TaskBoard."""

from self_ai.agent_runtime.task_board import TaskBoard


def test_task_board_create_claim_update_list_get() -> None:
    board = TaskBoard()
    t = board.create_task(goal="Implement", agent_name="coder")
    task_id = t["task_id"]

    claimed = board.claim_task(task_id, claimer="runner")
    assert claimed is not None
    assert claimed["status"] == "claimed"

    updated = board.update_task(task_id, status="running", metadata={"progress": 20})
    assert updated is not None
    assert updated["status"] == "running"
    assert updated["metadata"]["progress"] == 20

    done = board.update_task(task_id, status="done")
    assert done is not None
    assert done["status"] == "done"

    assert board.get_task(task_id) is not None
    assert len(board.list_tasks()) == 1
    assert len(board.list_tasks(status="done")) == 1

