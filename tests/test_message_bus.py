# coding=utf-8
"""Tests for MessageBus."""

from self_ai.agent_runtime.message_bus import MessageBus


def test_message_bus_send_inbox_broadcast_list() -> None:
    bus = MessageBus()
    msg = bus.send(sender="a", recipient="b", content="hello")
    assert msg["recipient"] == "b"
    assert len(bus.inbox("b")) == 1

    sent = bus.broadcast(sender="system", recipients=["x", "y"], content="sync")
    assert len(sent) == 2
    assert len(bus.inbox("x")) == 1
    assert len(bus.inbox("y")) == 1
    assert len(bus.list_messages()) == 3

