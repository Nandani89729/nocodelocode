"""Draft-tolerant workflow definition checks shared by every save path.

Both the UI save route (`PUT /workflow/{id}`, `POST /workflow/create`) and
the SDK/MCP authoring service (`services/workflow/sdk_authoring.py`) run
these before persisting a draft. They deliberately do not require a
complete, DTO-valid workflow, so partially-built drafts still save.
"""

from __future__ import annotations

from typing import Optional

from api.services.workflow.errors import WorkflowError
from api.services.workflow.workflow_graph import (
    validate_node_instance_constraints,
    validate_unique_transition_tool_names,
)


def node_instance_validation_errors(
    workflow_definition: Optional[dict],
) -> list[WorkflowError]:
    """Validate spec-driven max_instances without requiring a complete draft."""
    if not workflow_definition:
        return []
    nodes = workflow_definition.get("nodes")
    if not isinstance(nodes, list):
        return []

    node_types = [
        node.get("type")
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("type"), str)
    ]
    return validate_node_instance_constraints(
        node_types,
        enforce_min_instances=False,
    )


def transition_tool_name_validation_errors(
    workflow_definition: Optional[dict],
) -> list[WorkflowError]:
    """Validate transition names without requiring a complete draft DTO."""
    if not workflow_definition:
        return []
    edges = workflow_definition.get("edges")
    if not isinstance(edges, list):
        return []

    transitions: list[tuple[str, str, str]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        edge_id = edge.get("id")
        source = edge.get("source")
        data = edge.get("data")
        label = data.get("label") if isinstance(data, dict) else None
        if (
            isinstance(edge_id, str)
            and isinstance(source, str)
            and isinstance(label, str)
            and label
        ):
            transitions.append((edge_id, source, label))

    return validate_unique_transition_tool_names(transitions)
