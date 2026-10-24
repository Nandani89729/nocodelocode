"""add self-improve tables

Milestone 2b (Self-improve): workflow_briefs + workflow_brief_changes (the
workflow brief / North Star and copilot-proposed edits to it), self_improve
(per-workflow settings and spend guard), self_improve_iterations (rounds),
self_improve_runs (counted calls), copilot_sessions.kind, and the copilot
proposal fields iteration_id / accepted_change_ids / applied_definition_id /
hypotheses.

copilot_proposals.iteration_id and self_improve_iterations.proposal_id point
at each other; both FKs are named so downgrade can drop them in order.

Revision ID: b7d2f4a91c60
Revises: 4e9a1c7d3b25
Create Date: 2026-10-02 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7d2f4a91c60"
down_revision: Union[str, None] = "4e9a1c7d3b25"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    # -- copilot_sessions.kind --------------------------------------------
    op.add_column(
        "copilot_sessions",
        sa.Column("kind", sa.String(length=20), server_default="chat", nullable=False),
    )
    op.create_index(
        "uq_copilot_sessions_self_improve_workflow",
        "copilot_sessions",
        ["workflow_id"],
        unique=True,
        postgresql_where=sa.text("kind = 'self_improve'"),
    )

    # -- workflow_briefs ----------------------------------------------------
    op.create_table(
        "workflow_briefs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column(
            "brief", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=False
        ),
        sa.Column("version", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "history", sa.JSON(), server_default=sa.text("'[]'::json"), nullable=False
        ),
        sa.Column("updated_by", sa.String(length=20), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workflow_id"),
    )
    op.create_index(
        op.f("ix_workflow_briefs_id"), "workflow_briefs", ["id"], unique=False
    )
    op.create_index(
        "ix_workflow_briefs_organization_id",
        "workflow_briefs",
        ["organization_id"],
        unique=False,
    )

    # -- workflow_brief_changes ---------------------------------------------
    op.create_table(
        "workflow_brief_changes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("change_uuid", sa.String(length=36), nullable=False),
        sa.Column("workflow_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("base_version", sa.Integer(), nullable=False),
        sa.Column(
            "proposed", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=False
        ),
        sa.Column("reason", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "status", sa.String(length=20), server_default="pending", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_workflow_brief_changes_id"),
        "workflow_brief_changes",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_workflow_brief_changes_change_uuid"),
        "workflow_brief_changes",
        ["change_uuid"],
        unique=True,
    )
    op.create_index(
        "ix_workflow_brief_changes_organization_id",
        "workflow_brief_changes",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_brief_changes_workflow_status",
        "workflow_brief_changes",
        ["workflow_id", "status"],
        unique=False,
    )

    # -- self_improve ---------------------------------------------------------
    op.create_table(
        "self_improve",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column(
            "enabled", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "every_n_calls", sa.Integer(), server_default=sa.text("10"), nullable=False
        ),
        sa.Column("enabled_by_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "window_multiplier",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "consecutive_failures",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "paused", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("paused_reason", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["enabled_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workflow_id"),
    )
    op.create_index(op.f("ix_self_improve_id"), "self_improve", ["id"], unique=False)
    op.create_index(
        "ix_self_improve_organization_id",
        "self_improve",
        ["organization_id"],
        unique=False,
    )

    # -- self_improve_iterations ----------------------------------------------
    op.create_table(
        "self_improve_iterations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column(
            "status", sa.String(length=20), server_default="queued", nullable=False
        ),
        sa.Column(
            "run_ids", sa.JSON(), server_default=sa.text("'[]'::json"), nullable=False
        ),
        sa.Column("report_id", sa.Integer(), nullable=True),
        sa.Column("proposal_id", sa.Integer(), nullable=True),
        sa.Column(
            "carryover",
            sa.JSON(),
            server_default=sa.text("'[]'::json"),
            nullable=False,
        ),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        *_timestamps(),
        sa.Column(
            "stage_started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["report_id"], ["workflow_analysis_reports.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["proposal_id"],
            ["copilot_proposals.id"],
            ondelete="SET NULL",
            name="fk_self_improve_iterations_proposal_id",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "workflow_id", "number", name="uq_self_improve_iterations_workflow_number"
        ),
    )
    op.create_index(
        op.f("ix_self_improve_iterations_id"),
        "self_improve_iterations",
        ["id"],
        unique=False,
    )
    op.create_index(
        "ix_self_improve_iterations_organization_id",
        "self_improve_iterations",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_self_improve_iterations_status",
        "self_improve_iterations",
        ["status"],
        unique=False,
    )

    # -- self_improve_runs ------------------------------------------------------
    op.create_table(
        "self_improve_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column(
            "counted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("iteration_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["run_id"], ["workflow_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["iteration_id"], ["self_improve_iterations.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )
    op.create_index(
        op.f("ix_self_improve_runs_id"), "self_improve_runs", ["id"], unique=False
    )
    op.create_index(
        "ix_self_improve_runs_workflow_iteration",
        "self_improve_runs",
        ["workflow_id", "iteration_id"],
        unique=False,
    )

    # -- copilot_proposals: Self-improve fields -------------------------------
    op.add_column(
        "copilot_proposals", sa.Column("iteration_id", sa.Integer(), nullable=True)
    )
    op.add_column(
        "copilot_proposals",
        sa.Column(
            "accepted_change_ids",
            sa.JSON(),
            server_default=sa.text("'[]'::json"),
            nullable=False,
        ),
    )
    op.add_column(
        "copilot_proposals",
        sa.Column("applied_definition_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "copilot_proposals",
        sa.Column(
            "hypotheses",
            sa.JSON(),
            server_default=sa.text("'{}'::json"),
            nullable=False,
        ),
    )
    op.create_foreign_key(
        "fk_copilot_proposals_iteration_id",
        "copilot_proposals",
        "self_improve_iterations",
        ["iteration_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_copilot_proposals_applied_definition_id",
        "copilot_proposals",
        "workflow_definitions",
        ["applied_definition_id"],
        ["id"],
    )
    op.create_index(
        "uq_copilot_proposals_iteration_id",
        "copilot_proposals",
        ["iteration_id"],
        unique=True,
        postgresql_where=sa.text("iteration_id IS NOT NULL"),
    )


def downgrade() -> None:
    # copilot_proposals first: its iteration_id FK points at the iterations.
    op.drop_index("uq_copilot_proposals_iteration_id", table_name="copilot_proposals")
    op.drop_constraint(
        "fk_copilot_proposals_applied_definition_id",
        "copilot_proposals",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_copilot_proposals_iteration_id", "copilot_proposals", type_="foreignkey"
    )
    op.drop_column("copilot_proposals", "hypotheses")
    op.drop_column("copilot_proposals", "applied_definition_id")
    op.drop_column("copilot_proposals", "accepted_change_ids")
    op.drop_column("copilot_proposals", "iteration_id")

    op.drop_index(
        "ix_self_improve_runs_workflow_iteration", table_name="self_improve_runs"
    )
    op.drop_index(op.f("ix_self_improve_runs_id"), table_name="self_improve_runs")
    op.drop_table("self_improve_runs")

    op.drop_index(
        "ix_self_improve_iterations_status", table_name="self_improve_iterations"
    )
    op.drop_index(
        "ix_self_improve_iterations_organization_id",
        table_name="self_improve_iterations",
    )
    op.drop_index(
        op.f("ix_self_improve_iterations_id"), table_name="self_improve_iterations"
    )
    op.drop_table("self_improve_iterations")

    op.drop_index("ix_self_improve_organization_id", table_name="self_improve")
    op.drop_index(op.f("ix_self_improve_id"), table_name="self_improve")
    op.drop_table("self_improve")

    op.drop_index(
        "ix_workflow_brief_changes_workflow_status",
        table_name="workflow_brief_changes",
    )
    op.drop_index(
        "ix_workflow_brief_changes_organization_id",
        table_name="workflow_brief_changes",
    )
    op.drop_index(
        op.f("ix_workflow_brief_changes_change_uuid"),
        table_name="workflow_brief_changes",
    )
    op.drop_index(
        op.f("ix_workflow_brief_changes_id"), table_name="workflow_brief_changes"
    )
    op.drop_table("workflow_brief_changes")

    op.drop_index("ix_workflow_briefs_organization_id", table_name="workflow_briefs")
    op.drop_index(op.f("ix_workflow_briefs_id"), table_name="workflow_briefs")
    op.drop_table("workflow_briefs")

    op.drop_index(
        "uq_copilot_sessions_self_improve_workflow", table_name="copilot_sessions"
    )
    op.drop_column("copilot_sessions", "kind")
