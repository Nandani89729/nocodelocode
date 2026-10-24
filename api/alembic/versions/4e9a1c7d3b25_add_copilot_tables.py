"""add agent copilot tables

Adds copilot_sessions (Agent Copilot chats, per workflow or build-new) and
copilot_proposals (reviewable change sets proposed in a session).

Revision ID: 4e9a1c7d3b25
Revises: 8c2d5e7f1a43
Create Date: 2026-10-01 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4e9a1c7d3b25"
down_revision: Union[str, None] = "8c2d5e7f1a43"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "copilot_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_uuid", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column(
            "transcript",
            sa.JSON(),
            server_default=sa.text("'[]'::json"),
            nullable=False,
        ),
        sa.Column("plan", sa.JSON(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_copilot_sessions_id"), "copilot_sessions", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_copilot_sessions_session_uuid"),
        "copilot_sessions",
        ["session_uuid"],
        unique=True,
    )
    op.create_index(
        "ix_copilot_sessions_organization_id",
        "copilot_sessions",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_copilot_sessions_workflow_updated",
        "copilot_sessions",
        ["workflow_id", "updated_at"],
        unique=False,
    )

    op.create_table(
        "copilot_proposals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("proposal_uuid", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.Integer(), nullable=True),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("summary", sa.Text(), server_default="", nullable=False),
        sa.Column("base_hash", sa.String(length=128), nullable=True),
        sa.Column(
            "candidate_json",
            sa.JSON(),
            server_default=sa.text("'{}'::json"),
            nullable=False,
        ),
        sa.Column("code", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "changes", sa.JSON(), server_default=sa.text("'[]'::json"), nullable=False
        ),
        sa.Column(
            "status", sa.String(length=20), server_default="pending", nullable=False
        ),
        sa.Column("workflow_name", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["session_id"], ["copilot_sessions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_copilot_proposals_id"), "copilot_proposals", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_copilot_proposals_proposal_uuid"),
        "copilot_proposals",
        ["proposal_uuid"],
        unique=True,
    )
    op.create_index(
        "ix_copilot_proposals_organization_id",
        "copilot_proposals",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_copilot_proposals_session_id",
        "copilot_proposals",
        ["session_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_copilot_proposals_session_id", table_name="copilot_proposals")
    op.drop_index(
        "ix_copilot_proposals_organization_id", table_name="copilot_proposals"
    )
    op.drop_index(
        op.f("ix_copilot_proposals_proposal_uuid"), table_name="copilot_proposals"
    )
    op.drop_index(op.f("ix_copilot_proposals_id"), table_name="copilot_proposals")
    op.drop_table("copilot_proposals")

    op.drop_index("ix_copilot_sessions_workflow_updated", table_name="copilot_sessions")
    op.drop_index("ix_copilot_sessions_organization_id", table_name="copilot_sessions")
    op.drop_index(
        op.f("ix_copilot_sessions_session_uuid"), table_name="copilot_sessions"
    )
    op.drop_index(op.f("ix_copilot_sessions_id"), table_name="copilot_sessions")
    op.drop_table("copilot_sessions")
