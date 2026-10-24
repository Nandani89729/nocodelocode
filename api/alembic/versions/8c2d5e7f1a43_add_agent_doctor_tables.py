"""add agent doctor tables

Adds workflow_analysis_reports (Agent Doctor reports, 90-day retention) and
regression_scenarios (real-call turns pinned for fix verification).

Revision ID: 8c2d5e7f1a43
Revises: 3a7b91c5d402
Create Date: 2026-10-01 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8c2d5e7f1a43"
down_revision: Union[str, None] = "3a7b91c5d402"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "workflow_analysis_reports",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("report_uuid", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column(
            "status", sa.String(length=20), server_default="queued", nullable=False
        ),
        sa.Column(
            "params", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=False
        ),
        sa.Column(
            "result", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=False
        ),
        sa.Column("error", sa.Text(), nullable=True),
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
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now() + interval '90 days'"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_workflow_analysis_reports_id"),
        "workflow_analysis_reports",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_workflow_analysis_reports_report_uuid"),
        "workflow_analysis_reports",
        ["report_uuid"],
        unique=True,
    )
    op.create_index(
        "ix_workflow_analysis_reports_organization_id",
        "workflow_analysis_reports",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_analysis_reports_workflow_created",
        "workflow_analysis_reports",
        ["workflow_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "regression_scenarios",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.Integer(), nullable=False),
        sa.Column("report_id", sa.Integer(), nullable=True),
        sa.Column("cluster_id", sa.String(length=64), nullable=True),
        sa.Column("source_run_id", sa.Integer(), nullable=True),
        sa.Column("turn", sa.Integer(), nullable=True),
        sa.Column("expected_behavior", sa.Text(), nullable=True),
        sa.Column(
            "expected_confirmed",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "verification",
            sa.JSON(),
            server_default=sa.text("'{}'::json"),
            nullable=False,
        ),
        sa.Column("candidate_definition_id", sa.Integer(), nullable=True),
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
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["report_id"], ["workflow_analysis_reports.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_run_id"], ["workflow_runs.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["candidate_definition_id"],
            ["workflow_definitions.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_regression_scenarios_id"),
        "regression_scenarios",
        ["id"],
        unique=False,
    )
    op.create_index(
        "ix_regression_scenarios_organization_id",
        "regression_scenarios",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_regression_scenarios_workflow_id",
        "regression_scenarios",
        ["workflow_id"],
        unique=False,
    )
    op.create_index(
        "ix_regression_scenarios_report_id",
        "regression_scenarios",
        ["report_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_regression_scenarios_report_id", table_name="regression_scenarios"
    )
    op.drop_index(
        "ix_regression_scenarios_workflow_id", table_name="regression_scenarios"
    )
    op.drop_index(
        "ix_regression_scenarios_organization_id", table_name="regression_scenarios"
    )
    op.drop_index(op.f("ix_regression_scenarios_id"), table_name="regression_scenarios")
    op.drop_table("regression_scenarios")

    op.drop_index(
        "ix_workflow_analysis_reports_workflow_created",
        table_name="workflow_analysis_reports",
    )
    op.drop_index(
        "ix_workflow_analysis_reports_organization_id",
        table_name="workflow_analysis_reports",
    )
    op.drop_index(
        op.f("ix_workflow_analysis_reports_report_uuid"),
        table_name="workflow_analysis_reports",
    )
    op.drop_index(
        op.f("ix_workflow_analysis_reports_id"),
        table_name="workflow_analysis_reports",
    )
    op.drop_table("workflow_analysis_reports")
