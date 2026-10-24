"""add copilot proposal settings and new tools

Milestone 2c: a Copilot proposal may also change the agent's settings
(workflow_configurations + template_context_variables) and create new
reusable tools. copilot_proposals gains:

- settings_candidate: {configurations, template_variables} as proposed
  (NULL for graph-only proposals)
- settings_base_hash: hash of the working copy's settings the proposal was
  made against (NULL for graph-only proposals; their staleness check is
  unchanged)
- new_tools: CreateToolRequest dumps of the tools the proposal creates

Revision ID: c3e8a5f27d14
Revises: b7d2f4a91c60
Create Date: 2026-10-03 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3e8a5f27d14"
down_revision: Union[str, None] = "b7d2f4a91c60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "copilot_proposals",
        sa.Column("settings_candidate", sa.JSON(), nullable=True),
    )
    op.add_column(
        "copilot_proposals",
        sa.Column("settings_base_hash", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "copilot_proposals",
        sa.Column(
            "new_tools",
            sa.JSON(),
            server_default=sa.text("'[]'::json"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("copilot_proposals", "new_tools")
    op.drop_column("copilot_proposals", "settings_base_hash")
    op.drop_column("copilot_proposals", "settings_candidate")
