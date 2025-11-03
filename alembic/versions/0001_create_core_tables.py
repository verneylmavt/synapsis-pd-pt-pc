from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "0001_create_core_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "video_sources",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("uri", sa.Text, nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "areas",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("video_source_id", sa.Integer, sa.ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("polygon", JSONB, nullable=False),  # list of {x:float, y:float} normalized 0..1
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_index("ix_areas_video_source_id", "areas", ["video_source_id"])


def downgrade():
    op.drop_index("ix_areas_video_source_id", table_name="areas")
    op.drop_table("areas")
    op.drop_table("video_sources")