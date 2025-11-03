from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002_create_detections_events"
down_revision = "0001_create_core_tables"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "detections",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("video_source_id", sa.Integer, sa.ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("tracker_id", sa.String(length=64), nullable=True),  # bytetrack id
        sa.Column("cls", sa.String(length=64), nullable=False),        # "person"
        sa.Column("conf", sa.Float, nullable=True),
        # normalized xyxy in [0,1]
        sa.Column("x1", sa.Float, nullable=False),
        sa.Column("y1", sa.Float, nullable=False),
        sa.Column("x2", sa.Float, nullable=False),
        sa.Column("y2", sa.Float, nullable=False),
        sa.Column("inside_area_id", sa.Integer, sa.ForeignKey("areas.id", ondelete="SET NULL"), nullable=True),
    )
    op.create_index("ix_detections_vs_ts", "detections", ["video_source_id", "timestamp"])

    op.create_table(
        "events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("video_source_id", sa.Integer, sa.ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("area_id", sa.Integer, sa.ForeignKey("areas.id", ondelete="CASCADE"), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("tracker_id", sa.String(length=64), nullable=True),
        sa.Column("type", sa.String(length=16), nullable=False),  # "enter" or "exit"
    )
    op.create_index("ix_events_area_ts", "events", ["area_id", "timestamp"])


def downgrade():
    op.drop_index("ix_events_area_ts", table_name="events")
    op.drop_table("events")
    op.drop_index("ix_detections_vs_ts", table_name="detections")
    op.drop_table("detections")