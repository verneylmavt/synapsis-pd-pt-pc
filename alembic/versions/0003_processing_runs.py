"""Add durable runs without changing legacy detection/event identities."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003"
down_revision = "0002_create_detections_events"
branch_labels = None
depends_on = None


def upgrade():
    for name, type_ in (("kind", sa.String(16)), ("width", sa.Integer()), ("height", sa.Integer()),
                        ("fps", sa.Float()), ("duration_seconds", sa.Float()), ("size_bytes", sa.Integer())):
        op.add_column("video_sources", sa.Column(name, type_, nullable=name != "kind", server_default="file" if name == "kind" else None))
    op.execute("UPDATE video_sources SET kind='live' WHERE uri ~* '^(rtsp|https?)://'")
    op.create_table("processing_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("video_source_id", sa.Integer(), sa.ForeignKey("video_sources.id"), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False), sa.Column("status", sa.String(16), nullable=False),
        sa.Column("owner_id", sa.String(36), nullable=False), sa.Column("settings", JSONB, nullable=False),
        sa.Column("areas", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        *[sa.Column(n, sa.DateTime(timezone=True)) for n in ("started_at", "ended_at", "heartbeat_at")],
        sa.Column("last_frame_index", sa.Integer(), server_default="0", nullable=False),
        sa.Column("segment_index", sa.Integer(), server_default="0", nullable=False),
        sa.Column("media_time_ms", sa.Integer()),
        sa.Column("fps", sa.Float(), server_default="0", nullable=False),
        sa.Column("dropped_frames", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_code", sa.String(64)),
        sa.CheckConstraint("status IN ('starting','running','reconnecting','stopping','completed','stopped','failed','interrupted')", name="ck_run_status"))
    op.create_index("uq_active_source_run", "processing_runs", ["video_source_id"], unique=True,
                    postgresql_where=sa.text("status IN ('starting','running','reconnecting','stopping')"))
    op.create_index("ix_runs_source_created", "processing_runs", ["video_source_id", "created_at"])
    for table in ("detections", "events"):
        op.add_column(table, sa.Column("run_id", sa.String(36), sa.ForeignKey("processing_runs.id", ondelete="CASCADE")))
        for name in ("frame_index", "segment_index", "media_time_ms"):
            op.add_column(table, sa.Column(name, sa.Integer()))
    op.add_column("detections", sa.Column("detection_index", sa.Integer()))
    op.create_unique_constraint("uq_detection_run_frame", "detections", ["run_id", "frame_index", "detection_index"])
    op.create_check_constraint("ck_detection_run_frame", "detections", "run_id IS NULL OR (frame_index >= 1 AND detection_index >= 0 AND segment_index >= 0)")
    op.create_unique_constraint("uq_event_run_frame", "events", ["run_id", "segment_index", "frame_index", "area_id", "tracker_id", "type"])
    op.create_check_constraint("ck_event_run_frame", "events", "run_id IS NULL OR (type IN ('enter','exit') AND frame_index >= 1 AND segment_index >= 0 AND tracker_id IS NOT NULL)")
    op.create_table("run_area_stats", sa.Column("run_id", sa.String(36), sa.ForeignKey("processing_runs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("area_id", sa.Integer(), primary_key=True),
        *[sa.Column(n, sa.Integer(), nullable=False, server_default="0") for n in ("total_in", "total_out", "currently_inside", "lost_tracks")])
    op.create_table("frame_samples", sa.Column("run_id", sa.String(36), sa.ForeignKey("processing_runs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("frame_index", sa.Integer(), primary_key=True), sa.Column("time_seconds", sa.Float(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False), sa.Column("stats", JSONB, nullable=False))


def downgrade():
    op.drop_table("frame_samples")
    op.drop_table("run_area_stats")
    for table, constraint in (("events", "uq_event_run_frame"), ("detections", "uq_detection_run_frame")):
        op.drop_constraint(constraint, table, type_="unique")
        op.drop_constraint("ck_event_run_frame" if table == "events" else "ck_detection_run_frame", table, type_="check")
        for name in ("run_id", "frame_index", "segment_index", "media_time_ms"):
            op.drop_column(table, name)
    op.drop_column("detections", "detection_index")
    op.drop_table("processing_runs")
    for name in ("kind", "width", "height", "fps", "duration_seconds", "size_bytes"):
        op.drop_column("video_sources", name)
