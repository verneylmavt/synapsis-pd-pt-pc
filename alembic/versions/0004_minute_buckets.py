"""Durable minute summaries and non-null idempotency keys for identified runs."""
from collections import defaultdict
from math import floor

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, insert

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

FIELDS = ("coverage_seconds", "occupancy_seconds", "sample_count", "in_count", "out_count")
DETECTION_CHECK = "run_id IS NULL OR (frame_index IS NOT NULL AND detection_index IS NOT NULL AND segment_index IS NOT NULL AND frame_index >= 1 AND detection_index >= 0 AND segment_index >= 0)"
EVENT_CHECK = "run_id IS NULL OR (type IN ('enter','exit') AND frame_index IS NOT NULL AND segment_index IS NOT NULL AND frame_index >= 1 AND segment_index >= 0 AND tracker_id IS NOT NULL)"


def upgrade():
    table = op.create_table("run_buckets",
        sa.Column("run_id", sa.String(36), sa.ForeignKey("processing_runs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("area_id", sa.Integer(), primary_key=True), sa.Column("window_start", sa.Float(), primary_key=True),
        *[sa.Column(key, sa.Float() if key.endswith("seconds") else sa.Integer(), nullable=False, server_default="0") for key in FIELDS])
    connection = op.get_bind()
    # Stream the audit backfill; no run history is loaded in full into Python.
    samples = sa.table("frame_samples", sa.column("run_id"), sa.column("time_seconds"), sa.column("duration_seconds"), sa.column("stats", JSONB))
    result = connection.execute(sa.select(samples).order_by(samples.c.run_id, samples.c.time_seconds)
                                .execution_options(stream_results=True))
    previous_run, covered_until = None, float("-inf")
    for partition in result.mappings().partitions(500):
        grouped = defaultdict(lambda: dict.fromkeys(FIELDS, 0))
        for sample in partition:
            if sample["run_id"] != previous_run:
                previous_run, covered_until = sample["run_id"], float("-inf")
            start, end = sample["time_seconds"], sample["time_seconds"] + sample["duration_seconds"]
            # Older packets could overlap. Count the union once, so duplicate
            # coverage cannot turn a real gap into observed zero traffic.
            start = max(start, covered_until)
            covered_until = max(covered_until, end)
            left = floor(start / 60) * 60
            while left < end - 1e-9:
                overlap = max(0, min(end, left + 60) - max(start, left))
                for stat in sample["stats"]:
                    delta = grouped[(sample["run_id"], stat["area_id"], left)]
                    delta["coverage_seconds"] += overlap
                    delta["occupancy_seconds"] += overlap * stat["currently_inside"]
                    delta["sample_count"] += 1
                left += 60
        for (run_id, area_id, left), delta in grouped.items():
            stmt = insert(table).values(run_id=run_id, area_id=area_id, window_start=left, **delta)
            connection.execute(stmt.on_conflict_do_update(index_elements=["run_id", "area_id", "window_start"],
                set_={key: table.c[key] + stmt.excluded[key] for key in FIELDS}))
    op.execute("""INSERT INTO run_buckets(run_id,area_id,window_start,in_count,out_count)
        SELECT e.run_id,e.area_id,floor((CASE WHEN r.kind='file' AND e.media_time_ms IS NOT NULL THEN e.media_time_ms/1000.0 ELSE extract(epoch FROM e.timestamp) END)/60)*60,
            count(*) FILTER(WHERE e.type='enter'),count(*) FILTER(WHERE e.type='exit')
        FROM events e JOIN processing_runs r ON r.id=e.run_id GROUP BY 1,2,3
        ON CONFLICT(run_id,area_id,window_start) DO UPDATE SET in_count=excluded.in_count,out_count=excluded.out_count""")
    for table_name, name, condition in (("detections", "ck_detection_run_frame", DETECTION_CHECK), ("events", "ck_event_run_frame", EVENT_CHECK)):
        op.drop_constraint(name, table_name, type_="check")
        op.create_check_constraint(name, table_name, condition)


def downgrade():
    op.drop_table("run_buckets")
    for table_name, name, condition in (
        ("detections", "ck_detection_run_frame", "run_id IS NULL OR (frame_index >= 1 AND detection_index >= 0 AND segment_index >= 0)"),
        ("events", "ck_event_run_frame", "run_id IS NULL OR (type IN ('enter','exit') AND frame_index >= 1 AND segment_index >= 0 AND tracker_id IS NOT NULL)")):
        op.drop_constraint(name, table_name, type_="check")
        op.create_check_constraint(name, table_name, condition)
