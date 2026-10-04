import pytest

pytestmark = pytest.mark.integration


def test_migrated_schema_matches_orm_metadata(db_engine):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from app.db.models import Base
    with db_engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []
