"""Models migration

Revision ID: f8eef7042623
Revises: 8b3646c8eaff
Create Date: 2026-02-16 14:17:16.072024

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f8eef7042623'
down_revision: Union[str, Sequence[str], None] = '8b3646c8eaff'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add new column as nullable first
    op.add_column('base_models', sa.Column('model_download_link', sa.String(length=500), nullable=True))

    # Copy data from old column to new column
    op.execute("UPDATE base_models SET model_download_link = huggingface_id WHERE huggingface_id IS NOT NULL")

    # Make the column NOT NULL now that it has data
    op.alter_column('base_models', 'model_download_link', nullable=False)

    # Drop old column
    op.drop_column('base_models', 'huggingface_id')


def downgrade() -> None:
    """Downgrade schema."""
    # Add old column back as nullable
    op.add_column('base_models', sa.Column('huggingface_id', sa.VARCHAR(length=500), autoincrement=False, nullable=True))

    # Copy data back
    op.execute("UPDATE base_models SET huggingface_id = model_download_link WHERE model_download_link IS NOT NULL")

    # Make it NOT NULL
    op.alter_column('base_models', 'huggingface_id', nullable=False)

    # Drop new column
    op.drop_column('base_models', 'model_download_link')
