"""Add hf_model_id to base_models

Revision ID: a1b2c3d4e5f6
Revises: 0c62556b8850
Create Date: 2026-02-16 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '0c62556b8850'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('base_models', sa.Column('hf_model_id', sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column('base_models', 'hf_model_id')
