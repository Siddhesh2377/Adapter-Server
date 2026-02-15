"""add_server_defaults

Revision ID: 8b3646c8eaff
Revises: ddc29ed28bd6
Create Date: 2026-02-16 02:03:50.838525

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8b3646c8eaff'
down_revision: Union[str, Sequence[str], None] = 'ddc29ed28bd6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
