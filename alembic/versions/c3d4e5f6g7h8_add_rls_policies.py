"""Add RLS policies for all tables

Revision ID: c3d4e5f6g7h8
Revises: b2c3d4e5f6a7
Create Date: 2026-02-16 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6g7h8'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── base_models ──
    op.execute("ALTER TABLE public.base_models ENABLE ROW LEVEL SECURITY;")

    op.execute("DROP POLICY IF EXISTS \"public_read_active_base_models\" ON public.base_models;")
    op.execute("""
        CREATE POLICY "public_read_active_base_models"
        ON public.base_models
        FOR SELECT
        USING (is_active = true);
    """)

    op.execute("DROP POLICY IF EXISTS \"service_role_full_access_base_models\" ON public.base_models;")
    op.execute("""
        CREATE POLICY "service_role_full_access_base_models"
        ON public.base_models
        FOR ALL
        TO authenticated
        USING (auth.role() = 'service_role')
        WITH CHECK (auth.role() = 'service_role');
    """)

    # ── adapters ──
    op.execute("ALTER TABLE public.adapters ENABLE ROW LEVEL SECURITY;")

    op.execute("DROP POLICY IF EXISTS \"public_read_published_adapters\" ON public.adapters;")
    op.execute("""
        CREATE POLICY "public_read_published_adapters"
        ON public.adapters
        FOR SELECT
        USING (is_published = true AND status = 'active');
    """)

    op.execute("DROP POLICY IF EXISTS \"service_role_full_access_adapters\" ON public.adapters;")
    op.execute("""
        CREATE POLICY "service_role_full_access_adapters"
        ON public.adapters
        FOR ALL
        TO authenticated
        USING (auth.role() = 'service_role')
        WITH CHECK (auth.role() = 'service_role');
    """)

    # ── adapter_deployments ──
    op.execute("ALTER TABLE public.adapter_deployments ENABLE ROW LEVEL SECURITY;")

    op.execute("DROP POLICY IF EXISTS \"public_read_active_deployments\" ON public.adapter_deployments;")
    op.execute("""
        CREATE POLICY "public_read_active_deployments"
        ON public.adapter_deployments
        FOR SELECT
        USING (is_active = true);
    """)

    op.execute("DROP POLICY IF EXISTS \"service_role_full_access_deployments\" ON public.adapter_deployments;")
    op.execute("""
        CREATE POLICY "service_role_full_access_deployments"
        ON public.adapter_deployments
        FOR ALL
        TO authenticated
        USING (auth.role() = 'service_role')
        WITH CHECK (auth.role() = 'service_role');
    """)

    # ── update_logs ──
    op.execute("ALTER TABLE public.update_logs ENABLE ROW LEVEL SECURITY;")

    op.execute("DROP POLICY IF EXISTS \"devices_can_insert_logs\" ON public.update_logs;")
    op.execute("""
        CREATE POLICY "devices_can_insert_logs"
        ON public.update_logs
        FOR INSERT
        WITH CHECK (true);
    """)

    op.execute("DROP POLICY IF EXISTS \"devices_read_own_logs\" ON public.update_logs;")
    op.execute("""
        CREATE POLICY "devices_read_own_logs"
        ON public.update_logs
        FOR SELECT
        USING (device_id = current_setting('request.headers')::json->>'x-device-id');
    """)

    op.execute("DROP POLICY IF EXISTS \"service_role_full_access_logs\" ON public.update_logs;")
    op.execute("""
        CREATE POLICY "service_role_full_access_logs"
        ON public.update_logs
        FOR ALL
        TO authenticated
        USING (auth.role() = 'service_role')
        WITH CHECK (auth.role() = 'service_role');
    """)


def downgrade() -> None:
    # ── update_logs ──
    op.execute("DROP POLICY IF EXISTS \"service_role_full_access_logs\" ON public.update_logs;")
    op.execute("DROP POLICY IF EXISTS \"devices_read_own_logs\" ON public.update_logs;")
    op.execute("DROP POLICY IF EXISTS \"devices_can_insert_logs\" ON public.update_logs;")
    op.execute("ALTER TABLE public.update_logs DISABLE ROW LEVEL SECURITY;")

    # ── adapter_deployments ──
    op.execute("DROP POLICY IF EXISTS \"service_role_full_access_deployments\" ON public.adapter_deployments;")
    op.execute("DROP POLICY IF EXISTS \"public_read_active_deployments\" ON public.adapter_deployments;")
    op.execute("ALTER TABLE public.adapter_deployments DISABLE ROW LEVEL SECURITY;")

    # ── adapters ──
    op.execute("DROP POLICY IF EXISTS \"service_role_full_access_adapters\" ON public.adapters;")
    op.execute("DROP POLICY IF EXISTS \"public_read_published_adapters\" ON public.adapters;")
    op.execute("ALTER TABLE public.adapters DISABLE ROW LEVEL SECURITY;")

    # ── base_models ──
    op.execute("DROP POLICY IF EXISTS \"service_role_full_access_base_models\" ON public.base_models;")
    op.execute("DROP POLICY IF EXISTS \"public_read_active_base_models\" ON public.base_models;")
    op.execute("ALTER TABLE public.base_models DISABLE ROW LEVEL SECURITY;")
