import json
import sqlite3
import pytest
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3
from magda_agent.operations.cron_export_v1 import HermesCronExporterV1, ScheduledTaskConfig


@pytest.mark.asyncio
async def test_cron_exporter_empty_db():
    conn = sqlite3.connect(":memory:")
    exporter = HermesCronExporterV1(conn)

    tasks = exporter.fetch_tasks()
    assert tasks == []

    jobs_dict = exporter.export_jobs_dict()
    assert jobs_dict == []

    jobs_json = exporter.export_jobs_json()
    data = json.loads(jobs_json)
    assert data == {"scheduled_tasks": []}

    crontab = exporter.export_crontab()
    assert "# Hermes Agent Exported Crontab" in crontab

    systemd_units = exporter.export_systemd_units()
    assert systemd_units == {}


@pytest.mark.asyncio
async def test_cron_exporter_with_scheduler():
    scheduler = HermesCronSchedulerV3(db_path=":memory:")

    async def dummy_backup():
        pass

    async def dummy_report():
        pass

    scheduler.schedule("0 2 * * *", dummy_backup, name="nightly_backup")
    scheduler.schedule("0 8 * * *", dummy_report, name="daily_report")

    exporter = HermesCronExporterV1(scheduler._get_conn())

    tasks = exporter.fetch_tasks()
    assert len(tasks) == 2
    task_names = [t.name for t in tasks]
    assert "daily_report" in task_names
    assert "nightly_backup" in task_names

    jobs_dict = exporter.export_jobs_dict()
    assert len(jobs_dict) == 2

    jobs_json = exporter.export_jobs_json()
    parsed = json.loads(jobs_json)
    assert "scheduled_tasks" in parsed
    assert len(parsed["scheduled_tasks"]) == 2

    crontab = exporter.export_crontab(command_prefix="python -m runner")
    assert "0 8 * * * python -m runner --task daily_report" in crontab
    assert "0 2 * * * python -m runner --task nightly_backup" in crontab

    units = exporter.export_systemd_units()
    assert "daily_report" in units
    assert "nightly_backup" in units

    daily_report_units = units["daily_report"]
    assert "service" in daily_report_units
    assert "timer" in daily_report_units
    assert "OnCalendar=0 8 * * *" in daily_report_units["timer"]
    assert "ExecStart=/usr/bin/python3 -m magda_agent.operations.runner --task daily_report" in daily_report_units["service"]

    await scheduler.stop()


def test_scheduled_task_config_dict():
    config = ScheduledTaskConfig(
        name="test_job",
        cron_expr="*/5 * * * *",
        last_run="2026-01-01T00:00:00",
        next_run="2026-01-01T00:05:00"
    )

    d = config.to_dict()
    assert d == {
        "name": "test_job",
        "cron_expr": "*/5 * * * *",
        "last_run": "2026-01-01T00:00:00",
        "next_run": "2026-01-01T00:05:00"
    }
