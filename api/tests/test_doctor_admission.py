"""Doctor job admission: non-blocking Redis semaphore + arq Retry (D27)."""

from types import SimpleNamespace

import pytest
from arq import Retry

from api.services.workflow.doctor.contracts import (
    DOCTOR_ADMISSION_MAX_TRIES,
    DOCTOR_ADMISSION_RETRY_DEFER_SECONDS,
    DOCTOR_GLOBAL_CONCURRENCY,
    DOCTOR_JOB_TIMEOUT_SECONDS,
)
from api.tasks import doctor as doctor_task


class FakeRedis:
    """The subset of redis.asyncio the limiter uses (SET NX EX / GET / DEL)."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        self.ttls[key] = ex
        return True

    async def get(self, key):
        value = self.store.get(key)
        return value.encode() if value is not None else None

    async def delete(self, key):
        self.store.pop(key, None)
        return 1


@pytest.fixture
def fake_db(monkeypatch):
    db = SimpleNamespace(updates=[])

    async def update_analysis_report(report_uuid, organization_id, **kwargs):
        db.updates.append(kwargs)

    db.update_analysis_report = update_analysis_report
    monkeypatch.setattr(doctor_task, "db_client", db)
    return db


@pytest.fixture
def analysis_calls(monkeypatch):
    calls = []

    async def fake_run_analysis(report_uuid, organization_id):
        calls.append((report_uuid, organization_id))
        return "completed"

    monkeypatch.setattr(doctor_task, "run_analysis", fake_run_analysis)
    return calls


async def test_global_limiter_full_raises_retry_and_holds_no_slot(
    fake_db, analysis_calls
):
    redis = FakeRedis()
    for i in range(DOCTOR_GLOBAL_CONCURRENCY):
        redis.store[doctor_task.GLOBAL_SLOT_KEY.format(slot=i)] = f"other-{i}"
    before = dict(redis.store)

    with pytest.raises(Retry) as exc:
        await doctor_task.analyze_workflow_calls(
            {"redis": redis, "job_try": 1}, "rep-1", 7
        )
    assert exc.value.defer_score == DOCTOR_ADMISSION_RETRY_DEFER_SECONDS * 1000
    assert redis.store == before  # the org slot taken first was given back
    assert analysis_calls == [] and fake_db.updates == []  # report stays queued


async def test_org_limiter_full_retries_with_backoff(fake_db, analysis_calls):
    redis = FakeRedis()
    redis.store[doctor_task.ORG_SLOT_KEY.format(org=7, slot=0)] = "other"
    with pytest.raises(Retry) as exc:
        await doctor_task.analyze_workflow_calls(
            {"redis": redis, "job_try": 3}, "rep-1", 7
        )
    assert exc.value.defer_score == DOCTOR_ADMISSION_RETRY_DEFER_SECONDS * 4 * 1000
    assert not any(k.startswith("doctor:sem:global") for k in redis.store)


async def test_other_org_is_admitted_and_slots_released(fake_db, analysis_calls):
    redis = FakeRedis()
    redis.store[doctor_task.ORG_SLOT_KEY.format(org=7, slot=0)] = "other"
    out = await doctor_task.analyze_workflow_calls(
        {"redis": redis, "job_try": 1}, "rep-2", 8
    )
    assert out == "completed" and analysis_calls == [("rep-2", 8)]
    assert redis.store == {doctor_task.ORG_SLOT_KEY.format(org=7, slot=0): "other"}


async def test_slots_taken_with_job_timeout_ttl_and_released_on_error(
    fake_db, monkeypatch
):
    redis = FakeRedis()
    seen = {}

    async def boom(report_uuid, organization_id):
        seen.update(redis.ttls)
        raise RuntimeError("db down")

    monkeypatch.setattr(doctor_task, "run_analysis", boom)
    out = await doctor_task.analyze_workflow_calls(
        {"redis": redis, "job_try": 1}, "rep-3", 7
    )
    assert out == "failed"
    assert set(seen.values()) == {DOCTOR_JOB_TIMEOUT_SECONDS} and len(seen) == 2
    assert redis.store == {}
    assert fake_db.updates[-1]["status"].value == "failed"


async def test_gives_up_after_max_tries(fake_db, analysis_calls):
    redis = FakeRedis()
    for i in range(DOCTOR_GLOBAL_CONCURRENCY):
        redis.store[doctor_task.GLOBAL_SLOT_KEY.format(slot=i)] = "other"
    out = await doctor_task.analyze_workflow_calls(
        {"redis": redis, "job_try": DOCTOR_ADMISSION_MAX_TRIES}, "rep-4", 7
    )
    assert out == "failed"
    assert fake_db.updates[-1]["status"].value == "failed"
    assert analysis_calls == []


async def _fill_global_slots(redis):
    for i in range(DOCTOR_GLOBAL_CONCURRENCY):
        await redis.set(f"doctor:sem:global:{i}", "busy", nx=True, ex=60)


async def test_verify_job_shares_admission_and_frees_slot(monkeypatch):
    """Verification uses the same non-blocking limiter as analysis (D27)."""
    from api.tasks import doctor_verify

    redis = FakeRedis()
    await _fill_global_slots(redis)
    ran = []

    async def fake_verify_fix(**kwargs):
        ran.append(kwargs)

    monkeypatch.setattr(doctor_verify, "verify_fix", fake_verify_fix)
    with pytest.raises(Retry):
        await doctor_verify.verify_doctor_fix(
            {"redis": redis, "job_try": 1}, scenario_id=5, organization_id=9
        )
    assert ran == []
    # Only the pre-filled global slots remain: nothing leaked from the attempt.
    assert set(redis.store) == {
        f"doctor:sem:global:{i}" for i in range(DOCTOR_GLOBAL_CONCURRENCY)
    }


async def test_verify_job_marks_failed_when_admission_exhausted(monkeypatch):
    from api.tasks import doctor_verify

    redis = FakeRedis()
    await _fill_global_slots(redis)
    failed = []

    async def fake_mark_failed(scenario_id, organization_id, message):
        failed.append((scenario_id, organization_id))

    monkeypatch.setattr(doctor_verify, "mark_verification_failed", fake_mark_failed)
    result = await doctor_verify.verify_doctor_fix(
        {"redis": redis, "job_try": DOCTOR_ADMISSION_MAX_TRIES},
        scenario_id=5,
        organization_id=9,
    )
    assert result["status"] == "failed"
    assert failed == [(5, 9)]
