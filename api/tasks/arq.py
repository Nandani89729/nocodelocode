"""ARQ worker configuration - setup logging before importing tasks"""

import ssl
from urllib.parse import urlparse

from api.constants import REDIS_URL, TEXT_CHAT_INACTIVITY_SWEEP_INTERVAL_MINUTES

# Setup logging - this is now idempotent and safe to call multiple times
from api.logging_config import setup_logging
from api.tasks.function_names import FunctionNames

setup_logging()

# Now import ARQ and task dependencies
from arq import create_pool, cron, func
from arq.connections import ArqRedis, RedisSettings
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

parsed_url = urlparse(REDIS_URL)

# Check if we're using TLS (rediss://)
use_ssl = parsed_url.scheme == "rediss"

# Create SSL context if using rediss://
ssl_context = None
if use_ssl:
    ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

# A Redis blip must not be fatal to the worker. arq's own loop has no
# ConnectionError handling: a dropped connection propagates out of
# ``Worker.main()``, ``Worker.run()`` then fails again inside its
# ``finally: close()``, and the process exits for good. That is how every arq
# worker in the cluster died on 2026-08-26 and stayed dead until restarted by
# hand -- nothing supervises them. Retrying inside the Redis client keeps a
# short outage (a service restart on the Redis host) from ever reaching arq.
# The budget below is roughly a minute of exponential backoff, which covers a
# restart while still surfacing a genuinely unreachable Redis.
_REDIS_RETRY = Retry(ExponentialBackoff(cap=10, base=0.1), retries=12)

REDIS_SETTINGS = RedisSettings(
    host=parsed_url.hostname or "localhost",
    port=parsed_url.port or 6379,
    password=parsed_url.password,
    conn_timeout=10,
    retry=_REDIS_RETRY,
    retry_on_timeout=True,
    retry_on_error=[RedisConnectionError, RedisTimeoutError],
    ssl=use_ssl,
    ssl_ca_certs=None if not use_ssl else None,
    ssl_certfile=None,
    ssl_keyfile=None,
    ssl_check_hostname=False if use_ssl else None,
)

from api.services.workflow.doctor.contracts import (
    DOCTOR_ADMISSION_MAX_TRIES,
    DOCTOR_JOB_TIMEOUT_SECONDS,
)
from api.tasks.campaign_tasks import (
    process_campaign_batch,
    sync_campaign_source,
)
from api.tasks.doctor import analyze_workflow_calls
from api.tasks.doctor_verify import verify_doctor_fix
from api.tasks.knowledge_base_processing import process_knowledge_base_document
from api.tasks.run_integrations import run_integrations_post_workflow_run
from api.tasks.self_improve import (
    self_improve_iteration,
    sweep_stuck_self_improve_rounds,
)
from api.tasks.text_chat_inactivity import (
    complete_inactive_text_chat_session,
    sweep_inactive_text_chat_sessions,
)
from api.tasks.webhook_delivery import deliver_webhook, sweep_webhook_deliveries
from api.tasks.workflow_completion import process_workflow_completion


class WorkerSettings:
    functions = [
        run_integrations_post_workflow_run,
        process_workflow_completion,
        sync_campaign_source,
        process_campaign_batch,
        process_knowledge_base_document,
        deliver_webhook,
        complete_inactive_text_chat_session,
        # Agent Doctor: long-running; non-blocking admission retries need more
        # tries than arq's default of 5.
        func(
            analyze_workflow_calls,
            name=FunctionNames.ANALYZE_WORKFLOW_CALLS,
            timeout=DOCTOR_JOB_TIMEOUT_SECONDS,
            max_tries=DOCTOR_ADMISSION_MAX_TRIES + 1,
        ),
        func(
            verify_doctor_fix,
            name=FunctionNames.VERIFY_DOCTOR_FIX,
            timeout=DOCTOR_JOB_TIMEOUT_SECONDS,
            max_tries=DOCTOR_ADMISSION_MAX_TRIES + 1,
        ),
        # Self-improve round: Doctor + one copilot turn under the Doctor's
        # admission permit (RI-6). The job id is deterministic per round, so
        # no result is kept: the stuck-round sweep must be able to
        # re-enqueue a round whose job timed out (the job is idempotent).
        func(
            self_improve_iteration,
            name=FunctionNames.SELF_IMPROVE_ITERATION,
            timeout=DOCTOR_JOB_TIMEOUT_SECONDS,
            max_tries=DOCTOR_ADMISSION_MAX_TRIES + 1,
            keep_result=0,
        ),
    ]
    cron_jobs = [
        # Safety net for webhook deliveries whose ARQ job was lost (worker
        # restart / Redis flush): re-enqueue any pending delivery that is overdue.
        cron(
            sweep_webhook_deliveries,
            minute=set(range(0, 60, 5)),
            second=0,
            run_at_startup=True,
        ),
        # Text chats do not have a transport disconnect callback. Close stale
        # sessions so their transcript and completion integrations are finalized.
        cron(
            sweep_inactive_text_chat_sessions,
            minute=set(range(0, 60, TEXT_CHAT_INACTIVITY_SWEEP_INTERVAL_MINUTES)),
            second=30,
            run_at_startup=True,
        ),
        # Self-improve rounds stuck in queued/diagnosing/proposing past the
        # job timeout (worker restart, lost job) are re-enqueued (O5).
        cron(
            sweep_stuck_self_improve_rounds,
            minute=set(range(0, 60, 5)),
            second=45,
            run_at_startup=True,
        ),
    ]
    redis_settings = REDIS_SETTINGS
    max_jobs = 10


LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    # --- Handlers ---
    "handlers": {
        "console": {  # everything goes to stdout
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "level": "WARNING",  # only WARNING and above
            "formatter": "simple",
        },
    },
    # --- Formatters (optional) ---
    "formatters": {
        "simple": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        },
    },
    # --- Root logger ---
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    # --- Optionally silence Arq itself explicitly ---
    "loggers": {
        "arq": {  # arq.* loggers
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}


_redis_pool: ArqRedis | None = None


async def get_arq_redis() -> ArqRedis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = await create_pool(REDIS_SETTINGS)
    return _redis_pool


async def enqueue_job(function_name: FunctionNames, *args, **kwargs):
    redis = await get_arq_redis()
    # kwargs forwards ARQ job options (e.g. _job_id, _defer_by) used for
    # deterministic, backed-off webhook delivery retries.
    return await redis.enqueue_job(function_name, *args, **kwargs)
