"""Разбор очереди QaGenerationJob. Отдельного Celery в проекте нет."""
from __future__ import annotations

import logging
import os
import time

from app.db import SessionLocal
from app.services.qa_jobs import process_queued

logger = logging.getLogger("most.worker")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    idle = float(os.getenv("MOST_WORKER_INTERVAL") or "5")
    while True:
        db = SessionLocal()
        try:
            jobs = process_queued(db, limit=1)
        except Exception:
            db.rollback()
            logger.exception("worker_tick_failed")
            jobs = []
            time.sleep(idle)
        else:
            if not jobs:
                time.sleep(idle)
            elif any(getattr(job, "status", "") == "failed" for job in jobs):
                logger.info("worker_job_failed n=%s", len(jobs))
                time.sleep(max(idle, 30))
            else:
                logger.info("worker_job_done n=%s", len(jobs))
        finally:
            db.close()


if __name__ == "__main__":
    main()
