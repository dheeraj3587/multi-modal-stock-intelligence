import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from typing import Optional

logger = logging.getLogger(__name__)

class SchedulerService:
    _instance: Optional['SchedulerService'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SchedulerService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Ensure init only runs once
        if getattr(self, "_initialized", False):
            return
            
        self.scheduler = AsyncIOScheduler()
        self._initialized = True
        logger.info("SchedulerService initialized (Singleton)")
        
    def start(self):
        """Start the scheduler if not already running."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("🚀 Scheduler started")
        else:
            logger.warning("Scheduler already running")
            
    def shutdown(self):
        """Shutdown the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("👋 Scheduler shutdown")
            
    def add_job(self, func, trigger_type="interval", id=None, replace_existing=True, **trigger_args):
        """
        Add a job to the scheduler.
        
        Args:
            func: The async function to execute
            trigger_type: 'interval', 'date', or 'cron'
            id: Unique job ID (prevents duplicates)
            replace_existing: Whether to replace existing job with same ID
            **trigger_args: Arguments for the trigger (e.g., minutes=30)
        """
        try:
            self.scheduler.add_job(
                func,
                trigger_type,
                id=id,
                replace_existing=replace_existing,
                **trigger_args
            )
            logger.info(f"Job '{id}' added/updated: {trigger_type} {trigger_args}")
        except Exception as e:
            logger.error(f"Failed to add job '{id}': {e}")

# Global instance
scheduler_service = SchedulerService()
