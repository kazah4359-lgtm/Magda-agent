import logging
from typing import Optional, Any
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

class DailyReportManagerV3:
    """
    Manages scheduled daily reports generation using HermesCronSchedulerV3.
    It aggregates daily events from EpisodicMemory and generates a report via LLM.
    Inspired by Hermes Agent trend.
    """
    def __init__(
        self,
        scheduler: Optional[HermesCronSchedulerV3] = None,
        memory: Optional[EpisodicMemory] = None,
        llm: Optional[LLMClient] = None
    ):
        """
        Initializes the DailyReportManagerV3.
        """
        self.scheduler = scheduler or HermesCronSchedulerV3()
        self.memory = memory
        self.llm = llm

    def register_daily_report(self, name: str, cron_expr: str = "0 8 * * *") -> None:
        """
        Registers the report generation as a daily task.

        Args:
            name: The name of the report task.
            cron_expr: The cron expression indicating when to generate the report. Defaults to "0 8 * * *".
        """
        async def _generate() -> str:
            return await self.generate_aggregated_report(name)

        # Required to fix name resolution inside lambda/closure within task list if multiple registered
        _generate.__name__ = f"report_{name}"

        self.scheduler.schedule(cron_expr, _generate, name=name)
        logger.info(f"Registered daily report '{name}' with schedule '{cron_expr}'")

    async def generate_aggregated_report(self, name: str) -> str:
        """
        Aggregates recent events from EpisodicMemory and uses LLM to generate a report.
        """
        if not self.memory or not self.llm:
            error_msg = "Error: Memory or LLM not configured."
            logger.error(error_msg)
            return error_msg

        try:
            # We get up to 100 recent events
            events = self.memory.get_all_events(limit=100)

            if not events:
                msg = f"Report {name}: No recent events to report."
                logger.info(msg)
                return msg

            event_texts = [evt["text"] for evt in events]
            context = "\n".join(event_texts)
            prompt = f"Please generate a daily report named '{name}' summarizing the following events:\n{context}"

            report = await self.llm.generate(prompt)
            logger.info(f"Generated daily report '{name}': {report}")
            return report
        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
            return f"Failed to generate report: {e}"

    async def generate_report_now(self, name: str) -> Optional[str]:
        """
        Immediately runs a report task that has been registered.
        """
        func = self.scheduler._func_registry.get(name)
        if not func:
            logger.error(f"Cannot generate report '{name}', it is not registered.")
            return None

        logger.info(f"Generating daily report '{name}' immediately.")
        try:
            result = await func()
            return result
        except Exception as e:
            logger.error(f"Error generating daily report '{name}': {e}", exc_info=True)
            return None

    async def start(self) -> None:
        """
        Starts the underlying scheduler loop.
        """
        logger.info("Starting DailyReportManagerV3 scheduler")
        await self.scheduler.start()

    async def stop(self) -> None:
        """
        Stops the underlying scheduler loop.
        """
        logger.info("Stopping DailyReportManagerV3 scheduler")
        await self.scheduler.stop()
