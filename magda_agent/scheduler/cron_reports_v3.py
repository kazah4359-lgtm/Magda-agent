import logging
import time
from typing import Optional, List, Dict
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

class DailyReportManagerV3:
    """
    Manages scheduled daily reports generation using HermesCronSchedulerV3.
    Aggregates memory usage and procedural learnings into a summary document.
    Inspired by Hermes Agent trend.
    """
    def __init__(
        self,
        scheduler: Optional[HermesCronSchedulerV3] = None,
        memory: Optional[EpisodicMemory] = None,
        procedural: Optional[ProceduralMemory] = None,
        llm: Optional[LLMClient] = None
    ):
        self.scheduler = scheduler or HermesCronSchedulerV3()
        self.memory = memory
        self.procedural = procedural
        self.llm = llm

    def register_daily_report(self, name: str, cron_expr: str = "0 8 * * *") -> None:
        """
        Registers the report generation as a daily task.
        """
        async def _generate() -> str:
            return await self.generate_aggregated_report(name)

        _generate.__name__ = f"report_{name}"
        self.scheduler.schedule(cron_expr, _generate, name=name)
        logger.info(f"Registered daily report '{name}' with schedule '{cron_expr}'")

    async def generate_aggregated_report(self, name: str) -> str:
        """
        Aggregates data and uses LLM to generate a markdown report.
        """
        if not self.memory or not self.procedural or not self.llm:
            error_msg = "Error: EpisodicMemory, ProceduralMemory, or LLMClient not configured."
            logger.error(error_msg)
            return error_msg

        try:
            # Stats
            episodic_count = len(self.memory.get_all_events(limit=10000))
            procedural_data = self.procedural.collection.get()
            procedural_count = len(procedural_data['ids']) if procedural_data and 'ids' in procedural_data else 0

            # Recent events
            events = self.memory.get_all_events(limit=50)
            event_texts = [f"- {evt['text']}" for evt in events]
            context = "\n".join(event_texts)

            prompt = (
                f"Generate a daily summary report titled '{name}'.\n"
                f"Memory Usage Stats:\n"
                f"- Episodic Events: {episodic_count}\n"
                f"- Procedural Learnings: {procedural_count}\n\n"
                f"Recent Events:\n{context}\n\n"
                "Please provide a concise markdown report summarizing findings and progress."
            )

            report_content = await self.llm.generate(prompt)

            timestamp = int(time.time())
            filename = f"daily_report_{timestamp}.md"

            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(f"Generated daily report saved to {filename}")
            return report_content
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
        await self.scheduler.start()

    async def stop(self) -> None:
        await self.scheduler.stop()
