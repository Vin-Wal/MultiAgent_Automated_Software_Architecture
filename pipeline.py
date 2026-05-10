import logging
from dataclasses import dataclass, field

from agents import (
    RequirementsAgent,
    ArchitectureAgent,
    DataModelerAgent,
    DiagramAgent,
    CriticAgent,
)
from agents.critic_agent import extract_action_items

log = logging.getLogger(__name__)

PASS_THRESHOLD = 7


@dataclass
class PipelineResult:
    requirements: str = ""
    architecture: str = ""
    data_model:   str = ""
    diagrams:     str = ""
    critique:     str = ""
    score:        int = 0
    rounds:       int = 0
    errors: dict[str, str] = field(default_factory=dict)


class Pipeline:

    def __init__(self, force_reindex: bool = False):
        self.requirements_agent = RequirementsAgent(force_reindex=force_reindex)
        self.architecture_agent = ArchitectureAgent(force_reindex=force_reindex)
        self.data_modeler_agent = DataModelerAgent(force_reindex=force_reindex)
        self.diagram_agent      = DiagramAgent()
        self.critic_agent       = CriticAgent(force_reindex=force_reindex)

    def run(
        self,
        user_input: str,
        use_rag: bool = True,
        requirements_override: str | None = None,
    ) -> PipelineResult:
        result = PipelineResult()

        if requirements_override is not None:
            result.requirements = requirements_override
            log.info("Using requirements override.")
        else:
            log.info("Step 1/5 — RequirementsAgent")
            try:
                result.requirements = self.requirements_agent.run(user_input, use_rag=use_rag)
            except Exception as e:
                log.error("RequirementsAgent failed: %s", e)
                result.errors["requirements"] = str(e)
                return result

        log.info("Step 2/5 — ArchitectureAgent (round 1)")
        try:
            result.architecture = self.architecture_agent.run(
                result.requirements, use_rag=use_rag
            )
        except Exception as e:
            log.error("ArchitectureAgent failed: %s", e)
            result.errors["architecture"] = str(e)
            return result

        log.info("Step 3/5 — DataModelerAgent (round 1)")
        try:
            result.data_model = self.data_modeler_agent.run(
                result.requirements, result.architecture, use_rag=use_rag
            )
        except Exception as e:
            log.error("DataModelerAgent failed: %s", e)
            result.errors["data_model"] = str(e)
            return result

        log.info("Step 4/5 — CriticAgent (round 1)")
        try:
            result.critique, result.score = self.critic_agent.run(
                result.requirements, result.architecture, result.data_model, use_rag=use_rag
            )
        except Exception as e:
            log.error("CriticAgent failed: %s", e)
            result.errors["critic"] = str(e)
            return result

        result.rounds = 1
        log.info("Critic score: %d/10", result.score)

        if result.score < PASS_THRESHOLD:
            log.info("Score %d < %d — running round 2", result.score, PASS_THRESHOLD)
            action_items = extract_action_items(result.critique)
            log.info("Injecting %d action items into round 2", len(action_items.splitlines()))

            try:
                result.architecture = self.architecture_agent.run(
                    result.requirements, use_rag=use_rag, prior_critique=action_items,
                )
            except Exception as e:
                log.error("ArchitectureAgent round 2 failed: %s", e)
                result.errors["architecture_r2"] = str(e)

            try:
                result.data_model = self.data_modeler_agent.run(
                    result.requirements, result.architecture,
                    use_rag=use_rag, prior_critique=action_items,
                )
            except Exception as e:
                log.error("DataModelerAgent round 2 failed: %s", e)
                result.errors["data_model_r2"] = str(e)

            try:
                critique2, score2 = self.critic_agent.run(
                    result.requirements, result.architecture, result.data_model,
                    use_rag=use_rag, prior_critique=action_items,
                )
                result.critique = critique2
                result.score    = score2
            except Exception as e:
                log.error("CriticAgent round 2 failed: %s", e)
                result.errors["critic_r2"] = str(e)

            result.rounds = 2
            log.info("Critic score after round 2: %d/10", result.score)
        else:
            log.info("Score %d >= %d — skipping round 2", result.score, PASS_THRESHOLD)

        log.info("Step 5/5 — DiagramAgent")
        try:
            result.diagrams = self.diagram_agent.run(result.architecture, result.data_model)
        except Exception as e:
            log.error("DiagramAgent failed: %s", e)
            result.errors["diagrams"] = str(e)

        log.info("Done. Rounds: %d, final score: %d/10", result.rounds, result.score)
        return result
