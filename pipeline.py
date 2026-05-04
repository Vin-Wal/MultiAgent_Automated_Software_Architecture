from dataclasses import dataclass, field

from agents import (
    RequirementsAgent,
    ArchitectureAgent,
    DataModelerAgent,
    DiagramAgent,
    CriticAgent,
)
from agents.critic_agent import extract_action_items

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
            print("[Pipeline] Using requirements override.")
        else:
            print("[Pipeline] Step 1/5 — RequirementsAgent ...")
            try:
                result.requirements = self.requirements_agent.run(user_input, use_rag=use_rag)
            except Exception as e:
                result.errors["requirements"] = str(e)
                return result

        print("[Pipeline] Step 2/5 — ArchitectureAgent (round 1) ...")
        try:
            result.architecture = self.architecture_agent.run(
                result.requirements, use_rag=use_rag
            )
        except Exception as e:
            result.errors["architecture"] = str(e)
            return result

        print("[Pipeline] Step 3/5 — DataModelerAgent (round 1) ...")
        try:
            result.data_model = self.data_modeler_agent.run(
                result.requirements, result.architecture, use_rag=use_rag
            )
        except Exception as e:
            result.errors["data_model"] = str(e)
            return result

        print("[Pipeline] Step 4/5 — CriticAgent (round 1) ...")
        try:
            result.critique, result.score = self.critic_agent.run(
                result.requirements, result.architecture, result.data_model, use_rag=use_rag
            )
        except Exception as e:
            result.errors["critic"] = str(e)
            return result

        result.rounds = 1
        print(f"[Pipeline] Critic score: {result.score}/10")

        if result.score < PASS_THRESHOLD:
            print(f"[Pipeline] Score {result.score} < {PASS_THRESHOLD} — running round 2 ...")
            action_items = extract_action_items(result.critique)
            print(f"[Pipeline] Injecting {len(action_items.splitlines())} action items into round 2.")

            try:
                result.architecture = self.architecture_agent.run(
                    result.requirements, use_rag=use_rag, prior_critique=action_items,
                )
            except Exception as e:
                result.errors["architecture_r2"] = str(e)

            try:
                result.data_model = self.data_modeler_agent.run(
                    result.requirements, result.architecture,
                    use_rag=use_rag, prior_critique=action_items,
                )
            except Exception as e:
                result.errors["data_model_r2"] = str(e)

            try:
                critique2, score2 = self.critic_agent.run(
                    result.requirements, result.architecture, result.data_model,
                    use_rag=use_rag, prior_critique=action_items,
                )
                result.critique = critique2
                result.score    = score2
            except Exception as e:
                result.errors["critic_r2"] = str(e)

            result.rounds = 2
            print(f"[Pipeline] Critic score after round 2: {result.score}/10")
        else:
            print(f"[Pipeline] Score {result.score} >= {PASS_THRESHOLD} — skipping round 2.")

        print("[Pipeline] Step 5/5 — DiagramAgent ...")
        try:
            result.diagrams = self.diagram_agent.run(result.architecture, result.data_model)
        except Exception as e:
            result.errors["diagrams"] = str(e)

        print(f"[Pipeline] Done. Rounds: {result.rounds}, final score: {result.score}/10")
        return result
