import unittest
import pathlib
import importlib.util
import sys
import types


def _noop_logger():
    class _Logger:
        def debug(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

    return _Logger()


def _register_stub_modules() -> dict[str, object]:
    originals: dict[str, object] = {}

    def _set_module(name: str, module: object) -> None:
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    # config
    config_mod = types.ModuleType("config")

    class _Cfg:
        orchestrator = types.SimpleNamespace(
            model="dummy-model",
            system_prompt="",
            temperature=0.0,
            max_tokens=64,
        )

    config_mod.get_config = lambda: _Cfg()
    _set_module("config", config_mod)

    # utils logger/prompt/gemini
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    _set_module("utils", utils_pkg)

    logger_mod = types.ModuleType("utils.logger")
    logger_mod.get_logger = lambda *_args, **_kwargs: _noop_logger()
    _set_module("utils.logger", logger_mod)

    prompt_mod = types.ModuleType("utils.prompt_templates")
    prompt_mod.ORCHESTRATOR_PROMPTS = {"decision_format": "{}"}
    _set_module("utils.prompt_templates", prompt_mod)

    gemini_mod = types.ModuleType("utils.gemini_client")

    class GeminiClientWrapper:
        def __init__(self, *args, **kwargs):
            pass

    gemini_mod.GeminiClientWrapper = GeminiClientWrapper
    _set_module("utils.gemini_client", gemini_mod)

    # agents package + tool stubs used by imports in api_orchestrator.py
    agents_pkg = types.ModuleType("agents")
    agents_pkg.__path__ = []
    _set_module("agents", agents_pkg)

    tools_pkg = types.ModuleType("agents.tools")
    tools_pkg.__path__ = []
    _set_module("agents.tools", tools_pkg)

    memory_tool_mod = types.ModuleType("agents.tools.memory_tool")

    class MemoryTool:
        pass

    memory_tool_mod.MemoryTool = MemoryTool
    _set_module("agents.tools.memory_tool", memory_tool_mod)

    doc_tool_mod = types.ModuleType("agents.tools.document_retrieval_tool")

    class DocumentRetrievalTool:
        pass

    doc_tool_mod.DocumentRetrievalTool = DocumentRetrievalTool
    _set_module("agents.tools.document_retrieval_tool", doc_tool_mod)

    web_tool_mod = types.ModuleType("agents.tools.web_search_tool")

    class WebSearchTool:
        pass

    web_tool_mod.WebSearchTool = WebSearchTool
    _set_module("agents.tools.web_search_tool", web_tool_mod)

    motion_tool_mod = types.ModuleType("agents.tools.motion_candidate_retriever")

    class MotionCandidateRetriever:
        def retrieve_top_k(self, *_args, **_kwargs):
            return []

    motion_tool_mod.MotionCandidateRetriever = MotionCandidateRetriever
    motion_tool_mod.MotionCandidate = type("MotionCandidate", (), {})
    _set_module("agents.tools.motion_candidate_retriever", motion_tool_mod)

    query_transform_mod = types.ModuleType("agents.query_transform")

    class QueryTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def transform_query(self, query):
            return {"expanded_query": query, "hyde_document": query}

    query_transform_mod.QueryTransformer = QueryTransformer
    _set_module("agents.query_transform", query_transform_mod)

    # ── Real agents.intents (no external deps; we want canonical enums) ────
    intents_path = pathlib.Path(__file__).resolve().parents[2] / "agents" / "intents.py"
    intents_spec = importlib.util.spec_from_file_location("agents.intents", intents_path)
    intents_mod = importlib.util.module_from_spec(intents_spec)
    assert intents_spec and intents_spec.loader
    intents_spec.loader.exec_module(intents_mod)
    _set_module("agents.intents", intents_mod)

    # ── agents.double_rag stub (constructor/run never invoked in these tests) ──
    double_rag_mod = types.ModuleType("agents.double_rag")

    class DoubleRAGAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):  # pragma: no cover — not exercised here
            return types.SimpleNamespace(
                clinical_docs=[], constraints="", motion_candidates=[],
                refined_hyde_document=None, has_motion_match=False,
            )

    class DoubleRAGResult:  # pragma: no cover
        pass

    double_rag_mod.DoubleRAGAgent = DoubleRAGAgent
    double_rag_mod.DoubleRAGResult = DoubleRAGResult
    _set_module("agents.double_rag", double_rag_mod)

    # ── core package + tracing + resource_guard stubs ──────────────────────
    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = []
    _set_module("core", core_pkg)

    tracing_mod = types.ModuleType("core.tracing")

    class AgentTrace:  # pragma: no cover — not exercised here
        def add_decision(self, *_a, **_k):
            return None

        def increment_llm_calls(self, *_a, **_k):
            return None

        def finish(self, *_a, **_k):
            return self

    class TraceStage:  # pragma: no cover
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    tracing_mod.AgentTrace = AgentTrace
    tracing_mod.TraceStage = TraceStage
    _set_module("core.tracing", tracing_mod)

    rg_mod = types.ModuleType("core.resource_guard")

    class _DummyExecutor:  # pragma: no cover
        def submit(self, fn, *args, **kwargs):
            class _F:
                def result(self_inner):
                    return fn(*args, **kwargs)

            return _F()

    rg_mod.shared_tool_executor = lambda: _DummyExecutor()
    _set_module("core.resource_guard", rg_mod)

    return originals


def _load_api_orchestrator_module():
    originals = _register_stub_modules()
    try:
        project_root = pathlib.Path(__file__).resolve().parents[2]
        module_path = project_root / "agents" / "api_orchestrator.py"
        spec = importlib.util.spec_from_file_location("isolated_api_orchestrator", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


_api_orchestrator = _load_api_orchestrator_module()
ActionType = _api_orchestrator.ActionType
IntentType = _api_orchestrator.IntentType
OrchestratorAgent = _api_orchestrator.OrchestratorAgent
OrchestratorDecision = _api_orchestrator.OrchestratorDecision


class TestApiOrchestratorGates(unittest.TestCase):
    def setUp(self) -> None:
        # Avoid heavy constructor dependencies; these tests target pure gate logic.
        self.agent = OrchestratorAgent.__new__(OrchestratorAgent)

    def test_personal_query_enables_memory_gate(self) -> None:
        gates = self.agent._compute_intent_gates(
            intent=IntentType.KNOWLEDGE_QUERY,
            query="Based on my history of lower back pain, what should I do?",
            requested_actions={"use_web_search": True},
        )
        self.assertTrue(gates["use_memory"])
        self.assertTrue(gates["use_documents"])
        self.assertTrue(gates["use_web_search"])
        self.assertFalse(gates["generate_motion"])

    def test_non_personal_query_disables_memory_gate(self) -> None:
        gates = self.agent._compute_intent_gates(
            intent=IntentType.KNOWLEDGE_QUERY,
            query="What is the difference between static and dynamic stretching?",
            requested_actions={"use_web_search": True},
        )
        self.assertFalse(gates["use_memory"])
        self.assertTrue(gates["use_documents"])

    def test_motion_generation_only_for_visualize_intent(self) -> None:
        non_visual_intents = [
            IntentType.CONVERSATION,
            IntentType.KNOWLEDGE_QUERY,
            IntentType.EXERCISE_RECOMMENDATION,
        ]
        for intent in non_visual_intents:
            gates = self.agent._compute_intent_gates(
                intent=intent,
                query="Show me a plan for neck pain",
                requested_actions={"use_web_search": True},
            )
            self.assertFalse(gates["generate_motion"])

        visualize_gates = self.agent._compute_intent_gates(
            intent=IntentType.VISUALIZE_MOTION,
            query="Show me how to do a squat",
            requested_actions={"use_web_search": True},
        )
        self.assertTrue(visualize_gates["generate_motion"])
        self.assertFalse(visualize_gates["use_documents"])

    def test_select_tools_honors_memory_gate(self) -> None:
        self.agent._memory_tool = object()
        self.agent._document_tool = object()
        self.agent._web_search_tool = object()

        decision_no_memory = OrchestratorDecision(
            action=ActionType.CALL_LLM,
            confidence=0.9,
            reasoning="test",
            parameters={
                "use_memory": False,
                "use_documents": True,
                "use_web_search": False,
            },
        )
        selected_no_memory = self.agent._select_tools(
            decision=decision_no_memory,
            intent=IntentType.CONVERSATION,
        )
        self.assertEqual(selected_no_memory, [])

        decision_with_docs = OrchestratorDecision(
            action=ActionType.CALL_LLM,
            confidence=0.9,
            reasoning="test",
            parameters={
                "use_memory": True,
                "use_documents": True,
                "use_web_search": False,
            },
        )
        selected_with_docs = self.agent._select_tools(
            decision=decision_with_docs,
            intent=IntentType.KNOWLEDGE_QUERY,
        )
        self.assertEqual(set(selected_with_docs), {"memory", "documents"})


if __name__ == "__main__":
    unittest.main()
