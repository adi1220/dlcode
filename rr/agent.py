import json
from typing import Any
from typing import Dict
from typing import Optional

from gaialite import GAIAClient
from langchain.core.runnables.config import RunnableConfig
from langfuse.callback import CallbackHandler
from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from typing_extensions import override

from core.agents.agent import Agent
from core.agents.agent_schema import InputSchema
from core.agents.agent_schema import OutputSchema
from core.agents.response_rewriter.prompts import CLASSIFIER_PROMPT
from core.agents.response_rewriter.prompts import CLASSIFIER_RESPONSE_SCHEMA
from core.agents.response_rewriter.prompts import PROMPT_MAP
from core.agents.response_rewriter.prompts import REWRITER_RESPONSE_SCHEMA
from core.agents.response_rewriter.prompts import get_prompt_for_situation
from core.agents.response_rewriter.prompts import HYBRID_PROMPT
from core.agents.response_rewriter.states import QueryDifficulty
from core.agents.response_rewriter.states import QueryType
from core.agents.response_rewriter.states import RewriteState


# Keep the default for backward compatibility
REWRITER_PROMPT = PROMPT_MAP["default"]


class RewriterAgent(Agent):
    def __init__(self, name: str, provider: str, model: str):
        super().__init__(name)
        self.llm_client = GAIAClient().init(provider=provider, api_transport="rest")
        self.model = model
        self.config: RunnableConfig = {"callbacks": [CallbackHandler()]}

    def classify_node(self, state: RewriteState) -> RewriteState:
        messages = [
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": [{"text": state.user_text, "type": "text"}]},
        ]
        res = self.llm_client.generate(
            model=self.model,
            messages=messages,
            response_type="json_schema",
            response_schema=CLASSIFIER_RESPONSE_SCHEMA,
        )["choices"][0]["message"]["content"][0]["text"]
        classification = json.loads(res)
        state.query_type = QueryType(classification["type"])
        state.query_difficulty = QueryDifficulty(classification["difficulty"])
        return state

    def personality_node(self, state: RewriteState) -> RewriteState:
        master_personality = """
        Name: Bixby
        Occupation: Mascot
        Background: Born in South Korea. Cute character that loves talking to people.
        Functionality: Control TV
        Personality: Child friendly, helpful
        Target Audience: Ages 5-80
        Knowledge Level: UNIVERSITY
        """

        difficulty_map = {"PRIMARY_SCHOOL": 1, "MIDDLE_SCHOOL": 2, "HIGH_SCHOOL": 3, "UNIVERSITY": 4, "PHD": 5}

        agent_diff = 2  # UNIVERSITY level
        query_diff = difficulty_map.get(state.query_difficulty, agent_diff)

        if query_diff < agent_diff:
            reaction = """
            slightly annoyed at being asked simple questions and uses words like Aghh...
            when it is annoyed or frustrated
            """
        elif query_diff > agent_diff:
            reaction = """
            thoughtful and somewhat doubtful about complex questions and uses pauses and words like Hmm...
            Uhhh... when it is thoughtful
            """
        else:
            reaction = "comfortable and confident"

        messages = [
            {
                "role": "system",
                "content": f"""
                Create a personality profile based on:
                - Master Personality: {master_personality}
                - Query Difficulty: {state.query_difficulty}
                - Agent Knowledge Level: UNIVERSITY
                - Emotional Reaction: {reaction}
                
                Output a personality description that:
                1. Maintains core Bixby traits
                2. Adjusts for difficulty mismatch
                3. Includes appropriate emotional tone
                4. Keeps Korean cultural references
                """,
            },
            {"role": "user", "content": state.user_text},
        ]

        res = self.llm_client.generate(model=self.model, messages=messages, response_type="text")["choices"][0][
            "message"
        ]["content"]

        # Ensure personality is stored as a string
        if isinstance(res, list):
            state.personality = json.dumps(res)
        else:
            state.personality = str(res)
        return state

    def rewrite_node(self, state: RewriteState) -> RewriteState:
        # Check for prompt override in config
        config = getattr(self, '_current_config', self.config)
        
        # Determine which prompt to use
        if config and 'override_prompt' in config:
            # Use the provided prompt override
            prompt_template = config['override_prompt']
        elif config and 'use_smart_selection' in config:
            # Use smart prompt selection
            special_case = self.detect_special_case(state.user_text, state.bixby_text)
            prompt_template = get_prompt_for_situation(
                query_type=state.query_type,
                difficulty=state.query_difficulty,
                special_case=special_case
            )
        elif config and 'prompt_strategy' in config:
            # Use specific strategy
            strategy = config['prompt_strategy']
            if strategy == 'hybrid':
                prompt_template = HYBRID_PROMPT
            else:
                prompt_template = PROMPT_MAP.get(strategy, REWRITER_PROMPT)
        else:
            # Default behavior - use the default prompt
            prompt_template = REWRITER_PROMPT
        
        messages = [
            {
                "role": "system",
                "content": f"""
                {prompt_template}
                
                Current Personality:
                {state.personality}
                """,
            },
            {
                "role": "user",
                "content": [
                    {"text": f"User: {state.user_text}", "type": "text"},
                    {"text": f"Bixby: {state.bixby_text}", "type": "text"},
                ],
            },
        ]
        res = self.llm_client.generate(
            model=self.model,
            messages=messages,
            response_type="json_schema",
            response_schema=REWRITER_RESPONSE_SCHEMA,
        )["choices"][0]["message"]["content"][0]["text"]
        state.rewritten_text = json.loads(res)["Rewritten"]
        return state

    def detect_special_case(self, user_text: str, bixby_text: str) -> Optional[str]:
        """Detect if this needs special handling."""
        # Error patterns
        error_keywords = ["error", "not found", "cannot", "failed", "unable", "no device"]
        if any(keyword in bixby_text.lower() for keyword in error_keywords):
            return "error"
        
        # Sensitive topics
        sensitive_keywords = ["health", "medical", "personal", "private", "death", "suicide"]
        if any(keyword in user_text.lower() for keyword in sensitive_keywords):
            return "sensitive"
        
        # Ambiguous queries
        if len(user_text.split()) < 3 and "?" not in user_text:
            return "ambiguous"
        
        return None

    @override
    def build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(RewriteState)
        graph_builder.add_node("classify_node", self.classify_node)
        graph_builder.add_node("personality_node", self.personality_node)
        graph_builder.add_node("rewrite_node", self.rewrite_node)
        graph_builder.add_edge("classify_node", "personality_node")
        graph_builder.add_edge("personality_node", "rewrite_node")
        graph_builder.add_edge("rewrite_node", END)
        graph_builder.set_entry_point("classify_node")
        graph = graph_builder.compile()
        return graph

    @override
    def invoke_graph(self, state: BaseModel, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        # Store the config for use in nodes
        self._current_config = config if config is not None else self.config
        
        # Invoke the graph with the provided config
        response = self.graph.invoke(state, config=self._current_config)
        
        # Clean up
        self._current_config = self.config
        return response

    @override
    def act(self, user_id: str, thread_id: str, messages: InputSchema) -> OutputSchema:
        # This method can be updated to support prompt strategy selection
        # For now, it maintains backward compatibility
        pass
