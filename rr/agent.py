# quick_fix_agent.py - Drop this in as agent.py for immediate improvement

import json
from typing import Any, Dict, Optional
from gaialite import GAIAClient
from langchain.core.runnables.config import RunnableConfig
from langfuse.callback import CallbackHandler
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from typing_extensions import override

from core.agents.agent import Agent
from core.agents.agent_schema import InputSchema, OutputSchema
from core.agents.response_rewriter.states import RewriteState, QueryDifficulty, QueryType
from core.agents.response_rewriter.prompts import PROMPT_MAP

# Pre-computed personalities for speed
PERSONALITY_MAP = {
    "PRIMARY_SCHOOL": "[happy] Wow! Let me help you with that! *excited*",
    "MIDDLE_SCHOOL": "[friendly] Hey there! Let me explain this...",
    "HIGH_SCHOOL": "Let me break this down for you clearly.",
    "UNIVERSITY": "Here's a comprehensive analysis:",
    "PHD": "From a theoretical perspective, considering the implications..."
}

class RewriterAgent(Agent):
    def __init__(self, name: str, provider: str, model: str):
        super().__init__(name)
        self.llm_client = GAIAClient().init(provider=provider, api_transport="rest")
        self.model = model
        self.config: RunnableConfig = {"callbacks": [CallbackHandler()]}
        
        # Use faster model if available
        if "gpt-4" in model and provider == "openai":
            self.fast_model = "gpt-3.5-turbo"
            print(f"Using fast model {self.fast_model} for classification")
        else:
            self.fast_model = model
    
    def combined_node(self, state: RewriteState) -> RewriteState:
        """Single node that does everything in one LLM call."""
        
        # Minimal prompt for speed
        prompt = f"""You are Bixby, a friendly Korean TV mascot. 
Analyze this query and rewrite the response appropriately.

User: {state.user_text}
Original: {state.bixby_text}

Respond with JSON only:
{{
  "type": "GENERAL_KNOWLEDGE|PHILOSOPHY|STEM_QUERY|CHIT_CHAT",
  "difficulty": "PRIMARY_SCHOOL|MIDDLE_SCHOOL|HIGH_SCHOOL|UNIVERSITY|PHD",
  "rewritten": "your friendly rewritten response with [emotions] and pauses..."
}}"""
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Analyze and rewrite."}
        ]
        
        # Use faster settings
        res = self.llm_client.generate(
            model=self.fast_model,  # Use faster model
            messages=messages,
            response_type="json",
            temperature=0.3,  # Lower temperature for consistency
            max_tokens=200,   # Limit response length
            timeout=5         # Strict timeout
        )
        
        try:
            result = json.loads(res["choices"][0]["message"]["content"])
            state.query_type = QueryType(result.get("type", "GENERAL_KNOWLEDGE"))
            state.query_difficulty = QueryDifficulty(result.get("difficulty", "MIDDLE_SCHOOL"))
            state.rewritten_text = result.get("rewritten", state.bixby_text)
            
            # Add personality prefix if missing
            difficulty = result.get("difficulty", "MIDDLE_SCHOOL")
            prefix = PERSONALITY_MAP.get(difficulty, "")
            if prefix and not state.rewritten_text.startswith("["):
                state.rewritten_text = f"{prefix} {state.rewritten_text}"
                
        except Exception as e:
            # Fallback - just enhance the original
            state.query_type = QueryType.GENERAL_KNOWLEDGE
            state.query_difficulty = QueryDifficulty.MIDDLE_SCHOOL
            state.rewritten_text = f"[friendly] {state.bixby_text} Is there anything else I can help with?"
        
        return state
    
    # Keep original methods for compatibility but use fast implementation
    def classify_node(self, state: RewriteState) -> RewriteState:
        return self.combined_node(state)
    
    def personality_node(self, state: RewriteState) -> RewriteState:
        # Skip - already done in combined node
        state.personality = PERSONALITY_MAP.get(str(state.query_difficulty), "friendly")
        return state
    
    def rewrite_node(self, state: RewriteState) -> RewriteState:
        # Skip - already done in combined node
        return state
    
    def detect_special_case(self, user_text: str, bixby_text: str) -> Optional[str]:
        """Quick special case detection."""
        text_lower = (user_text + bixby_text).lower()
        if any(word in text_lower for word in ["error", "failed", "cannot"]):
            return "error"
        return None
    
    @override
    def build_graph(self) -> CompiledStateGraph:
        """Build optimized graph with single node."""
        graph_builder = StateGraph(RewriteState)
        
        # Single node does everything
        graph_builder.add_node("process", self.combined_node)
        graph_builder.add_edge("process", END)
        graph_builder.set_entry_point("process")
        
        graph = graph_builder.compile()
        return graph
    
    @override
    def act(self, user_id: str, thread_id: str, messages: InputSchema) -> OutputSchema:
        """Process messages with optimized flow."""
        user_text = ""
        bixby_text = ""
        
        for msg in messages.messages:
            if msg.role == "user":
                user_text = msg.content
            elif msg.role == "assistant":
                bixby_text = msg.content
        
        state = RewriteState(
            user_text=user_text,
            bixby_text=bixby_text
        )
        
        result = self.invoke_graph(state, self.config)
        
        return OutputSchema(
            output=result.rewritten_text if hasattr(result, 'rewritten_text') else bixby_text,
            metadata={
                "query_type": str(result.query_type) if hasattr(result, 'query_type') else "unknown",
                "difficulty": str(result.query_difficulty) if hasattr(result, 'query_difficulty') else "unknown",
                "optimized": True
            }
        )
