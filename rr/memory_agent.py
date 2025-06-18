import json
import re
from typing import Any, Dict, Optional, List
from pydantic import BaseModel

# Import base agent components (following the pattern from agent.py)
try:
    from gaialite import GAIAClient
    from langchain.core.runnables.config import RunnableConfig
    from langfuse.callback import CallbackHandler
    from langgraph.graph import END, StateGraph
    from langgraph.graph.state import CompiledStateGraph
    from typing_extensions import override
    
    from core.agents.agent import Agent
    from core.agents.agent_schema import InputSchema, OutputSchema
    HAS_AGENT_FRAMEWORK = True
except ImportError:
    # Fallback for standalone usage
    HAS_AGENT_FRAMEWORK = False
    
    class Agent:
        def __init__(self, name: str):
            self.name = name
            self.graph = None
            
        def build_graph(self):
            pass
            
        def invoke_graph(self, state, config=None):
            return state
    
    class BaseModel:
        pass
    
    class RunnableConfig(dict):
        pass

from memory_tool import MemoryTool


# Define states for memory operations
class MemoryState(BaseModel):
    """State for memory agent processing."""
    user_input: str
    operation_type: Optional[str] = None  # 'add', 'fetch', 'delete', 'assess'
    operation_params: Optional[Dict[str, Any]] = None
    importance_assessment: Optional[str] = None
    result: Optional[str] = None


class MemoryAgent(Agent if HAS_AGENT_FRAMEWORK else object):
    """Agent that manages user memories intelligently."""
    
    def __init__(self, name: str = "memory_agent", memory_tool_instance: Optional[MemoryTool] = None,
                 provider: str = "openai", model: str = "gpt-3.5-turbo"):
        """
        Initialize the MemoryAgent.
        
        Args:
            name: Name of the agent
            memory_tool_instance: Instance of MemoryTool
            provider: LLM provider (for GAIAClient)
            model: Model to use for decisions
        """
        if HAS_AGENT_FRAMEWORK:
            super().__init__(name)
            self.llm_client = GAIAClient().init(provider=provider, api_transport="rest")
            self.model = model
            self.config = RunnableConfig({"callbacks": [CallbackHandler()]})
        else:
            self.name = name
            self.llm_client = None
            self.model = model
            self.config = {}
        
        # Initialize memory tool
        self.memory_tool = memory_tool_instance or MemoryTool()
    
    def detect_command(self, user_input: str) -> Dict[str, Any]:
        """
        Detect if the user input contains an explicit memory command.
        
        Returns:
            Dict with 'type' and 'params' if command found, None otherwise
        """
        user_input_lower = user_input.lower()
        
        # ADD commands
        add_patterns = [
            r"remember that (.+)",
            r"remember: (.+)",
            r"add this memory: (.+)",
            r"store: (.+)",
            r"note down (.+)",
            r"note that (.+)"
        ]
        
        for pattern in add_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                content = user_input[match.start(1):match.end(1)]  # Get original case
                return {'type': 'add', 'params': {'content': content}}
        
        # FETCH commands
        if any(phrase in user_input_lower for phrase in [
            "what do you remember about",
            "tell me about",
            "fetch memories about",
            "search memories for",
            "find memories about"
        ]):
            # Extract search query
            query_match = re.search(r"about (.+?)(?:\?|$)", user_input_lower)
            if query_match:
                query = query_match.group(1).strip()
                return {'type': 'fetch', 'params': {'query': query}}
        
        if any(phrase in user_input_lower for phrase in [
            "show me recent memories",
            "show recent memories",
            "what are my recent memories",
            "list recent memories"
        ]):
            return {'type': 'fetch', 'params': {'query': 'recent'}}
        
        # DELETE commands
        delete_match = re.search(r"(?:forget|delete) memory (?:id )?([a-zA-Z0-9_]+)", user_input_lower)
        if delete_match:
            memory_id = delete_match.group(1)
            return {'type': 'delete', 'params': {'memory_id': memory_id}}
        
        return None
    
    def assess_importance(self, user_input: str) -> tuple[bool, str]:
        """
        Assess if the user input is important enough to store as memory.
        
        Returns:
            Tuple of (is_important, reason)
        """
        if self.llm_client:
            # Use LLM for assessment
            prompt = f"""Given the user input: "{user_input}"

Should this information be stored as a long-term memory? Consider if it's:
- A fact or piece of information
- A personal preference or detail
- An important date or deadline
- Something that might need recall in future conversations
- Information that seems stable and reusable

Respond with exactly one of these formats:
YES: [brief reason why this should be stored]
NO: [brief reason why this shouldn't be stored]"""

            try:
                res = self.llm_client.generate(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a memory assessment system. Be concise."},
                        {"role": "user", "content": prompt}
                    ],
                    response_type="text",
                    temperature=0.3,
                    max_tokens=100
                )
                
                response = res["choices"][0]["message"]["content"]
                if isinstance(response, list):
                    response = response[0]
                
                is_important = response.strip().startswith("YES:")
                reason = response.strip()[4:].strip() if ":" in response else ""
                
                return is_important, reason
                
            except Exception as e:
                # Fallback to rule-based
                print(f"LLM assessment failed: {e}, using rule-based fallback")
        
        # Rule-based fallback
        return self._rule_based_importance(user_input)
    
    def _rule_based_importance(self, user_input: str) -> tuple[bool, str]:
        """Rule-based importance assessment as fallback."""
        user_input_lower = user_input.lower()
        
        # Criteria for YES
        important_keywords = [
            'always', 'important', 'note', 'favorite', 'prefer',
            'birthday', 'anniversary', 'deadline', 'allerg',
            'my name is', "i'm", "i am", 'my job', 'i work',
            'i like', 'i love', 'i hate', 'remember'
        ]
        
        # Check for facts (contains "is", "are", "was", "were")
        fact_patterns = [' is ', ' are ', ' was ', ' were ', ' = ']
        
        # Criteria for NO
        casual_phrases = [
            'how are you', 'hello', 'hi', 'hey', 'thanks',
            'thank you', 'okay', 'ok', 'sure', 'yes', 'no',
            'what is', 'what are', 'when is', 'where is',
            'can you', 'could you', 'would you', '?'
        ]
        
        # Check if it's a question (ends with ?)
        if user_input.strip().endswith('?'):
            return False, "Questions are typically not stored"
        
        # Check for important keywords
        for keyword in important_keywords:
            if keyword in user_input_lower:
                return True, f"Contains important keyword: '{keyword}'"
        
        # Check for facts
        for pattern in fact_patterns:
            if pattern in user_input_lower and len(user_input.split()) > 3:
                return True, "Appears to be a factual statement"
        
        # Check for casual conversation
        for phrase in casual_phrases:
            if phrase in user_input_lower and len(user_input) < 20:
                return False, "Casual conversation"
        
        # Length-based heuristic
        if len(user_input.split()) > 5:
            return True, "Substantial information provided"
        
        return False, "Brief input without clear importance"
    
    def process_input(self, user_input: str) -> str:
        """
        Main method to process user input and perform memory operations.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Response string
        """
        # Check for explicit commands first
        command = self.detect_command(user_input)
        
        if command:
            # Execute command
            if command['type'] == 'add':
                content = command['params']['content']
                result = self.memory_tool.add_memory(content)
                return f"✅ {result}\nContent: '{content}'"
            
            elif command['type'] == 'fetch':
                query = command['params']['query']
                memories = self.memory_tool.fetch_memories(query, limit=5)
                
                if not memories:
                    return f"No memories found for '{query}'"
                
                response = f"Found {len(memories)} memories:\n\n"
                for mem in memories:
                    response += f"🔹 [{mem['ID']}] {mem['Timestamp']}\n   {mem['Content']}\n\n"
                return response.strip()
            
            elif command['type'] == 'delete':
                memory_id = command['params']['memory_id']
                result = self.memory_tool.delete_memory(memory_id)
                return f"🗑️ {result}"
        
        # No explicit command - assess importance
        is_important, reason = self.assess_importance(user_input)
        
        if is_important:
            # Store as memory
            result = self.memory_tool.add_memory(user_input)
            return f"💡 I've noted that down for future reference.\n({reason})\n{result}"
        else:
            # Just acknowledge without storing
            responses = [
                "I understand.",
                "Got it.",
                "Okay, I see.",
                "That's interesting.",
                "Thanks for sharing that.",
                "Noted (but not stored)."
            ]
            import random
            return f"{random.choice(responses)} ({reason})"
    
    # Methods for langgraph integration (if available)
    @override
    def build_graph(self) -> Optional[CompiledStateGraph]:
        """Build the processing graph for memory operations."""
        if not HAS_AGENT_FRAMEWORK:
            return None
            
        graph_builder = StateGraph(MemoryState)
        
        # Add nodes
        graph_builder.add_node("detect_command", self.detect_command_node)
        graph_builder.add_node("assess_importance", self.assess_importance_node)
        graph_builder.add_node("execute_operation", self.execute_operation_node)
        
        # Add edges
        graph_builder.add_edge("detect_command", "execute_operation")
        graph_builder.add_edge("assess_importance", "execute_operation")
        graph_builder.add_edge("execute_operation", END)
        
        # Conditional routing
        graph_builder.add_conditional_edges(
            "detect_command",
            lambda state: "execute" if state.operation_type else "assess",
            {
                "execute": "execute_operation",
                "assess": "assess_importance"
            }
        )
        
        graph_builder.set_entry_point("detect_command")
        
        return graph_builder.compile()
    
    def detect_command_node(self, state: MemoryState) -> MemoryState:
        """Node for command detection."""
        command = self.detect_command(state.user_input)
        if command:
            state.operation_type = command['type']
            state.operation_params = command['params']
        return state
    
    def assess_importance_node(self, state: MemoryState) -> MemoryState:
        """Node for importance assessment."""
        is_important, reason = self.assess_importance(state.user_input)
        state.operation_type = 'add' if is_important else 'acknowledge'
        state.importance_assessment = reason
        if is_important:
            state.operation_params = {'content': state.user_input}
        return state
    
    def execute_operation_node(self, state: MemoryState) -> MemoryState:
        """Node for executing the determined operation."""
        # Implementation would mirror process_input logic
        state.result = self.process_input(state.user_input)
        return state


# Example usage and demonstration
if __name__ == "__main__":
    print("=== Memory Agent Demo ===\n")
    
    # Create memory tool and agent
    memory_tool = MemoryTool("sample_memory_db.tsv")
    
    # Try to use LLM if available, otherwise use rule-based
    try:
        agent = MemoryAgent(memory_tool_instance=memory_tool)
        print("Using LLM-based importance assessment\n")
    except:
        agent = MemoryAgent(memory_tool_instance=memory_tool)
        print("Using rule-based importance assessment\n")
    
    # Demo interactions
    test_inputs = [
        "Remember that my car is a blue Toyota Camry",  # Explicit ADD
        "My sister's name is Sarah and she lives in Boston",  # Important info
        "How are you doing today?",  # Casual conversation
        "What do you remember about my car?",  # Explicit FETCH
        "I'm allergic to shellfish - this is very important",  # Important info
        "Show me recent memories",  # Explicit FETCH recent
        "Delete memory ID 1735052520_i9j0k1l2",  # Explicit DELETE
        "What do you remember about allergies?",  # FETCH to confirm
    ]
    
    for user_input in test_inputs:
        print(f"👤 User: {user_input}")
        response = agent.process_input(user_input)
        print(f"🤖 Agent: {response}")
        print("-" * 50 + "\n")
