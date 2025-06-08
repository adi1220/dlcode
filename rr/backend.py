from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from config import Config
import logging
from core.agents.response_rewriter.agent import RewriterAgent
from core.agents.response_rewriter.states import RewriteState
from prompts import PROMPT_MAP, get_prompt_for_situation, HYBRID_PROMPT

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the RewriterAgent
provider = os.getenv('provider', 'openai')
model = os.getenv('model', Config.LLM_MODEL)

# Create the agent instance
rewriter_agent = RewriterAgent("rewriter_agent", provider, model)
rewriter_agent.build_graph()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "response-rewriter-backend"})

@app.route('/rewrite', methods=['POST'])
def rewrite_response():
    """Main endpoint to rewrite responses using RewriterAgent."""
    try:
        data = request.json
        
        # Validate input
        if not data or 'user_text' not in data or 'bix_response' not in data:
            return jsonify({"error": "Missing 'user_text' or 'bix_response' in request body"}), 400
        
        user_text = data['user_text']
        bix_response = data['bix_response']
        prompt_type = data.get('prompt_type', 'default')
        use_smart_selection = data.get('use_smart_selection', False)
        
        logger.info(f"Received rewrite request - Strategy: {prompt_type}, Smart: {use_smart_selection}")
        
        # Create initial state
        initial_state = RewriteState(
            user_text=user_text,
            bixby_text=bix_response
        )
        
        # Prepare config based on request
        config = rewriter_agent.config.copy()
        
        if use_smart_selection:
            # Enable smart selection in the agent
            config['use_smart_selection'] = True
        elif prompt_type == "hybrid":
            # Use hybrid prompt
            config['override_prompt'] = HYBRID_PROMPT
            config['prompt_strategy'] = "hybrid"
        elif prompt_type in PROMPT_MAP:
            # Use specific strategy from PROMPT_MAP
            config['override_prompt'] = PROMPT_MAP[prompt_type]
            config['prompt_strategy'] = prompt_type
        else:
            # Default strategy
            config['prompt_strategy'] = "default"
        
        # Invoke the agent with configured strategy
        result = rewriter_agent.invoke_graph(initial_state, config)
        
        # Extract the results
        if hasattr(result, 'rewritten_text') and result.rewritten_text:
            response_data = {
                "status": "success",
                "rewritten_response": result.rewritten_text,
                "classification": {
                    "type": str(result.query_type) if hasattr(result, 'query_type') else 'unknown',
                    "difficulty": str(result.query_difficulty) if hasattr(result, 'query_difficulty') else 'unknown'
                },
                "prompt_strategy_used": config.get('prompt_strategy', 'default')
            }
            
            logger.info(f"Success - Type: {response_data['classification']['type']}, "
                       f"Difficulty: {response_data['classification']['difficulty']}, "
                       f"Strategy: {response_data['prompt_strategy_used']}")
            
            return jsonify(response_data)
        else:
            logger.error("No rewritten text in agent result")
            return jsonify({
                "status": "error",
                "error": "Failed to generate rewritten response"
            }), 500
    
    except Exception as e:
        logger.error(f"Error in rewrite_response: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/classify', methods=['POST'])
def classify_query():
    """Endpoint to only classify a query without rewriting."""
    try:
        data = request.json
        
        if not data or 'user_text' not in data:
            return jsonify({"error": "Missing 'user_text' in request body"}), 400
        
        user_text = data['user_text']
        bix_response = data.get('bix_response', '')
        
        # Create state for classification
        state = RewriteState(
            user_text=user_text,
            bixby_text=bix_response
        )
        
        # Run only the classification node
        classified_state = rewriter_agent.classify_node(state)
        
        # Detect special cases using agent's method
        special_case = rewriter_agent.detect_special_case(user_text, bix_response)
        
        response_data = {
            "status": "success",
            "classification": {
                "type": str(classified_state.query_type),
                "difficulty": str(classified_state.query_difficulty)
            },
            "special_case": special_case
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in classify_query: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/benchmark', methods=['POST'])
def benchmark_prompts():
    """Benchmark all prompt strategies on the same input."""
    try:
        data = request.json
        
        if not data or 'user_text' not in data or 'bix_response' not in data:
            return jsonify({"error": "Missing required fields"}), 400
        
        user_text = data['user_text']
        bix_response = data['bix_response']
        
        results = {}
        
        # Test each prompt strategy
        for strategy_name, prompt_template in PROMPT_MAP.items():
            try:
                state = RewriteState(
                    user_text=user_text,
                    bixby_text=bix_response
                )
                
                config = rewriter_agent.config.copy()
                config['override_prompt'] = prompt_template
                config['prompt_strategy'] = strategy_name
                
                result = rewriter_agent.invoke_graph(state, config)
                
                results[strategy_name] = {
                    "success": True,
                    "response": result.rewritten_text if hasattr(result, 'rewritten_text') else 'No output',
                    "length": len(result.rewritten_text) if hasattr(result, 'rewritten_text') else 0
                }
            except Exception as e:
                results[strategy_name] = {
                    "success": False,
                    "response": f"Error: {str(e)}",
                    "length": 0
                }
        
        # Also test hybrid prompt
        try:
            state = RewriteState(user_text=user_text, bixby_text=bix_response)
            config = rewriter_agent.config.copy()
            config['override_prompt'] = HYBRID_PROMPT
            config['prompt_strategy'] = 'hybrid'
            
            result = rewriter_agent.invoke_graph(state, config)
            results['hybrid'] = {
                "success": True,
                "response": result.rewritten_text if hasattr(result, 'rewritten_text') else 'No output',
                "length": len(result.rewritten_text) if hasattr(result, 'rewritten_text') else 0
            }
        except Exception as e:
            results['hybrid'] = {
                "success": False,
                "response": f"Error: {str(e)}",
                "length": 0
            }
        
        # Test smart selection
        try:
            state = RewriteState(user_text=user_text, bixby_text=bix_response)
            config = rewriter_agent.config.copy()
            config['use_smart_selection'] = True
            
            result = rewriter_agent.invoke_graph(state, config)
            results['smart_selection'] = {
                "success": True,
                "response": result.rewritten_text if hasattr(result, 'rewritten_text') else 'No output',
                "length": len(result.rewritten_text) if hasattr(result, 'rewritten_text') else 0,
                "classification": {
                    "type": str(result.query_type) if hasattr(result, 'query_type') else 'unknown',
                    "difficulty": str(result.query_difficulty) if hasattr(result, 'query_difficulty') else 'unknown'
                }
            }
        except Exception as e:
            results['smart_selection'] = {
                "success": False,
                "response": f"Error: {str(e)}",
                "length": 0
            }
        
        return jsonify({
            "status": "success",
            "benchmark_results": results
        })
    
    except Exception as e:
        logger.error(f"Error in benchmark_prompts: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=True
    )
