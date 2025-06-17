"""
Flask Backend for Response Rewriter Arena
Enhanced for arena-style evaluation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import time
from datetime import datetime
from typing import Dict, Optional

from core.agents.response_rewriter.agent import RewriterAgent
from core.agents.response_rewriter.states import RewriteState
from core.agents.response_rewriter.prompts import PROMPT_MAP, get_prompt_for_situation, HYBRID_PROMPT

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
from config import Config

# Initialize the RewriterAgent
provider = os.getenv('provider', 'openai')
model = os.getenv('model', Config.LLM_MODEL)

logger.info(f"Initializing RewriterAgent with provider: {provider}, model: {model}")

try:
    rewriter_agent = RewriterAgent("rewriter_agent", provider, model)
    rewriter_agent.build_graph()
    logger.info("RewriterAgent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RewriterAgent: {str(e)}")
    rewriter_agent = None

# Cache for responses to speed up arena battles
response_cache: Dict[str, Dict] = {}

def get_cache_key(user_text: str, bix_response: str, strategy: str) -> str:
    """Generate cache key for response."""
    return f"{hash(user_text)}_{hash(bix_response)}_{strategy}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if rewriter_agent else "unhealthy",
        "service": "response-rewriter-arena-backend",
        "timestamp": datetime.now().isoformat(),
        "agent_status": "ready" if rewriter_agent else "not initialized"
    })

@app.route('/rewrite', methods=['POST'])
def rewrite_response():
    """Main endpoint to rewrite responses."""
    start_time = time.time()
    
    try:
        data = request.json
        
        # Validate input
        if not data or 'user_text' not in data or 'bix_response' not in data:
            return jsonify({"error": "Missing required fields"}), 400
        
        user_text = data['user_text']
        bix_response = data['bix_response']
        prompt_type = data.get('prompt_type', 'default')
        use_cache = data.get('use_cache', True)
        
        # Check cache first
        cache_key = get_cache_key(user_text, bix_response, prompt_type)
        if use_cache and cache_key in response_cache:
            logger.info(f"Cache hit for strategy: {prompt_type}")
            cached = response_cache[cache_key]
            cached['from_cache'] = True
            cached['processing_time'] = 0.001
            return jsonify(cached)
        
        logger.info(f"Processing rewrite request - Strategy: {prompt_type}")
        
        if not rewriter_agent:
            return jsonify({
                "status": "error",
                "error": "Agent not initialized"
            }), 503
        
        # Create initial state
        initial_state = RewriteState(
            user_text=user_text,
            bixby_text=bix_response
        )
        
        # Prepare config based on strategy
        config = rewriter_agent.config.copy()
        
        if prompt_type == "smart":
            # Use smart selection
            config['use_smart_selection'] = True
        elif prompt_type == "hybrid":
            # Use hybrid prompt
            config['override_prompt'] = HYBRID_PROMPT
            config['prompt_strategy'] = "hybrid"
        elif prompt_type in PROMPT_MAP:
            # Use specific strategy
            config['override_prompt'] = PROMPT_MAP[prompt_type]
            config['prompt_strategy'] = prompt_type
        else:
            # Default strategy
            config['prompt_strategy'] = "default"
        
        # Invoke the agent
        result = rewriter_agent.invoke_graph(initial_state, config)
        
        # Extract results
        if hasattr(result, 'rewritten_text') and result.rewritten_text:
            processing_time = time.time() - start_time
            
            response_data = {
                "status": "success",
                "rewritten_response": result.rewritten_text,
                "classification": {
                    "type": str(result.query_type) if hasattr(result, 'query_type') else 'unknown',
                    "difficulty": str(result.query_difficulty) if hasattr(result, 'query_difficulty') else 'unknown'
                },
                "prompt_strategy_used": getattr(result, 'prompt_strategy_used', prompt_type),
                "processing_time": round(processing_time, 3),
                "from_cache": False,
                "metadata": {
                    "response_length": len(result.rewritten_text),
                    "original_length": len(bix_response),
                    "length_change": len(result.rewritten_text) - len(bix_response)
                }
            }
            
            # Cache the response
            if use_cache:
                response_cache[cache_key] = response_data.copy()
                # Limit cache size
                if len(response_cache) > 1000:
                    # Remove oldest entries
                    for key in list(response_cache.keys())[:100]:
                        del response_cache[key]
            
            logger.info(f"Success - Type: {response_data['classification']['type']}, "
                       f"Difficulty: {response_data['classification']['difficulty']}, "
                       f"Strategy: {response_data['prompt_strategy_used']}, "
                       f"Time: {processing_time:.3f}s")
            
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
            "error": str(e),
            "processing_time": time.time() - start_time
        }), 500

@app.route('/batch_rewrite', methods=['POST'])
def batch_rewrite():
    """Batch rewrite endpoint for arena battles."""
    try:
        data = request.json
        
        if not data or 'requests' not in data:
            return jsonify({"error": "Missing requests array"}), 400
        
        results = []
        
        for req in data['requests']:
            # Process each request
            user_text = req.get('user_text', '')
            bix_response = req.get('bix_response', '')
            strategy = req.get('prompt_type', 'default')
            
            if not user_text or not bix_response:
                results.append({
                    "status": "error",
                    "error": "Missing required fields"
                })
                continue
            
            # Use the regular rewrite logic
            req_data = {
                "user_text": user_text,
                "bix_response": bix_response,
                "prompt_type": strategy,
                "use_cache": True
            }
            
            # Make internal call
            with app.test_request_context(json=req_data):
                response = rewrite_response()
                results.append(response.get_json())
        
        return jsonify({
            "status": "success",
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error in batch_rewrite: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/strategies', methods=['GET'])
def get_strategies():
    """Get available strategies."""
    strategies = []
    
    for key, prompt in PROMPT_MAP.items():
        # Extract description from prompt
        desc_start = prompt.find('**')
        desc_end = prompt.find('**', desc_start + 2) if desc_start >= 0 else -1
        description = prompt[desc_start+2:desc_end] if desc_start >= 0 and desc_end > desc_start else key
        
        strategies.append({
            "id": key,
            "name": key.upper(),
            "description": description[:100] + "..." if len(description) > 100 else description
        })
    
    # Add special strategies
    strategies.extend([
        {
            "id": "hybrid",
            "name": "HYBRID",
            "description": "Combines best aspects of all strategies"
        },
        {
            "id": "smart",
            "name": "SMART SELECTION",
            "description": "AI automatically selects the best strategy"
        }
    ])
    
    return jsonify({
        "status": "success",
        "strategies": strategies,
        "total": len(strategies)
    })

@app.route('/classify', methods=['POST'])
def classify_query():
    """Classify a query without rewriting."""
    try:
        data = request.json
        
        if not data or 'user_text' not in data:
            return jsonify({"error": "Missing 'user_text'"}), 400
        
        user_text = data['user_text']
        bix_response = data.get('bix_response', '')
        
        if not rewriter_agent:
            return jsonify({"error": "Agent not initialized"}), 503
        
        # Create state
        state = RewriteState(
            user_text=user_text,
            bixby_text=bix_response
        )
        
        # Run classification
        classified_state = rewriter_agent.classify_node(state)
        
        # Detect special cases
        special_case = rewriter_agent.detect_special_case(user_text, bix_response)
        
        # Get recommended strategy
        recommended_strategy = get_prompt_for_situation(
            query_type=str(classified_state.query_type),
            difficulty=str(classified_state.query_difficulty),
            special_case=special_case
        )
        
        # Find which strategy this maps to
        strategy_name = "custom"
        for name, template in PROMPT_MAP.items():
            if template == recommended_strategy:
                strategy_name = name
                break
        
        return jsonify({
            "status": "success",
            "classification": {
                "type": str(classified_state.query_type),
                "difficulty": str(classified_state.query_difficulty)
            },
            "special_case": special_case,
            "recommended_strategy": strategy_name
        })
    
    except Exception as e:
        logger.error(f"Error in classify_query: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the response cache."""
    try:
        cache_size = len(response_cache)
        response_cache.clear()
        
        return jsonify({
            "status": "success",
            "message": f"Cleared {cache_size} cached responses"
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get backend statistics."""
    return jsonify({
        "status": "success",
        "stats": {
            "cache_size": len(response_cache),
            "agent_status": "ready" if rewriter_agent else "not initialized",
            "available_strategies": len(PROMPT_MAP) + 2,  # +2 for hybrid and smart
            "uptime": datetime.now().isoformat()
        }
    })

@app.after_request
def log_performance(response):
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        if duration > 3:
            logger.warning(f"Slow request: {duration:.2f}s - {request.path}")
    return response
    
# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    # Print startup information
    print(f"""
    ╔═══════════════════════════════════════════════╗
    ║     Response Rewriter Arena Backend           ║
    ║═══════════════════════════════════════════════║
    ║ Provider: {provider:<35} ║
    ║ Model: {model:<38} ║
    ║ Host: {Config.FLASK_HOST:<39} ║
    ║ Port: {Config.FLASK_PORT:<39} ║
    ╚═══════════════════════════════════════════════╝
    
    Starting Flask server...
    """)
    
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.DEBUG
    )
