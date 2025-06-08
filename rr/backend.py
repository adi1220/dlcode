from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from config import Config
import logging
from core.agents.response_rewriter.agent import RewriterAgent
from core.agents.response_rewriter.states import RewriteState
from prompts import PROMPT_MAP

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the RewriterAgent
# Get provider and model from environment or config
provider = os.getenv('provider', 'openai')  # or your default provider
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
        
        logger.info(f"Received rewrite request - Prompt type: {prompt_type}")
        
        # Create initial state
        initial_state = RewriteState(
            user_text=user_text,
            bixby_text=bix_response
        )
        
        # Check if we need to force a specific prompt type
        if prompt_type != 'default':
            # You might need to modify your agent to accept prompt type override
            # For now, we'll process normally and track which prompt was selected
            pass
        
        # Invoke the agent graph
        result = rewriter_agent.invoke_graph(initial_state, rewriter_agent.config)
        
        # Extract the rewritten response and classification info
        if hasattr(result, 'rewritten_text') and result.rewritten_text:
            response_data = {
                "status": "success",
                "rewritten_response": result.rewritten_text,
                "classification": {
                    "type": getattr(result, 'type', 'unknown'),
                    "difficulty": getattr(result, 'difficulty', 'unknown')
                }
            }
            
            # Log which prompt strategy was used
            logger.info(f"Classification - Type: {response_data['classification']['type']}, "
                       f"Difficulty: {response_data['classification']['difficulty']}")
            
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

@app.route('/rewrite-with-prompt', methods=['POST'])
def rewrite_with_specific_prompt():
    """Alternative endpoint that forces a specific prompt type."""
    try:
        data = request.json
        
        # Validate input
        if not data or 'user_text' not in data or 'bix_response' not in data:
            return jsonify({"error": "Missing required fields"}), 400
        
        user_text = data['user_text']
        bix_response = data['bix_response']
        prompt_type = data.get('prompt_type', 'default')
        
        # Validate prompt type
        if prompt_type not in PROMPT_MAP:
            return jsonify({"error": f"Invalid prompt type: {prompt_type}"}), 400
        
        logger.info(f"Forcing prompt type: {prompt_type}")
        
        # Create a modified state that forces the prompt type
        initial_state = RewriteState(
            user_text=user_text,
            bixby_text=bix_response
        )
        
        # Get the specific prompt template
        prompt_template = PROMPT_MAP[prompt_type]
        
        # You might need to modify your agent to accept this override
        # For now, we'll use the standard flow
        config = rewriter_agent.config.copy()
        config['force_prompt_type'] = prompt_type
        
        # Invoke the agent
        result = rewriter_agent.invoke_graph(initial_state, config)
        
        response_data = {
            "status": "success",
            "rewritten_response": getattr(result, 'rewritten_text', ''),
            "prompt_type_used": prompt_type,
            "classification": {
                "type": getattr(result, 'type', 'unknown'),
                "difficulty": getattr(result, 'difficulty', 'unknown')
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in rewrite_with_specific_prompt: {str(e)}", exc_info=True)
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
