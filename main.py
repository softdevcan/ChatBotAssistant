import json
import os
import uuid
from flask import Flask, request, jsonify
from groq import Groq
import custom_functions
from waitress import serve

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")

# Create Flask app
app = Flask(__name__)

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# In-memory storage for conversation threads
conversations = {}

# Create or load assistant configuration
assistant_config = custom_functions.create_assistant(client)


# Start conversation thread
@app.route('/start', methods=['GET'])
def start_conversation():
    print("Starting a new conversation...")
    thread_id = str(uuid.uuid4())

    # Initialize conversation with system message
    conversations[thread_id] = []

    print(f"New thread created with ID: {thread_id}")
    return jsonify({"thread_id": thread_id})


# Generate response
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    thread_id = data.get('thread_id')
    user_input = data.get('message', '')

    if not thread_id:
        print("Error: Missing thread_id")
        return jsonify({"error": "Missing thread_id"}), 400

    if thread_id not in conversations:
        print("Error: Invalid thread_id")
        return jsonify({"error": "Invalid thread_id"}), 400

    print(f"Received message: {user_input} for thread ID: {thread_id}")

    # Add user message to conversation history
    conversations[thread_id].append({"role": "user", "content": user_input})

    try:
        # Process message with assistant using custom_functions
        response, updated_config = custom_functions.process_message_with_assistant(
            user_input,
            conversations[thread_id],
            assistant_config
        )

        if not response:
            return jsonify({"error": "Failed to get response from assistant"}), 500

        assistant_message = response.choices[0].message

        # Check if the model wants to call a function
        if assistant_message.tool_calls:
            # Add assistant message with tool calls to conversation
            conversations[thread_id].append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    } for tool_call in assistant_message.tool_calls
                ]
            })

            # Process function calls
            tool_outputs = []
            for tool_call in assistant_message.tool_calls:
                if tool_call.function.name == "create_lead":
                    try:
                        # Process lead creation using custom_functions
                        arguments = json.loads(tool_call.function.arguments)
                        name = arguments.get('name', '')
                        company_name = arguments.get('company_name', '')
                        phone = arguments.get('phone', '')
                        email = arguments.get('email', '')

                        output = custom_functions.create_lead(name, company_name, phone, email)

                        # Add function result to conversation
                        conversations[thread_id].append({
                            "role": "tool",
                            "content": json.dumps(output),
                            "tool_call_id": tool_call.id
                        })

                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(output)
                        })

                    except json.JSONDecodeError as e:
                        print(f"Error parsing function arguments: {e}")
                        error_output = {"error": "Invalid function arguments"}
                        conversations[thread_id].append({
                            "role": "tool",
                            "content": json.dumps(error_output),
                            "tool_call_id": tool_call.id
                        })
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(error_output)
                        })

            # Get final response after function execution
            # Prepare messages for final API call
            messages = [{"role": "system", "content": assistant_config["system_prompt"]}]

            # Add conversation history excluding system messages
            for msg in conversations[thread_id]:
                if msg.get("role") != "system":
                    messages.append(msg)

            final_response = custom_functions.get_groq_response(
                messages,
                assistant_config["tools"]
            )

            if final_response:
                final_message = final_response.choices[0].message.content
                conversations[thread_id].append({"role": "assistant", "content": final_message})

                print(f"Assistant response: {final_message}")
                return jsonify({"response": final_message})
            else:
                return jsonify({"error": "Failed to get final response"}), 500

        else:
            # No function call, just regular response
            response_content = assistant_message.content
            conversations[thread_id].append({"role": "assistant", "content": response_content})

            print(f"Assistant response: {response_content}")
            return jsonify({"response": response_content})

    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": "Internal server error"}), 500


# Get conversation history
@app.route('/history', methods=['POST'])
def get_history():
    data = request.json
    thread_id = data.get('thread_id')

    if not thread_id or thread_id not in conversations:
        return jsonify({"error": "Invalid thread_id"}), 400

    # Filter out system messages and tool calls for cleaner history
    history = []
    for msg in conversations[thread_id]:
        if msg.get('role') in ['user', 'assistant'] and 'tool_calls' not in msg:
            history.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    return jsonify({"history": history})


# Clear conversation
@app.route('/clear', methods=['POST'])
def clear_conversation():
    data = request.json
    thread_id = data.get('thread_id')

    if not thread_id:
        return jsonify({"error": "Missing thread_id"}), 400

    if thread_id in conversations:
        # Reset conversation
        conversations[thread_id] = []
        return jsonify({"message": "Conversation cleared"})

    return jsonify({"error": "Invalid thread_id"}), 400


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "assistant_model": assistant_config.get("model", "unknown"),
        "active_conversations": len(conversations)
    })


# Get assistant info
@app.route('/assistant-info', methods=['GET'])
def get_assistant_info():
    return jsonify({
        "model": assistant_config.get("model", "unknown"),
        "tools_available": [tool["function"]["name"] for tool in assistant_config.get("tools", [])],
        "knowledge_base_loaded": assistant_config.get("knowledge_base") is not None
    })


if __name__ == '__main__':
    print("Starting Academy Club Assistant with Groq...")
    print(f"Model: {assistant_config.get('model', 'unknown')}")
    print(f"Knowledge base loaded: {assistant_config.get('knowledge_base') is not None}")
    serve(app, host='0.0.0.0', port=8080)