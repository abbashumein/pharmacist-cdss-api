import gradio as gr
import requests
import uuid

# Configuration constants targeting our active FastAPI LangGraph backend
BACKEND_URL = "http://127.0.0.1:8000/chat"

def generate_new_session():
    """Generates a pristine, isolated tracking token for LangGraph state isolation."""
    return str(uuid.uuid4())

def submit_agent_query(message, medication, session_id, history):
    """Bridges data payloads across the local network loop to trigger LangGraph nodes."""
    if not message.strip():
        return "", history, "⚠️ System Warning: Cannot route empty string through graph.", []
    
    # Initialize history as a list if it comes in as None
    if history is None:
        history = []
        
    # Construct the state contract request body
    payload = {
        "session_id": session_id,
        "message": message,
        "medication": medication if medication else "None"
    }
    
    try:
        response = requests.post(BACKEND_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            guidance = data.get("clinical_guidance", "Empty data structure.")
            emotions = data.get("detected_emotions", [])
            context_blocks = data.get("retrieved_database_context", [])
            
            # Append messages using Gradio 6's native role/content dictionary format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": guidance})
            
            # Construct a clean, human-readable layout for our backend telemetry variables
            telemetry_string = f"🧠 NLP EMOTION ENGINE STATE:\n{emotions}\n\n"
            telemetry_string += "🔧 ACTIVE RETRIEVED GRAPH CONTEXT STORES:\n"
            for i, chunk in enumerate(context_blocks):
                telemetry_string += f"📍 Source [{i+1}]: {chunk}\n"
                
            return "", history, telemetry_string, session_id
        else:
            return "", history, f"❌ Backend Error Status: {response.status_code}", session_id
    except Exception as e:
        return "", history, f"❌ Fatal Loop Connection Error: {str(e)}", session_id

def reset_entire_system():
    """Clears all volatile conversational history arrays and issues a fresh session key."""
    new_token = generate_new_session()
    return "", [], "🔄 Centralized graph state wiped. New Session initialized successfully.", new_token

# --- GRADIO BLOCKS USER INTERFACE DESIGN ---
with gr.Blocks(title="Pharmacist CDSS - LangGraph Playboard") as demo:
    
    # Structural Session State variables maintained safely behind the scenes
    session_holder = gr.State(value=generate_new_session)
    
    gr.Markdown("# 👨‍⚕️ Pharmacist Clinical Decision Support System")
    gr.Markdown("### Stateful LangGraph AI Agent Core Playground")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### 📋 Patient Assessment Inputs")
            med_input = gr.Textbox(
                label="💊 Active Maintenance Medication Profile", 
                placeholder="Type active prescriptions here (e.g., sertraline, warfarin)... Value defaults to 'None'",
                value="None"
            )
            msg_input = gr.Textbox(
                label="💬 Clinical Consultation Input", 
                placeholder="Describe current patient presentation or triage scenario...",
                lines=2
            )
            
            # The Dual Control Button Row explicitly handling workflows and system purges
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit Query to Agent", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Session Memory", variant="stop")
                
        with gr.Column(scale=1):
            gr.Markdown("#### 🔑 Execution Metadata")
            session_display = gr.Textbox(label="Active Session Token", interactive=False)

    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Conversational Dialogue Records")
            # Fixed: Removed the unexpected 'type' keyword argument
            chatbot_ui = gr.Chatbot(label="CDSS Orchestrated Dialogue Board")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Live State Machine Telemetry Panel")
            telemetry_ui = gr.Textbox(
                label="Asynchronous Runtime Graph Variables", 
                placeholder="Awaiting first input sequence loop execution...", 
                lines=12,
                interactive=False
            )

    # --- EVENT BINDING INTERACTORS ---
    demo.load(fn=lambda token: token, inputs=session_holder, outputs=session_display)
    
    submit_btn.click(
        fn=submit_agent_query,
        inputs=[msg_input, med_input, session_holder, chatbot_ui],
        outputs=[msg_input, chatbot_ui, telemetry_ui, session_display]
    )
    
    clear_btn.click(
        fn=reset_entire_system,
        inputs=[],
        outputs=[msg_input, chatbot_ui, telemetry_ui, session_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8501, theme=gr.themes.Soft())
