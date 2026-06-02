from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import OpenAI
import edge_tts
import base64
import asyncio
import os
from dotenv import load_dotenv

# Load the API key from .env file
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Setup Mistral client using the OpenAI library format
client = OpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)

app = FastAPI()

# This is the system prompt - the "Personality" of your receptionist
RECEPTIONIST_PROMPT = """
You are Aria, a professional, polite, and helpful AI receptionist for a modern tech company called Meridian Corp. 
Your goal is to answer questions, schedule appointments, and direct calls. 
Keep your answers concise and conversational since you are speaking out loud.
"""

@app.get("/")
async def get():
    # This serves the HTML "Phone" interface
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    conversation_history = [
        {"role": "system", "content": RECEPTIONIST_PROMPT}
    ]

    try:
        while True:
            # 1. Receive the text from the browser's microphone
            data = await websocket.receive_json()
            user_message = data.get("text", "")
            
            if not user_message:
                continue

            print(f"Caller said: {user_message}")

            # 2. Add user message to history and ask Mistral
            conversation_history.append({"role": "user", "content": user_message})
            
            response = client.chat.completions.create(
                model="mistral-small-latest",
                messages=conversation_history
            )
            
            ai_message = response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": ai_message})
            
            print(f"Aria said: {ai_message}")

            # 3. Convert AI text to Speech using Edge-TTS
            communicate = edge_tts.Communicate(ai_message, voice="en-US-AriaNeural")
            
            # Generate the audio file temporarily
            temp_audio_path = "temp_response.mp3"
            await communicate.save(temp_audio_path)

            # 4. Read the audio file and send it back to the browser
            with open(temp_audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            # Convert audio to base64 so we can send it over WebSocket
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            await websocket.send_json({
                "text": ai_message,
                "audio": audio_base64
            })

            # Clean up the temp file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    except WebSocketDisconnect:
        print("Caller hung up.")
    except Exception as e:
        print(f"Error: {e}")
