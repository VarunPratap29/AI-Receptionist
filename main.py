# =============================================================================
# IMPORTS & CONFIGURATION
# =============================================================================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import edge_tts
import base64
import os
import re
import httpx
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/ai-receptionist")

# Initialize Mistral Client
client = OpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)

# Initialize FastAPI App
app = FastAPI(title="AI Receptionist - GrowPilot")

# Enable CORS for n8n and external channels
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# SYSTEM PROMPT & MEMORY
# =============================================================================

RECEPTIONIST_PROMPT = """
You are Aria, a professional, polite, and helpful AI receptionist for GrowPilot, a modern tech company.

## LANGUAGE RULE (CRITICAL):
- Detect the language the user is speaking and REPLY IN THAT EXACT LANGUAGE.
- If the user speaks HINDI or HINGLISH, reply in fluent, natural Hinglish.
- If the user speaks ENGLISH, reply in fluent, professional English.
- If the user says "Hindi please", switch to Hinglish immediately.

## FORMATTING RULE (DO NOT FORGET):
1. NEVER use emojis.
2. You MUST end EVERY response with exactly ONE of these hidden tags: 
   - End with [LANG:HI] if you spoke Hindi/Hinglish.
   - End with [LANG:EN] if you spoke English.
   Example Hindi: "Bilkul, main kar deta hoon. [LANG:HI]"
   Example English: "Sure, I can help with that. [LANG:EN]"

## VOICE DATA COLLECTION RULE (CRITICAL):
- NEVER ask for an email address over voice. Voice AI cannot understand "at the rate".
- ALWAYS ask for a 10-digit PHONE NUMBER first.
- Once the user provides their phone number, say: "Main is number par WhatsApp par ek message bhej rahi hoon. Usme aap apna email aur reason type kar dijiye." (I am sending a message on this WhatsApp number. Please type your email and reason in it.)
- When repeating a phone number back, put a space between EVERY digit (e.g., 7 9 0 6 2 3 0 5 4 7).

## Guidelines:
- Keep answers VERY CONCISE (2-3 short sentences max)
- Be friendly but professional

## Your Role:
- Answer questions about GrowPilot
- Schedule appointments and consultations
- Collect visitor information (name, email, phone, date/time)
- Escalate complex issues to human staff

## Company Info:
- Services: Software Development, AI Solutions, Cloud Consulting, Technical Support
- Hours: Monday-Friday 9AM-6PM, Saturday 10AM-2PM
- Location: 123 Tech Avenue, Silicon Valley, CA
- Phone: (555) 123-4567
- Email: hello@growpilot.com

## Guidelines:
- Keep answers VERY CONCISE (2-3 short sentences max)
- Be friendly but professional
"""

# In-memory storage (Replace with database in production)
conversation_sessions: Dict[str, Dict] = {}
lead_captures: list = []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_contact_info(text: str) -> Dict[str, str]:
    """Extracts email and phone numbers, handling spoken formats like 'at the rate'."""
    info = {}
    
    # Handle spoken email formats
    text_for_email = text.lower().replace(" at the rate ", "@").replace(" at rate ", "@").replace(" at the rate of ", "@")
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text_for_email)
    if email_match:
        info["email"] = email_match.group()
    
    # Handle phone numbers by removing ALL spaces to catch "7906230 547"
    text_for_phone = text.replace(" ", "").replace("-", "")
    if re.search(r'\d{10}', text_for_phone):
        match = re.search(r'\d{10}', text_for_phone)
        info["phone"] = match.group()
        
    return info

def detect_language(text: str) -> str:
    """Detects Hindi/Hinglish vs English using a massive root-word list."""
    # 1. Check for actual Hindi script (Devanagari)
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"
    
    # 2. Extract pure words
    words = set(re.findall(r'\b\w+\b', text.lower()))
    
    # 3. Massive Hinglish dictionary (roots, verbs, postpositions)
    hindi_indicators = {
        'main', 'mein', 'mai', 'mujhe', 'mujhse', 'aap', 'aapko', 'tum', 'tumhe', 'hum', 'humein', 
        'uska', 'uski', 'iska', 'iski', 'apna', 'apni', 'mera', 'meri', 'tumhara',
        'hai', 'hain', 'hoon', 'hoga', 'honge', 'hoti', 'hota', 'tha', 'thi', 'the', 'hona',
        'karna', 'kare', 'karo', 'karu', 'kari', 'karni', 'kiya', 'karega', 'karo',
        'bolna', 'bolo', 'bolu', 'bole', 'bola', 'boli',
        'jana', 'jao', 'jaou', 'jaaye', 'gayi', 'gaya', 'jaenge',
        'aana', 'aao', 'aayi', 'aaya', 'aayenge',
        'dena', 'do', 'diya', 'di', 'de', 'deta', 'deti', 'denge', 'dedo',
        'lena', 'lo', 'liya', 'li', 'le', 'leta', 'leti', 'lenge', 'lelo',
        'chahiye', 'chahte', 'chahti', 'chahta',
        'ka', 'ki', 'ke', 'ko', 'se', 'pe', 'par', 'tak', 'andar', 'bahar', 'saath', 'bina', 'liye', 'pehle', 'baad',
        'kya', 'kaise', 'kab', 'kahan', 'kyun', 'kyon', 'kaun', 'kitna', 'kitne', 'kis',
        'haan', 'nahi', 'bilkul', 'theek', 'thik', 'achha', 'accha', 'sahi', 'zaroor', 'abhi', 
        'aaj', 'kal', 'parso', 'shukriya', 'namaste', 'swagat', 'ji', 'bhai', 'sahab', 
        'chal', 'chalo', 'dekho', 'suno', 'mat', 'kuch', 'yahan', 'wahan', 'sab', 'log', 'baat'
    }
    
    # 4. If 2 or more Hindi words match, it's Hindi
    matches = sum(1 for word in words if word in hindi_indicators)
    return "hi" if matches >= 2 else "en"

def analyze_intent(user_message: str, ai_response: str) -> Dict[str, Any]:
    """Analyzes message to determine intent type."""
    intent = {
        "type": "general",
        "needs_escalation": False,
        "is_lead": False,
        "wants_appointment": False,
        "has_contact_info": False
    }
    
    text = user_message.lower()
    
    if any(word in text for word in ["appointment", "schedule", "book", "meet", "consultation"]):
        intent["type"] = "appointment"
        intent["wants_appointment"] = True
    elif any(word in text for word in ["human", "real person", "manager", "complaint"]):
        intent["type"] = "escalation"
        intent["needs_escalation"] = True
    elif any(word in text for word in ["price", "cost", "pricing", "quote", "how much"]):
        intent["type"] = "pricing"
        intent["is_lead"] = True
        
    if extract_contact_info(user_message):
        intent["has_contact_info"] = True
        intent["is_lead"] = True
        
    return intent

async def send_to_n8n(payload: Dict[str, Any]):
    """Sends data to n8n webhook silently."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as n8n_client:
            await n8n_client.post(N8N_WEBHOOK_URL, json=payload)
            print(f"   -> Notified n8n: {payload.get('event', 'unknown')}")
    except Exception as e:
        print(f"   ⚠️ n8n notification failed: {e}")


# =============================================================================
# WEB SOCKET ENDPOINT (Voice Phone Interface)
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    conversation_history = [{"role": "system", "content": RECEPTIONIST_PROMPT}]
    visitor_info = {"source": "voice_phone", "session_id": session_id}
    
    conversation_sessions[session_id] = {
        "history": conversation_history,
        "visitor_info": visitor_info,
        "created_at": datetime.now().isoformat()
    }

    try:
        while True:
            # 1. Receive text from browser's microphone
            data = await websocket.receive_json()
            user_message = data.get("text", "")
            
            if not user_message:
                continue

            print(f"\n[{session_id[:8]}] Caller said: {user_message}")

            # 2. Extract any contact info (Handles spaces and "at the rate")
            extracted_info = extract_contact_info(user_message)
            if extracted_info:
                visitor_info.update(extracted_info)
                print(f"[{session_id[:8]}] Extracted info: {extracted_info}")

                       # 3. Detect language using Python (Vote 1)
            python_lang = detect_language(user_message)

            # 4. Add user message to memory and send to Mistral
            conversation_history.append({"role": "user", "content": user_message})
            
            response = client.chat.completions.create(
                model="mistral-small-latest",
                messages=conversation_history,
                temperature=0.7
            )
            
            raw_ai_message = response.choices[0].message.content.strip()
            
            # 5. Detect language from AI's hidden tag (Vote 2)
            ai_lang = "en" # Default fallback
            if "[LANG:HI]" in raw_ai_message:
                ai_lang = "hi"
            
            # Clean the message (Remove the tag before saving or speaking)
            ai_message = raw_ai_message.replace("[LANG:HI]", "").replace("[LANG:EN]", "").strip()
            
            # 6. THE JUDGE: Cross-Verify and decide final language
            if python_lang == ai_lang:
                final_lang = python_lang # They agree, 100% confidence
            elif python_lang == "hi":
                final_lang = "hi" # Python found hard Hindi words (e.g., 'karna', 'hai'), trust it over AI
            else:
                final_lang = ai_lang # Python saw English, but AI understood context (e.g., 'Hindi please'), trust AI
                
            # Pick the final voice
            voice = "hi-IN-SwaraNeural" if final_lang == "hi" else "en-US-AriaNeural"

            # Save CLEAN message to memory (no tags)
            conversation_history.append({"role": "assistant", "content": ai_message})
            
            print(f"[{session_id[:8]}] Aria said ({final_lang.upper()} | Py:{python_lang} AI:{ai_lang}): {ai_message}")

             # 7. Analyze intent and send to n8n in background
            intent = analyze_intent(user_message, ai_message)
            
            # Create the base payload
            n8n_payload = {
                "event": "message_processed",
                "session_id": session_id,
                "user_message": user_message,
                "ai_response": ai_message,
                "intent": intent,
                "visitor_info": visitor_info,
                "timestamp": datetime.now().isoformat()
            }
            
            # TRIGGER: If we just extracted a phone number, tell n8n to send WhatsApp!
            if extracted_info and extracted_info.get("phone"):
                n8n_payload["event"] = "trigger_whatsapp_form"
                
            asyncio.create_task(send_to_n8n(n8n_payload))

            # 8. Convert AI text to speech
            communicate = edge_tts.Communicate(ai_message, voice=voice)
            temp_audio_path = f"temp_{session_id[:8]}.mp3"
            await communicate.save(temp_audio_path)

            # 9. Read audio, encode to base64, and send
            with open(temp_audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            await websocket.send_json({
                "text": ai_message,
                "audio": audio_base64,
                "intent": intent
            })

            # 10. Cleanup temp file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    except WebSocketDisconnect:
        print(f"\n[{session_id[:8]}] Caller hung up.")
        has_lead_info = visitor_info.get("email") or visitor_info.get("phone")
        await send_to_n8n({
            "event": "session_ended",
            "session_id": session_id,
            "visitor_info": visitor_info,
            "message_count": len(conversation_history) - 1,
            "is_qualified_lead": bool(has_lead_info),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"[{session_id[:8]}] Error: {e}")


# =============================================================================
# HTTP REST ENDPOINTS (For n8n & Website Chat Integrations)
# =============================================================================

@app.get("/")
async def get():
    """Serves the HTML Voice Phone interface."""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/chat")
async def http_chat(request: Request):
    """HTTP endpoint for text-based channels (Website widget, n8n routing)."""
    try:
        data = await request.json()
        message = data.get("message", "")
        session_id = data.get("session_id", str(uuid.uuid4()))
        channel = data.get("channel", "website")
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = {
                "history": [{"role": "system", "content": RECEPTIONIST_PROMPT}],
                "visitor_info": {"source": channel, "session_id": session_id},
                "created_at": datetime.now().isoformat()
            }
        
        session = conversation_sessions[session_id]
        session["history"].append({"role": "user", "content": message})
        
        response = client.chat.completions.create(
            model="mistral-small-latest",
            messages=session["history"][-10:],
            temperature=0.7
        )
        
        ai_message = response.choices[0].message.content
        session["history"].append({"role": "assistant", "content": ai_message})

        intent = analyze_intent(message, ai_message)
        await send_to_n8n({
            "event": "message_processed",
            "session_id": session_id,
            "user_message": message,
            "ai_response": ai_message,
            "intent": intent,
            "visitor_info": session["visitor_info"],
            "timestamp": datetime.now().isoformat()
        })

        return JSONResponse({
            "response": ai_message,
            "session_id": session_id,
            "intent": intent
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lead")
async def capture_lead(request: Request):
    """Direct endpoint to capture leads (can be called by n8n)."""
    data = await request.json()
    lead = {
        "id": str(uuid.uuid4()),
        **data,
        "captured_at": datetime.now().isoformat()
    }
    lead_captures.append(lead)
    await send_to_n8n({"event": "lead_capture", "data": lead})
    return {"success": True, "lead_id": lead["id"]}

@app.post("/api/appointment")
async def create_appointment(request: Request):
    """Direct endpoint to book appointments (can be called by n8n)."""
    data = await request.json()
    appointment = {
        "id": str(uuid.uuid4()),
        **data,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    await send_to_n8n({"event": "appointment_request", "data": appointment})
    return {"success": True, "appointment_id": appointment["id"]}

@app.get("/api/stats")
async def get_stats():
    """Returns basic stats for a dashboard."""
    return {
        "active_sessions": len(conversation_sessions),
        "total_leads": len(lead_captures),
        "recent_leads": lead_captures[-5:]
    }