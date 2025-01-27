import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
import aiohttp

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
SYSTEM_MESSAGE = """
You are Cognify's dedicated AI learning assistant, designed to be your personal academic companion. Your responses should be:

1. Educational and Insightful
   - Explain concepts clearly and thoroughly
   - Draw connections between related topics
   - Break down complex ideas into understandable parts

2. Encouraging and Supportive
   - Maintain a positive, motivating tone
   - Celebrate understanding and progress
   - Guide users through challenging concepts patiently

3. Professional yet Approachable
   - Speak like an experienced, friendly tutor
   - Balance academic precision with conversational warmth
   - Use clear, concise language while remaining engaging

4. Contextually Aware
   - Reference relevant class materials when available
   - Connect discussions to previous lessons when applicable
   - Suggest related topics or resources that might be helpful

Remember: You're not just providing information, you're supporting their learning journey. Help them understand not just the 'what' but also the 'why' and 'how' of their academic questions."""

VOICE = 'sage'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]
SHOW_TIMING_MATH = False

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    # Get form data from the request
    form_data = await request.form()
    
    # Get the caller's phone number
    caller_number = form_data.get('From', 'Unknown')
    print(f"Incoming call from: {caller_number}")
    
    response = VoiceResponse()
    response.say("We are connecting you to Cognify. Please wait while we verify your account.")
    response.pause(length=1)
    
    # Setup API request
    DATABASE_API_KEY = os.getenv('DATABASE_API_KEY')
    api_url = "https://www.cognify.cc/education/api/user-data-by-phone/"
    headers = {"X-API-Key": DATABASE_API_KEY}
    payload = {"phone_number": caller_number}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, headers=headers) as api_response:
                if api_response.status == 200:
                    # Get user data and save to file
                    user_data = await api_response.json()
                    temp_filename = f"tempdata_{caller_number.replace('+', '')}.json"
                    try:
                        with open(temp_filename, 'w') as f:
                            json.dump(user_data, f)
                    except Exception as e:
                        print(f"Error saving data: {e}")
                    
                    # User found, proceed with connection
                    response.say("Thank you for waiting! Connected! Ask what you need!")
                    host = request.url.hostname
                    connect = Connect()
                    connect.stream(url=f'wss://{host}/media-stream')
                    response.append(connect)
                else:
                    # User not found or other error
                    response.say("Sorry, it seems your phone number is not registered. Please make sure to create an account at cognify dot cc and then link your number. Have a great day!")
    except Exception as e:
        print(f"API request error: {e}")
        response.say("We're experiencing technical difficulties. Please try again later.")
    
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()
    
    openai_ws = None
    try:
        openai_ws = await websockets.connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview',
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        )
        await initialize_session(openai_ws)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
                    elif data['event'] == 'stop':
                        print(f"Stream {stream_sid} has ended")
                        raise WebSocketDisconnect()
            except WebSocketDisconnect:
                print("Client disconnected.")
                raise  # Re-raise to trigger cleanup in outer try/except

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    # Handle function calls
                    if response.get('type') == 'response.output_item.done':
                        item = response.get('item', {})
                        if item.get('type') == 'function_call':
                            if item['name'] == 'dummy_function':
                                result = await dummy_function()
                                # Send function result back to OpenAI
                                await openai_ws.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": item['call_id'],
                                        "output": json.dumps(result)
                                    }
                                }))
                                # Request response generation
                                await openai_ws.send(json.dumps({"type": "response.create"}))

                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)

                        if response_start_timestamp_twilio is None:
                            response_start_timestamp_twilio = latest_media_timestamp
                            if SHOW_TIMING_MATH:
                                print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                        # Update last_assistant_item safely
                        if response.get('item_id'):
                            last_assistant_item = response['item_id']

                        await send_mark(websocket, stream_sid)

                    # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
                    if response.get('type') == 'input_audio_buffer.speech_started':
                        print("Speech started detected.")
                        if last_assistant_item:
                            print(f"Interrupting response with id: {last_assistant_item}")
                            await handle_speech_started_event()
            except websockets.exceptions.ConnectionClosed:
                print("OpenAI connection closed")
                raise WebSocketDisconnect()
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")
                raise WebSocketDisconnect()

        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech started event.")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        # Run both tasks and wait for either to complete/fail
        await asyncio.gather(receive_from_twilio(), send_to_twilio())
        
    except WebSocketDisconnect:
        print("Cleaning up connections...")
    except Exception as e:
        print(f"Error in handle_media_stream: {e}")
    finally:
        # Clean up OpenAI connection
        if openai_ws and not openai_ws.closed:
            await openai_ws.close()
            print("OpenAI connection closed")
        
        # Clean up Twilio connection
        if not websocket.client_state.disconnected:
            await websocket.close()
            print("Twilio connection closed")
        
        # Clean up any temp files if they exist
        try:
            temp_files = [f for f in os.listdir('.') if f.startswith('tempdata_')]
            for f in temp_files:
                os.remove(f)
                print(f"Removed temp file: {f}")
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there! I am an AI voice assistant powered by Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or anything you can imagine. How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))

async def dummy_function():
    """Simple dummy function that returns a fixed string."""
    return "duck duck go, function called and ready to go"

async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "dummy_function",
                    "description": "A test function that always returns the same string",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }]
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    # await send_initial_conversation_item(openai_ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)