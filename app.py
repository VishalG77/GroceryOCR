import os
import json
import requests
from flask import Flask, request
from twilio.rest import Client
from dotenv import load_dotenv
import openai

load_dotenv()

app = Flask(__name__)

# ── Twilio ────────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN  = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_FROM        = os.environ["TWILIO_WHATSAPP_FROM"]

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── In-memory session store ───────────────────────────────────────────────────
# { "whatsapp:+27821234567": { "state": "...", "items": [...] } }
# States: "idle" | "awaiting_confirmation" | "edit_mode"
sessions: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — download media from Twilio
# ─────────────────────────────────────────────────────────────────────────────

def _download_media(url: str) -> tuple[bytes, str]:
    """Download any Twilio-hosted media; returns (raw_bytes, content_type)."""
    resp = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    resp.raise_for_status()
    return resp.content, resp.headers.get("Content-Type", "application/octet-stream")


# ─────────────────────────────────────────────────────────────────────────────
# PARSE ITEMS — common JSON extraction used by all input types
# ─────────────────────────────────────────────────────────────────────────────

def parse_items_from_text(text: str) -> list[dict]:
    """
    Ask GPT-4o to extract grocery items from free-form text.
    Returns a list of {"name": "...", "quantity": "..."} dicts.
    Used by both the voice transcription path and as a fallback.
    """
    prompt = (
        "Extract every grocery item from the following text and return ONLY a JSON array, "
        "no markdown, no explanation.\n"
        'Each element: {"name": "...", "quantity": "..."}.\n'
        "If quantity is unclear or not mentioned, use an empty string.\n\n"
        f"Text:\n{text}"
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# OCR  (GPT-4o Vision)
# ─────────────────────────────────────────────────────────────────────────────

def ocr_and_parse(image_url: str) -> list[dict]:
    """Download image from Twilio, base64-encode it, send to GPT-4o Vision."""
    import base64

    raw_bytes, content_type = _download_media(image_url)
    b64_image = base64.b64encode(raw_bytes).decode("utf-8")

    prompt = (
        "This is a photo of a grocery list. "
        "Extract every item and return ONLY a JSON array, no markdown, no explanation. "
        'Each element: {"name": "...", "quantity": "..."}. '
        "If quantity is unclear, use an empty string."
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{content_type};base64,{b64_image}"
                        }
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIBE  (OpenAI Whisper)
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(audio_url: str) -> str:
    """
    Download a voice note from Twilio and transcribe it using Whisper.
    Twilio sends OGG/Opus by default. Whisper handles it natively.
    Returns the transcribed plain text.
    """
    raw_bytes, content_type = _download_media(audio_url)

    # Whisper requires a file-like object with a name hint for the format
    import io
    audio_file = io.BytesIO(raw_bytes)
    audio_file.name = "voice_note.ogg"   # informs Whisper of the codec

    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return transcript.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# EDIT PARSING  (GPT-4o)
# ─────────────────────────────────────────────────────────────────────────────

def apply_edits(items: list[dict], edit_message: str) -> list[dict]:
    """
    Parse an edit message and apply changes to the item list.
    Supports natural-language commands:
      - Add:    "add Butter - 500g"
      - Update: "Sugar - 1kg" / "update 2 Sugar - 1kg"
      - Remove: "Remove Sugar" / "delete 3"
    """
    # Build a numbered list for context so GPT can resolve "delete 3" correctly
    numbered = "\n".join(
        f"{i+1}. {item['name']}" + (f" — {item['quantity']}" if item.get("quantity") else "")
        for i, item in enumerate(items)
    )

    prompt = (
        f"Current grocery list (numbered for reference):\n{numbered}\n\n"
        f"Current list as JSON:\n{json.dumps(items)}\n\n"
        f"User edit command:\n{edit_message}\n\n"
        "Apply the edits. Supported operations:\n"
        "- 'add Item - qty' → append new item\n"
        "- 'update N Item - qty' or 'Item - qty' → update/add that item\n"
        "- 'delete N' or 'remove Item' → remove item by number or name (case-insensitive)\n"
        "Return ONLY the updated JSON array, no markdown, no explanation. "
        'Each element: {"name": "...", "quantity": "..."}.'
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# ORDER DISPATCH  (stub — wire up your backend here)
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_order(items: list[dict], sender: str) -> None:
    """
    Called when a user confirms their final order.
    Currently just logs the order — replace or extend this to:
      - POST to your store API
      - Append to a Google Sheet
      - Send to an order management system
    """
    app.logger.info(f"ORDER CONFIRMED from {sender}: {json.dumps(items)}")
    # TODO: replace with your real order dispatch logic
    # e.g. requests.post("https://your-store-api.com/orders", json={"items": items, "customer": sender})


# ─────────────────────────────────────────────────────────────────────────────
# TWILIO HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def send_whatsapp(to: str, body: str) -> None:
    twilio_client.messages.create(from_=TWILIO_FROM, to=to, body=body)


def format_list_message(items: list[dict]) -> str:
    """Format the item list with numbers for easy reference during editing."""
    lines = "\n".join(
        f"{i+1}. {item['name']}" + (f" — {item['quantity']}" if item.get("quantity") else "")
        for i, item in enumerate(items)
    )
    return (
        f"🛒 Your list has *{len(items)} item(s)*:\n\n"
        f"{lines}\n\n"
        "Reply *1* to Confirm ✅\n"
        "Reply *2* to Edit ✏️"
    )


def format_confirmed_order(items: list[dict]) -> str:
    """Format the final confirmed order message."""
    lines = "\n".join(
        f"• {item['name']}" + (f" — {item['quantity']}" if item.get("quantity") else "")
        for item in items
    )
    return (
        f"✅ *Order Confirmed!*\n\n"
        f"{lines}\n\n"
        "Your order has been placed. Thank you! 🛍️"
    )


EDIT_HELP = (
    "✏️ *Edit Mode*\n\n"
    "Send your changes, for example:\n"
    "• *Add:* _add Butter - 500g_\n"
    "• *Update:* _Sugar - 2kg_ or _update 2 Sugar - 2kg_\n"
    "• *Delete:* _remove Milk_ or _delete 3_\n\n"
    "You can send multiple changes at once.\n"
    "Reply *done* when finished."
)


# ─────────────────────────────────────────────────────────────────────────────
# WEBHOOK
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/webhook", methods=["POST"])
def webhook():
    sender    = request.form.get("From", "")
    body      = request.form.get("Body", "").strip()
    num_media = int(request.form.get("NumMedia", 0))
    media_url = request.form.get("MediaUrl0", "")
    media_ct  = request.form.get("MediaContentType0", "")   # e.g. "image/jpeg" or "audio/ogg"
    session   = sessions.get(sender, {"state": "idle", "items": []})

    # ── MEDIA RECEIVED ────────────────────────────────────────────────────────
    if num_media > 0 and media_url:

        # ── Photo → OCR + parse ───────────────────────────────────────────────
        if media_ct.startswith("image/"):
            send_whatsapp(sender, "📸 Got your photo! Reading the list... ⏱️")
            try:
                items = ocr_and_parse(media_url)
                if not items:
                    send_whatsapp(sender, "😕 I couldn't find any items. Please send a clearer photo.")
                    return "", 204

                sessions[sender] = {"state": "awaiting_confirmation", "items": items}
                send_whatsapp(sender, format_list_message(items))

            except Exception as e:
                app.logger.error(f"OCR/parse error: {e}")
                send_whatsapp(sender, "❌ Something went wrong reading your photo. Please try again.")

        # ── Voice note → Whisper transcribe + parse ───────────────────────────
        elif media_ct.startswith("audio/"):
            send_whatsapp(sender, "🎙️ Got your voice note! Transcribing... ⏱️")
            try:
                transcript = transcribe_audio(media_url)
                app.logger.info(f"Whisper transcript for {sender}: {transcript}")

                items = parse_items_from_text(transcript)
                if not items:
                    send_whatsapp(
                        sender,
                        f"😕 I heard: _{transcript}_\n\nBut couldn't find any grocery items. "
                        "Try saying your list more clearly."
                    )
                    return "", 204

                sessions[sender] = {"state": "awaiting_confirmation", "items": items}
                send_whatsapp(
                    sender,
                    f"🎙️ I heard: _{transcript}_\n\n" + format_list_message(items)
                )

            except Exception as e:
                app.logger.error(f"Voice/transcription error: {e}")
                send_whatsapp(sender, "❌ Something went wrong with your voice note. Please try again.")

        # ── Unsupported media type ─────────────────────────────────────────────
        else:
            send_whatsapp(
                sender,
                "🤔 I can only read *photos* (of your grocery list) or *voice notes*. "
                "Please send one of those!"
            )

        return "", 204

    # ── TEXT REPLY ────────────────────────────────────────────────────────────
    state = session.get("state", "idle")
    items = session.get("items", [])

    # Confirm
    if state == "awaiting_confirmation" and body == "1":
        dispatch_order(items, sender)
        sessions.pop(sender, None)
        send_whatsapp(sender, format_confirmed_order(items))

    # Enter edit mode
    elif state == "awaiting_confirmation" and body == "2":
        sessions[sender] = {"state": "edit_mode", "items": items}
        send_whatsapp(sender, EDIT_HELP)

    # Exit edit mode — show current list for final review
    elif state == "edit_mode" and body.lower() in ("done", "finish", "finished"):
        sessions[sender] = {"state": "awaiting_confirmation", "items": items}
        send_whatsapp(sender, format_list_message(items))

    # Apply edits
    elif state == "edit_mode":
        try:
            updated_items = apply_edits(items, body)
            sessions[sender] = {"state": "edit_mode", "items": updated_items}
            # Show updated list and stay in edit mode for further changes
            lines = "\n".join(
                f"{i+1}. {item['name']}" + (f" — {item['quantity']}" if item.get("quantity") else "")
                for i, item in enumerate(updated_items)
            )
            send_whatsapp(
                sender,
                f"✅ Updated! Current list:\n\n{lines}\n\n"
                "Keep editing, or reply *done* to confirm."
            )
        except Exception as e:
            app.logger.error(f"Edit error: {e}")
            send_whatsapp(sender, "❌ Couldn't apply that edit. Please try again.")

    # Unknown / idle
    else:
        send_whatsapp(
            sender,
            "👋 Hi! Send me a *photo* or a *voice note* of your grocery list to get started!"
        )

    return "", 204


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)