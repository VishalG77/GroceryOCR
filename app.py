import os
import json
import requests
from flask import Flask, request
from twilio.rest import Client
from twilio.request_validator import RequestValidator
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
sessions: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# OCR + PARSE  (GPT-4o Vision — one call does both)
# ─────────────────────────────────────────────────────────────────────────────

def ocr_and_parse(image_url: str) -> list[dict]:
    """Use GPT-4o Vision to OCR the image AND return structured grocery items."""
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
                    {"type": "image_url", "image_url": {"url": image_url}},
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
# EDIT PARSING  (GPT-4o)
# ─────────────────────────────────────────────────────────────────────────────

def apply_edits(items: list[dict], edit_message: str) -> list[dict]:
    """
    Parse an edit message and apply changes to the item list.
    Supports:
      - Update:  "Sugar - 1kg"  or  "Sugar: 1kg"
      - Remove:  "Remove Sugar"
    """
    prompt = (
        f"Current grocery list (JSON):\n{json.dumps(items)}\n\n"
        f"User edit message:\n{edit_message}\n\n"
        "Apply the edits. Rules:\n"
        "- Lines like 'Item - qty' or 'Item: qty' → update or add that item.\n"
        "- Lines like 'Remove Item' → delete that item (case-insensitive match).\n"
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
# TWILIO HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def send_whatsapp(to: str, body: str) -> None:
    twilio_client.messages.create(from_=TWILIO_FROM, to=to, body=body)


def format_list_message(items: list[dict]) -> str:
    lines = "\n".join(
        f"{i+1}. {item['name']}" + (f" — {item['quantity']}" if item.get("quantity") else "")
        for i, item in enumerate(items)
    )
    return (
        f"🛒 I found *{len(items)} item(s)*:\n\n"
        f"{lines}\n\n"
        "Reply *1* to Confirm ✅\n"
        "Reply *2* to Edit ✏️"
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
    session   = sessions.get(sender, {"state": "idle", "items": []})

    # ── IMAGE RECEIVED → OCR + parse ─────────────────────────────────────────
    if num_media > 0 and media_url:
        send_whatsapp(sender, "Got it! Give me a second to read your list... ⏱️")
        try:
            items = ocr_and_parse(media_url)
            if not items:
                send_whatsapp(sender, "😕 I couldn't find any items. Please send a clearer photo.")
                return "", 204

            sessions[sender] = {"state": "awaiting_confirmation", "items": items}
            send_whatsapp(sender, format_list_message(items))

        except Exception as e:
            app.logger.error(f"OCR/parse error: {e}")
            send_whatsapp(sender, "❌ Something went wrong reading your list. Please try again.")

        return "", 204

    # ── TEXT REPLY ────────────────────────────────────────────────────────────
    state = session.get("state", "idle")
    items = session.get("items", [])

    # Confirm
    if state == "awaiting_confirmation" and body == "1":
        sessions.pop(sender, None)
        item_names = "\n".join(f"• {i['name']}" + (f" — {i['quantity']}" if i.get("quantity") else "") for i in items)
        send_whatsapp(sender, f"✅ Order confirmed!\n\n{item_names}\n\nThank you! 🛍️")

    # Enter edit mode
    elif state == "awaiting_confirmation" and body == "2":
        sessions[sender] = {"state": "edit_mode", "items": items}
        send_whatsapp(
            sender,
            "✏️ *Edit mode!*\n\n"
            "Send your changes, for example:\n"
            "• Update: _Sugar - 1kg_\n"
            "• Remove: _Remove Milk_\n\n"
            "You can send multiple changes at once."
        )

    # Apply edits
    elif state == "edit_mode":
        try:
            updated_items = apply_edits(items, body)
            sessions[sender] = {"state": "awaiting_confirmation", "items": updated_items}
            send_whatsapp(sender, "✅ List updated!\n\n" + format_list_message(updated_items))
        except Exception as e:
            app.logger.error(f"Edit error: {e}")
            send_whatsapp(sender, "❌ Couldn't apply edits. Please try again.")

    # Unknown / idle
    else:
        send_whatsapp(sender, "👋 Send me a photo of your grocery list to get started!")

    return "", 204


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
