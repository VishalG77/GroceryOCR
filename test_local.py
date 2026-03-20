"""
Local function tests for GroceryOCR app.py
These tests mock OpenAI and Twilio so you can run them WITHOUT any API keys.

Run with:
    python test_local.py
"""
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Patch environment variables BEFORE importing app
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("TWILIO_ACCOUNT_SID", "ACtest")
os.environ.setdefault("TWILIO_AUTH_TOKEN",  "test_token")
os.environ.setdefault("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")

# Patch Twilio and OpenAI clients before import
with patch("twilio.rest.Client"), patch("openai.OpenAI"):
    import app as grocery_app


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_openai_text(content: str):
    """Return a mock openai ChatCompletion response with the given content."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


SAMPLE_ITEMS = [
    {"name": "Milk",   "quantity": "2L"},
    {"name": "Eggs",   "quantity": "12"},
    {"name": "Bread",  "quantity": ""},
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. parse_items_from_text
# ─────────────────────────────────────────────────────────────────────────────
class TestParseItemsFromText(unittest.TestCase):

    def test_returns_list_of_dicts(self):
        """parse_items_from_text should return a list of {name, quantity} dicts."""
        expected = [{"name": "Sugar", "quantity": "1kg"}, {"name": "Oil", "quantity": ""}]
        grocery_app.openai_client.chat.completions.create = MagicMock(
            return_value=_mock_openai_text(json.dumps(expected))
        )

        result = grocery_app.parse_items_from_text("Sugar 1kg and some Oil")

        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["name"], "Sugar")
        self.assertEqual(result[0]["quantity"], "1kg")
        self.assertEqual(result[1]["name"], "Oil")

    def test_strips_markdown_fences(self):
        """parse_items_from_text should handle ```json ... ``` wrapping."""
        payload = [{"name": "Butter", "quantity": "500g"}]
        grocery_app.openai_client.chat.completions.create = MagicMock(
            return_value=_mock_openai_text(f"```json\n{json.dumps(payload)}\n```")
        )

        result = grocery_app.parse_items_from_text("Butter 500g")
        self.assertEqual(result[0]["name"], "Butter")


# ─────────────────────────────────────────────────────────────────────────────
# 2. apply_edits
# ─────────────────────────────────────────────────────────────────────────────
class TestApplyEdits(unittest.TestCase):

    def test_add_item(self):
        """Applying 'add Butter - 500g' should append Butter to the list."""
        updated = SAMPLE_ITEMS + [{"name": "Butter", "quantity": "500g"}]
        grocery_app.openai_client.chat.completions.create = MagicMock(
            return_value=_mock_openai_text(json.dumps(updated))
        )

        result = grocery_app.apply_edits(SAMPLE_ITEMS, "add Butter - 500g")
        self.assertEqual(len(result), 4)
        self.assertEqual(result[-1]["name"], "Butter")

    def test_delete_item_by_number(self):
        """Applying 'delete 2' should remove item at index 1 (Eggs)."""
        updated = [{"name": "Milk", "quantity": "2L"}, {"name": "Bread", "quantity": ""}]
        grocery_app.openai_client.chat.completions.create = MagicMock(
            return_value=_mock_openai_text(json.dumps(updated))
        )

        result = grocery_app.apply_edits(SAMPLE_ITEMS, "delete 2")
        self.assertEqual(len(result), 2)
        names = [i["name"] for i in result]
        self.assertNotIn("Eggs", names)

    def test_update_item(self):
        """Applying 'Milk - 1L' should update Milk's quantity to 1L."""
        updated = [
            {"name": "Milk", "quantity": "1L"},
            {"name": "Eggs", "quantity": "12"},
            {"name": "Bread", "quantity": ""},
        ]
        grocery_app.openai_client.chat.completions.create = MagicMock(
            return_value=_mock_openai_text(json.dumps(updated))
        )

        result = grocery_app.apply_edits(SAMPLE_ITEMS, "Milk - 1L")
        milk = next(i for i in result if i["name"] == "Milk")
        self.assertEqual(milk["quantity"], "1L")


# ─────────────────────────────────────────────────────────────────────────────
# 3. format_list_message
# ─────────────────────────────────────────────────────────────────────────────
class TestFormatListMessage(unittest.TestCase):

    def test_contains_item_count(self):
        msg = grocery_app.format_list_message(SAMPLE_ITEMS)
        self.assertIn("3 item(s)", msg)

    def test_items_are_numbered(self):
        msg = grocery_app.format_list_message(SAMPLE_ITEMS)
        self.assertIn("1. Milk", msg)
        self.assertIn("2. Eggs", msg)
        self.assertIn("3. Bread", msg)

    def test_quantity_shown_when_present(self):
        msg = grocery_app.format_list_message(SAMPLE_ITEMS)
        self.assertIn("2L", msg)

    def test_quantity_hidden_when_empty(self):
        items = [{"name": "Bread", "quantity": ""}]
        msg = grocery_app.format_list_message(items)
        # Should show "1. Bread" without a dash after it
        self.assertIn("1. Bread", msg)
        self.assertNotIn("Bread —", msg)

    def test_confirm_and_edit_prompts_present(self):
        msg = grocery_app.format_list_message(SAMPLE_ITEMS)
        self.assertIn("1", msg)   # confirm
        self.assertIn("2", msg)   # edit


# ─────────────────────────────────────────────────────────────────────────────
# 4. format_confirmed_order
# ─────────────────────────────────────────────────────────────────────────────
class TestFormatConfirmedOrder(unittest.TestCase):

    def test_contains_all_items(self):
        msg = grocery_app.format_confirmed_order(SAMPLE_ITEMS)
        self.assertIn("Milk", msg)
        self.assertIn("Eggs", msg)
        self.assertIn("Bread", msg)

    def test_confirmed_label_present(self):
        msg = grocery_app.format_confirmed_order(SAMPLE_ITEMS)
        self.assertIn("Confirmed", msg)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Webhook state machine (no real HTTP, no real Twilio)
# ─────────────────────────────────────────────────────────────────────────────
class TestWebhookStateMachine(unittest.TestCase):

    def setUp(self):
        grocery_app.sessions.clear()
        grocery_app.app.config["TESTING"] = True
        self.client = grocery_app.app.test_client()
        # Silence outgoing WhatsApp messages
        grocery_app.send_whatsapp = MagicMock()

    def _post(self, **form_data):
        defaults = {"From": "whatsapp:+19999999999", "Body": "", "NumMedia": "0"}
        defaults.update(form_data)
        return self.client.post("/webhook", data=defaults)

    def test_idle_greeting(self):
        """Unknown text on idle session → greeting."""
        self._post(Body="hello")
        grocery_app.send_whatsapp.assert_called_once()
        msg = grocery_app.send_whatsapp.call_args[0][1]
        self.assertIn("photo", msg.lower())

    def test_confirm_clears_session(self):
        """Reply '1' in awaiting_confirmation → order confirmed, session cleared."""
        sender = "whatsapp:+19999999999"
        grocery_app.sessions[sender] = {"state": "awaiting_confirmation", "items": SAMPLE_ITEMS}
        grocery_app.dispatch_order = MagicMock()   # stub order dispatch

        self._post(Body="1")

        self.assertNotIn(sender, grocery_app.sessions)
        grocery_app.dispatch_order.assert_called_once()

    def test_enter_edit_mode(self):
        """Reply '2' in awaiting_confirmation → state becomes edit_mode."""
        sender = "whatsapp:+19999999999"
        grocery_app.sessions[sender] = {"state": "awaiting_confirmation", "items": SAMPLE_ITEMS}

        self._post(Body="2")

        self.assertEqual(grocery_app.sessions[sender]["state"], "edit_mode")

    def test_edit_mode_done_returns_to_confirmation(self):
        """Reply 'done' in edit_mode → state returns to awaiting_confirmation."""
        sender = "whatsapp:+19999999999"
        grocery_app.sessions[sender] = {"state": "edit_mode", "items": SAMPLE_ITEMS}

        self._post(Body="done")

        self.assertEqual(grocery_app.sessions[sender]["state"], "awaiting_confirmation")

    def test_edit_applies_changes(self):
        """Edit command in edit_mode → apply_edits called, list updated."""
        sender = "whatsapp:+19999999999"
        grocery_app.sessions[sender] = {"state": "edit_mode", "items": SAMPLE_ITEMS}

        updated = SAMPLE_ITEMS + [{"name": "Butter", "quantity": "500g"}]
        grocery_app.apply_edits = MagicMock(return_value=updated)

        self._post(Body="add Butter - 500g")

        grocery_app.apply_edits.assert_called_once()
        self.assertEqual(len(grocery_app.sessions[sender]["items"]), 4)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
