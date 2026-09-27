"""Unit tests for the deterministic web-form surface helpers and the ROT13 guard.

Regression context (logs/challenge_3a8bfc5c): an agent posted SSTI payloads to a
field name it invented ("message") while the target's form exposed "content".
The app redirected regardless (302 -> /announce) and rendered an empty
announcement, so every injection was a silent no-op that FORGE recorded as a
failed technique. These tests pin the two things that would have caught it:
the real field name read off the markup, and the fact that a self-checking
probe ({{7*7}} -> 49) never rendered.

Pure and DB-free: no challenge rows are created, so production forge.db is
never touched.
"""

import unittest
import codecs
import os

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.recon.web_forms import describe_form, extract_forms, verify_template_probe
from backend.agents.swarm_state import SwarmBlackboard
from backend.agents.swarm_helpers import _decode_artifacts, _english_score

# ── Real fixtures, captured verbatim from the SSTI1 incident ──────────────────
# The captured page (log line 22). Note name="content" — the field the agent
# never used.
SSTI1_PAGE = """                <!doctype html>
                <title>SSTI1</title>

                <h1> Home </h1>

                <p> I built a cool website that lets you announce whatever you want!* </p>

                <form action="/" method="POST">
                What do you want to announce: <input name="content" id="announce">
                <button> Ok </button>
                </form>"""

# The 302 body agent_3 got back from every POST (log line 39).
REDIRECT_BODY = """<!doctype html>
<html lang=en>
<title>Redirecting...</title>
<h1>Redirecting...</h1>
<p>You should be redirected automatically to the target URL: <a href="/announce">/announce</a>. If not, click the link.
"""

# The empty announcement rendered for the unknown field (log line 55).
EMPTY_ANNOUNCE_BODY = """
                <!doctype html>
                <h1 style="font-size:100px;" align="center"></h1>...
"""

# The exact ROT13 hint the existing swarm test relies on.
ROT13_HINT = 'ABGR: Wnpx - grzcbenel olcnff: hfr urnqre "K-Qri-Npprff: lrf"'

class TestExtractForms(unittest.TestCase):
    """The field names an agent may legitimately post to."""

    def test_ssti1_form_exposes_content_not_a_guess(self):
        """Regression: the exact markup from the incident must yield 'content'."""
        forms = extract_forms(SSTI1_PAGE, base_url="http://rescued-float.picoctf.net:51800/")
        self.assertEqual(len(forms), 1)
        form = forms[0]
        self.assertEqual(form["method"], "POST")
        self.assertEqual(form["action"], "http://rescued-float.picoctf.net:51800/")
        names = [f["name"] for f in form["fields"]]
        self.assertIn("content", names)
        self.assertNotIn("message", names)
        content = next(f for f in form["fields"] if f["name"] == "content")
        self.assertEqual(content["id"], "announce")

    def test_relative_action_resolved_against_base(self):
        forms = extract_forms('<form action="/announce" method="post"><input name="msg"></form>',
                              base_url="http://host:8080/page")
        self.assertEqual(forms[0]["action"], "http://host:8080/announce")
        self.assertEqual(forms[0]["method"], "POST")

    def test_absolute_action_is_preserved(self):
        forms = extract_forms('<form action="https://other.example/submit"><input name="q"></form>',
                              base_url="http://host/")
        self.assertEqual(forms[0]["action"], "https://other.example/submit")

    def test_missing_action_falls_back_to_page_url(self):
        """Per the HTML spec an absent action submits to the current URL."""
        forms = extract_forms('<form method="POST"><input name="q"></form>', base_url="http://host/form")
        self.assertEqual(forms[0]["action"], "http://host/form")

    def test_default_method_is_get(self):
        forms = extract_forms('<form action="/x"><input name="q"></form>', base_url="http://host/")
        self.assertEqual(forms[0]["method"], "GET")

    def test_textarea_select_and_hidden_fields_are_collected(self):
        html = ('<form action="/x" method="POST">'
                '<input type="hidden" name="csrf" value="t">'
                '<textarea name="body"></textarea>'
                '<select name="kind"><option>a</option></select>'
                '</form>')
        forms = extract_forms(html, base_url="http://host/")
        types = {f["name"]: f["type"] for f in forms[0]["fields"]}
        self.assertEqual(types, {"csrf": "hidden", "body": "textarea", "kind": "select"})

    def test_unnamed_fields_are_skipped(self):
        """HTML does not submit a nameless control; reporting one would be a guess."""
        html = '<form action="/x"><input type="submit"><input id="only" name=""></form>'
        # The form has no usable field at all -> dropped entirely.
        self.assertEqual(extract_forms(html, base_url="http://host/"), [])

    def test_inputs_outside_a_form_are_ignored(self):
        """A bare <input> is not submittable, so it must not become a suggested field."""
        self.assertEqual(extract_forms('<input name="search" id="q">', base_url="http://host/"), [])

    def test_multiple_forms_are_all_returned(self):
        html = ('<form action="/login" method="POST"><input name="user"></form>'
                '<form action="/search" method="GET"><input name="q"></form>')
        forms = extract_forms(html, base_url="http://host/")
        self.assertEqual([f["action"] for f in forms], ["http://host/login", "http://host/search"])

    def test_malformed_markup_never_raises(self):
        for broken in (
            '<form action="/x" method="POST"><input name="a"',        # truncated
            '<form><input name="a"></div></form>',                    # mismatched close
            '<form action="/x"><input name="a" value="<b>"></form>',  # markup in attr
            '<FORM ACTION="/X" METHOD="POST"><INPUT NAME="A"></FORM>', # uppercase
        ):
            self.assertIsInstance(extract_forms(broken, base_url="http://host/"), list)

    def test_non_html_input_is_empty(self):
        for text in ("", None, "uid=0(root) gid=0(root) groups=0(root)",
                     "@@@@ #### %%%% ^^^^", "nmap scan report for 10.10.14.23"):
            self.assertEqual(extract_forms(text, base_url="http://host/"), [])

class TestDescribeForm(unittest.TestCase):
    """The one line that reaches the challenge log and every agent prompt."""

    def test_renders_method_action_and_field_names(self):
        forms = extract_forms(SSTI1_PAGE, base_url="http://rescued-float.picoctf.net:51800/")
        line = describe_form(forms[0])
        self.assertIn("POST", line)
        self.assertIn("http://rescued-float.picoctf.net:51800/", line)
        self.assertIn("content", line)
        self.assertIn("#announce", line)

    def test_caps_field_listing(self):
        fields = [{"name": f"f{i}", "type": "text", "id": ""} for i in range(12)]
        line = describe_form({"action": "/x", "method": "POST", "fields": fields})
        self.assertIn("+4 more", line)

class TestVerifyTemplateProbe(unittest.TestCase):
    """Did the self-checking payload actually render?"""

    def test_delivered_when_result_renders(self):
        probe = verify_template_probe('curl -s -X POST http://h/ -d "message={{7*7}}"',
                                      "<h1>49</h1>")
        self.assertIsNotNone(probe)
        self.assertTrue(probe["delivered"])
        self.assertEqual(probe["expected"], 49)

    def test_empty_announcement_is_not_delivered(self):
        """The exact SSTI1 signature: 200 OK, empty <h1>, payload gone."""
        probe = verify_template_probe('curl -s -X POST http://h/ -d "message={{7*7}}"',
                                      EMPTY_ANNOUNCE_BODY)
        self.assertIsNotNone(probe)
        self.assertFalse(probe["delivered"])
        self.assertEqual(probe["expected"], 49)

    def test_redirect_body_is_not_delivered(self):
        """A 302 to /announce says nothing about the sink — not a failed technique."""
        probe = verify_template_probe('curl -s -X POST http://h/ -d "message={{7*7}}"',
                                      REDIRECT_BODY)
        self.assertIsNotNone(probe)
        self.assertFalse(probe["delivered"])

    def test_reflected_verbatim_is_not_delivered(self):
        probe = verify_template_probe('curl -s "http://h/?q={{7*7}}"', "<p>you said {{7*7}}</p>")
        self.assertIsNotNone(probe)
        self.assertFalse(probe["delivered"])
        self.assertIn("verbatim", probe["reason"])

    def test_dollar_brace_probe_variant(self):
        probe = verify_template_probe('curl -s "http://h/?q=${6*7}"', "<p>42</p>")
        self.assertIsNotNone(probe)
        self.assertTrue(probe["delivered"])
        self.assertEqual(probe["expected"], 42)

    def test_incidental_digits_do_not_count_as_delivered(self):
        """'49' inside '1499' is not a rendered template."""
        probe = verify_template_probe('curl -s "http://h/?q={{7*7}}"', "<p>1499 items</p>")
        self.assertIsNotNone(probe)
        self.assertFalse(probe["delivered"])

    def test_division_probes_are_not_treated_as_self_checking(self):
        """A zero-division probe can legibly fail inside the target; do not judge it."""
        self.assertIsNone(verify_template_probe('curl -s "http://h/?q={{7/0}}"', "<h1></h1>"))

    def test_no_probe_and_no_output_yield_none(self):
        self.assertIsNone(verify_template_probe("curl -s http://h/", SSTI1_PAGE))
        self.assertIsNone(verify_template_probe('curl -s "http://h/?q={{7*7}}"', ""))
        self.assertIsNone(verify_template_probe("", "49"))

    def test_undelivered_probe_outranks_one_that_worked(self):
        """With several probes the actionable (undelivered) one is what gets reported."""
        rendered = verify_template_probe('curl -s "http://h/?a={{7*7}}&b={{8*8}}"', "<p>49 and 64</p>")
        self.assertTrue(rendered["delivered"])
        self.assertEqual(rendered["payload"], "{{7*7}}")

        partly = verify_template_probe('curl -s "http://h/?a={{7*7}}&b={{8*8}}"', "<p>49 only</p>")
        self.assertFalse(partly["delivered"])
        self.assertEqual(partly["payload"], "{{8*8}}")

class TestRot13Guard(unittest.TestCase):
    """ROT13 'decoding' of already-plain text is noise, not a finding."""

    def test_real_rot13_hint_still_decodes(self):
        """The existing swarm test's positive case must keep working."""
        decodes = _decode_artifacts(ROT13_HINT)
        self.assertTrue(any(d["scheme"] == "rot13" for d in decodes), f"lost the real decode: {decodes}")

    def test_plain_english_is_not_rot13(self):
        self.assertEqual([d for d in _decode_artifacts(
            'NOTE: Jack - temporary bypass: use header "X-Dev-Access: yes"') if d["scheme"] == "rot13"], [])

    def test_ssti1_page_is_not_rot13(self):
        """The 8 bogus decodes in the incident log all started here."""
        self.assertEqual([d for d in _decode_artifacts(SSTI1_PAGE) if d["scheme"] == "rot13"], [])

    def test_redirect_body_is_not_rot13(self):
        self.assertEqual([d for d in _decode_artifacts(REDIRECT_BODY) if d["scheme"] == "rot13"], [])

    def test_json_and_status_line_are_not_rot13(self):
        for text in ('{"user": "admin", "token": "abc"}', "GET /announce HTTP/1.1 302 Found"):
            self.assertEqual([d for d in _decode_artifacts(text) if d["scheme"] == "rot13"], [], text)

    def test_flag_string_is_not_rot13(self):
        self.assertEqual([d for d in _decode_artifacts(
            "picoCTF{s4rv3r_s1d3_t3mp14t3_1nj3ct10n5}") if d["scheme"] == "rot13"], [])

    def test_english_scorer_ranks_prose_above_its_rotation(self):
        plain = 'NOTE: Jack - temporary bypass: use header "X-Dev-Access: yes"'
        rotated = codecs.decode(plain, "rot_13")
        self.assertGreater(_english_score(plain), _english_score(rotated))
        # ...and the direction reverses for genuinely encoded text.
        self.assertGreater(_english_score(plain), _english_score(ROT13_HINT))

    def test_scorer_is_zero_without_letters(self):
        self.assertEqual(_english_score("1234 !@#$"), 0.0)

    def test_base64_branch_is_untouched(self):
        """Widening the ROT13 guard must not disable the other schemes."""
        decodes = _decode_artifacts("Q29uZ3JhdHVsYXRpb25zISBZb3UgZm91bmQgdGhlIHNlY3JldC4=")
        self.assertTrue(any(d["scheme"] == "base64" for d in decodes), f"base64 regressed: {decodes}")

class TestFormsInSharedContext(unittest.TestCase):
    """Every agent must see the real field names, not just the one that found them."""

    def test_observed_forms_reach_the_shared_prompt(self):
        board = SwarmBlackboard("t_ch", "t_run", "http://rescued-float.picoctf.net:51800/")
        board.observed_forms.extend(
            extract_forms(SSTI1_PAGE, base_url=board.target_scope))

        context = board.build_history_context("agent_2")
        self.assertIn("content", context)
        self.assertIn("USE THESE EXACT FIELD NAMES", context)

    def test_no_form_line_when_nothing_observed(self):
        board = SwarmBlackboard("t_ch", "t_run", "http://host/")
        self.assertNotIn("Known web forms", board.build_history_context("agent_1"))

if __name__ == "__main__":
    unittest.main()
