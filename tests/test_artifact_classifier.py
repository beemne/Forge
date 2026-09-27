"""Unit tests for the deterministic binary-artifact pre-classifier (Part 2).

Covers HTTP-response classification, local-file classification, magic-byte
detection, and — critically — byte-for-byte exactness of save_artifact_binary
(the guard against the text-mode decode corruption seen in the Transformation run).
"""

import hashlib
import os
import tempfile
import unittest

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.agents.artifact_classifier import (
    classify_http_response,
    classify_local_file,
    save_artifact_binary,
)

ELF_MAGIC = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 24
ZIP_MAGIC = b"\x50\x4b\x03\x04" + b"\x00" * 20

class TestHttpClassification(unittest.TestCase):
    def test_octet_stream_is_binary(self):
        r = classify_http_response("http://x/file", "application/octet-stream", "nginx", 5000, ELF_MAGIC)
        self.assertTrue(r.is_binary)
        self.assertEqual(r.artifact_type, "elf")

    def test_octet_stream_with_text_body_is_not_binary(self):
        body = b"import re\nflag = open('flag.txt').read()\n"
        r = classify_http_response("http://x/lyric-reader.py", "application/octet-stream", "nginx", len(body), body)
        self.assertFalse(r.is_binary)
        self.assertEqual(r.artifact_type, "text")

    def test_amazons3_server_is_binary(self):
        r = classify_http_response("http://bucket.example.com/a", "", "AmazonS3", 1234, ZIP_MAGIC)
        self.assertTrue(r.is_binary)
        self.assertIn("AmazonS3".lower(), r.reason.lower())

    def test_cdn_host_is_binary(self):
        r = classify_http_response("https://files.example.com/download/chall", "", "", 900, b"randombytes")
        self.assertTrue(r.is_binary)

    def test_small_nontext_blob_is_binary(self):
        r = classify_http_response("http://x/y", "application/x-unknown-thing", "", 300, b"\x89\x01\x02rawbytes")
        self.assertTrue(r.is_binary)

    def test_html_is_not_binary(self):
        r = classify_http_response("http://x/", "text/html; charset=utf-8", "nginx", 4000, b"<!DOCTYPE html><html>")
        self.assertFalse(r.is_binary)
        self.assertEqual(r.artifact_type, "text")

    def test_json_api_is_not_binary(self):
        r = classify_http_response("http://x/api", "application/json", "nginx", 40, b'{"ok":true}')
        self.assertFalse(r.is_binary)

class TestLocalFileClassification(unittest.TestCase):
    def test_elf_magic_local_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "mystery")            # no extension
            with open(p, "wb") as fh:
                fh.write(ELF_MAGIC + b"\x00" * 100)
            r = classify_local_file(p)
            self.assertTrue(r.is_binary)
            self.assertEqual(r.artifact_type, "elf")
            self.assertEqual(r.safe_file_path, p)

    def test_zip_extension(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.zip")
            with open(p, "wb") as fh:
                fh.write(ZIP_MAGIC)
            r = classify_local_file(p)
            self.assertTrue(r.is_binary)

    def test_text_source_is_not_binary(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "solve.py")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("print('hello world')\n")
            r = classify_local_file(p)
            self.assertFalse(r.is_binary)

    def test_binary_extension_with_text_content_is_not_binary(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "script.bin")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("import re\nflag = open('flag.txt').read()\n")
            r = classify_local_file(p)
            self.assertFalse(r.is_binary)
            self.assertEqual(r.artifact_type, "text")

    def test_missing_file(self):
        r = classify_local_file("/no/such/file.bin")
        self.assertFalse(r.is_binary)

class TestSaveArtifactByteExactness(unittest.TestCase):
    def test_bytes_roundtrip_sha256(self):
        # Include non-UTF-8 bytes that a text-mode decode(errors="replace") would corrupt.
        raw = bytes(range(256)) * 8 + b"\xff\xfe\x00MZ\x90\x00"
        with tempfile.TemporaryDirectory() as d:
            dest = save_artifact_binary(raw, d, "artifact.bin")
            with open(dest, "rb") as fh:
                got = fh.read()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), hashlib.sha256(got).hexdigest())
            self.assertEqual(raw, got)

    def test_no_clobber_appends_counter(self):
        raw1, raw2 = b"AAAA", b"BBBB"
        with tempfile.TemporaryDirectory() as d:
            p1 = save_artifact_binary(raw1, d, "dup.bin")
            p2 = save_artifact_binary(raw2, d, "dup.bin")
            self.assertNotEqual(p1, p2)
            with open(p1, "rb") as fh:
                self.assertEqual(fh.read(), raw1)
            with open(p2, "rb") as fh:
                self.assertEqual(fh.read(), raw2)

if __name__ == "__main__":
    unittest.main()
