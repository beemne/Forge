"""
Tests for Phase 4: Stream Condenser (Output distillation middleware)
"""
import os
import unittest

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.agents.stream_condenser import StreamCondenser

class TestStreamCondenser(unittest.TestCase):
    """Test StreamCondenser output distillation logic."""

    def test_short_output_passthrough(self):
        """Output shorter than max_lines should pass through unchanged."""
        short_output = "PORT   STATE SERVICE\n22/tcp open  ssh\n80/tcp open  http"
        result = StreamCondenser.condense_output("nmap", short_output, max_lines=25)
        self.assertEqual(result, short_output, "Short output should pass through unmodified")

    def test_gobuster_distillation(self):
        """Gobuster 2000-line output should be distilled to <=25 lines."""
        lines = []
        for i in range(2000):
            status = 404
            if i % 100 == 0:
                status = 200
            elif i % 150 == 0:
                status = 301
            elif i % 200 == 0:
                status = 403
            lines.append(f"/path_{i:04d}    (Status: {status}) [Size: {i * 10}]")

        raw_output = "\n".join(lines)
        result = StreamCondenser.condense_output("gobuster", raw_output, max_lines=25)

        result_lines = result.strip().splitlines()
        self.assertLessEqual(len(result_lines), 26, f"Should be <=26 lines (header + 25), got {len(result_lines)}")
        self.assertIn("[STREAM CONDENSER:", result_lines[0])

    def test_nmap_retains_open_ports(self):
        """Nmap distillation should retain lines with open ports and services."""
        nmap_output = "\n".join([
            "Starting Nmap scan...",
            "PORT     STATE  SERVICE  VERSION",
            "22/tcp   open   ssh      OpenSSH 8.2p1",
            "80/tcp   open   http     Apache 2.4.41",
            "443/tcp  closed https",
            "3306/tcp open   mysql    MySQL 5.7.33",
        ] + [f"{i}/tcp filtered unknown" for i in range(1000, 1100)])

        result = StreamCondenser.condense_output("nmap", nmap_output, max_lines=25)
        self.assertIn("22/tcp", result)
        self.assertIn("open", result.lower())
        self.assertIn("ssh", result.lower())

    def test_ffuf_retains_200_and_403(self):
        """ffuf distillation should retain status 200/301/403 responses."""
        ffuf_lines = []
        for i in range(500):
            if i == 42:
                ffuf_lines.append(f"/admin          [Status: 200, Size: 1234, Words: 56]")
            elif i == 99:
                ffuf_lines.append(f"/.git/HEAD      [Status: 403, Size: 289, Words: 14]")
            elif i == 150:
                ffuf_lines.append(f"/api            [Status: 301, Size: 0, Words: 0]")
            else:
                ffuf_lines.append(f"/random_{i:04d}   [Status: 404, Size: 0, Words: 0]")

        raw = "\n".join(ffuf_lines)
        result = StreamCondenser.condense_output("ffuf", raw, max_lines=25)
        self.assertIn("200", result)
        self.assertIn("403", result)

    def test_flag_always_extracted(self):
        """Stream condenser should ALWAYS extract flag patterns regardless of tool type."""
        output_lines = [f"Scanning port {i}..." for i in range(200)]
        output_lines[100] = "picoCTF{h1dd3n_1n_th3_n01s3}"
        raw = "\n".join(output_lines)

        result = StreamCondenser.condense_output("custom_tool", raw, max_lines=25)
        self.assertIn("picoCTF{h1dd3n_1n_th3_n01s3}", result)
        self.assertIn("FLAG FOUND", result)

    def test_binwalk_retains_interesting_strings(self):
        """Binwalk distillation should retain lines mentioning certificates, keys, etc."""
        binwalk_lines = [
            "DECIMAL       HEXADECIMAL     DESCRIPTION",
            "0             0x0             ELF, 64-bit LSB executable, AMD x86-64",
            "12345         0x3039          Zip archive data",
            "55555         0xD903          Certificate Authority certificate",
            "66666         0x10472         private key data",
        ] + [f"{i}         0x{i:X}         UNKNOWN_DATA" for i in range(100, 200)]

        raw = "\n".join(binwalk_lines)
        result = StreamCondenser.condense_output("binwalk", raw, max_lines=25)
        self.assertIn("ELF", result)
        self.assertIn("certificate", result.lower())

    def test_empty_output(self):
        """Empty output should return empty string."""
        result = StreamCondenser.condense_output("nmap", "")
        self.assertEqual(result, "")

    def test_none_output(self):
        """None output should return None."""
        result = StreamCondenser.condense_output("nmap", None)
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
