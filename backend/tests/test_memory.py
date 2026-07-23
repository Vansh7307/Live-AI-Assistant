import os
import tempfile
import unittest

import memory.sqlite_memory as mem_module
from memory.sqlite_memory import SQLiteMemory


class SQLiteMemoryTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self._tmpdir.name, "test_memory.sqlite3")
        os.environ["MEMORY_DB_PATH"] = db_path
        mem_module._ENGINE = None  # force a fresh engine bound to the temp db
        self.memory = SQLiteMemory()

    def tearDown(self):
        # Must dispose the engine's pooled connection(s) before the temp
        # directory cleanup runs, otherwise Windows refuses to delete a
        # file that's still open (PermissionError / WinError 32). Linux is
        # lenient about this; Windows is not - dispose explicitly so the
        # test is portable.
        if mem_module._ENGINE is not None:
            mem_module._ENGINE.dispose()
        mem_module._ENGINE = None
        os.environ.pop("MEMORY_DB_PATH", None)
        self._tmpdir.cleanup()

    def test_messages_are_isolated_per_session(self):
        # Regression test: memory used to be global with no session
        # concept, so every user shared one conversation history.
        self.memory.append_user_message("session-a", "hello from a")
        self.memory.append_assistant_message("session-a", "hi a")
        self.memory.append_user_message("session-b", "hello from b")

        history_a = self.memory.get_recent_messages("session-a")
        history_b = self.memory.get_recent_messages("session-b")

        self.assertEqual([m["content"] for m in history_a], ["hello from a", "hi a"])
        self.assertEqual([m["content"] for m in history_b], ["hello from b"])

    def test_get_recent_messages_respects_limit_and_order(self):
        for i in range(5):
            self.memory.append_user_message("session-c", f"msg-{i}")

        recent = self.memory.get_recent_messages("session-c", limit=2)
        self.assertEqual([m["content"] for m in recent], ["msg-3", "msg-4"])

    def test_unknown_session_returns_empty_history(self):
        self.assertEqual(self.memory.get_recent_messages("never-seen"), [])


if __name__ == "__main__":
    unittest.main()
