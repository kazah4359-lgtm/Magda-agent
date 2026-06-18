import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import sys
from magda_agent.cli import main

class TestCLI(unittest.TestCase):
    @patch('magda_agent.api.consciousness')
    @patch('sys.argv', ['magda', 'status'])
    @patch('builtins.print')
    def test_status(self, mock_print, mock_c):
        mock_c.get_internal_state.return_value = "Status: OK"
        main()
        mock_print.assert_any_call("Status: OK")

    @patch('magda_agent.api.memory_system')
    @patch('sys.argv', ['magda', 'memory', 'list'])
    @patch('builtins.print')
    def test_memory_list(self, mock_print, mock_mem):
        mock_entry = MagicMock()
        mock_entry.content = "Test memory"
        type(mock_mem).short_term = PropertyMock(return_value=[mock_entry])

        main()
        mock_print.assert_any_call("- Test memory")

    @patch('magda_agent.api.memory_system')
    @patch('sys.argv', ['magda', 'memory', 'clear'])
    @patch('builtins.print')
    def test_memory_clear(self, mock_print, mock_mem):
        main()
        mock_mem.working_memory.clear.assert_called_once()
        mock_print.assert_any_call("Memory cleared.")
