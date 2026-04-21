import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional, List
import logging
from types import SimpleNamespace

from .token_tracker import TokenUsageTracker


class TestTokenTracker(unittest.TestCase):
    """Test class for TokenUsageTracker utility"""

    def setUp(self):
        """Set up test environment before each test"""
        # Create a token tracker instance
        self.tracker = TokenUsageTracker()

        # Mock logger to test logging
        self.mock_logger = MagicMock()
        logging.getLogger("token_tracker").setLevel(logging.INFO)

    def test_track_usage_direct_usage(self):
        """Test tracking token usage from a result object with direct usage attribute"""
        # Create a mock result with direct usage attribute
        mock_result = MagicMock()
        mock_result.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        # Track usage
        self.tracker.track_usage("test_agent", mock_result, "gpt-4")

        # Verify token usage was tracked
        agent_usage = self.tracker.get_agent_usage("test_agent")
        self.assertEqual(agent_usage["prompt_tokens"], 100)
        self.assertEqual(agent_usage["completion_tokens"], 50)
        self.assertEqual(agent_usage["total_tokens"], 150)

    def test_track_usage_with_model_dump(self):
        """Test tracking token usage from a result object with model_dump method"""
        usage_obj = SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            }
        )
        mock_result = SimpleNamespace(usage=usage_obj)

        # Track usage
        self.tracker.track_usage("test_model_dump_agent", mock_result, "gpt-3.5-turbo")

        # Verify token usage was tracked
        agent_usage = self.tracker.get_agent_usage("test_model_dump_agent")
        self.assertEqual(agent_usage["prompt_tokens"], 200)
        self.assertEqual(agent_usage["completion_tokens"], 100)
        self.assertEqual(agent_usage["total_tokens"], 300)

    def test_track_usage_from_raw_responses(self):
        """Test tracking token usage from raw_responses array"""
        mock_usage = SimpleNamespace(input_tokens=300, output_tokens=150, total_tokens=450)
        mock_response = SimpleNamespace(usage=mock_usage)
        mock_result = SimpleNamespace(raw_responses=[mock_response])

        # Track usage
        self.tracker.track_usage("test_raw_responses_agent", mock_result, "azure-gpt-4")

        # Verify token usage was tracked
        agent_usage = self.tracker.get_agent_usage("test_raw_responses_agent")
        self.assertEqual(agent_usage["prompt_tokens"], 300)
        self.assertEqual(agent_usage["completion_tokens"], 150)
        self.assertEqual(agent_usage["total_tokens"], 450)

    def test_track_usage_from_raw_responses_direct(self):
        """Test the _track_usage_from_raw_responses method directly"""
        # Create first response with standard format
        mock_usage1 = SimpleNamespace(
            prompt_tokens=200, completion_tokens=100, total_tokens=300
        )
        mock_response1 = SimpleNamespace(usage=mock_usage1)

        # Create second response with Azure format (input/output tokens)
        mock_usage2 = SimpleNamespace(input_tokens=300, output_tokens=150)
        mock_response2 = SimpleNamespace(usage=mock_usage2)

        mock_result = SimpleNamespace(raw_responses=[mock_response1])

        # Standard OpenAI usage objects are not handled by this Azure-specific helper.
        mock_result.raw_responses = [mock_response1]
        token_info = self.tracker._track_usage_from_raw_responses(mock_result)
        self.assertIsNone(token_info)

        # Test with Azure OpenAI format
        mock_result.raw_responses = [mock_response2]
        token_info = self.tracker._track_usage_from_raw_responses(mock_result)
        self.assertEqual(token_info["prompt_tokens"], 300)
        self.assertEqual(token_info["completion_tokens"], 150)
        self.assertEqual(token_info["total_tokens"], 450)

        # Test with string representation parsing
        mock_usage3 = (
            "Usage(requests=1, input_tokens=500, output_tokens=250, total_tokens=750)"
        )
        mock_response3 = MagicMock()
        mock_response3.usage = mock_usage3

        mock_result.raw_responses = [mock_response3]
        token_info = self.tracker._track_usage_from_raw_responses(mock_result)
        self.assertEqual(token_info["prompt_tokens"], 500)
        self.assertEqual(token_info["completion_tokens"], 250)
        self.assertEqual(token_info["total_tokens"], 750)

    def test_create_synthetic_usage(self):
        """Test creating synthetic token usage"""
        # Mock result object for synthetic usage
        mock_result = MagicMock()

        # Test with provided content
        content = "This is a test content" * 10  # 20 chars * 10 = 200 chars
        token_info = self.tracker.create_synthetic_usage("test_agent", mock_result)

        # Verify synthetic usage was created
        self.assertEqual(token_info["prompt_tokens"], 500)
        self.assertTrue(
            token_info["completion_tokens"] > 0
        )  # Should have some completion tokens
        self.assertTrue(token_info["synthetic"])  # Should be marked as synthetic

        # Test with None content
        token_info = self.tracker.create_synthetic_usage("test_agent", mock_result)
        self.assertEqual(token_info["prompt_tokens"], 500)
        self.assertTrue(token_info["completion_tokens"] >= 1)
        self.assertTrue(token_info["total_tokens"] >= 501)
        self.assertTrue(token_info["synthetic"])

    def test_get_total_usage(self):
        """Test getting total usage statistics"""
        # Set up some usage data
        mock_result1 = MagicMock()
        mock_result1.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        mock_result2 = MagicMock()
        mock_result2.usage = {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        }

        # Track usage for different agents
        self.tracker.track_usage("agent1", mock_result1, "gpt-4")
        self.tracker.track_usage("agent2", mock_result2, "gpt-3.5-turbo")

        # Get total usage
        total_usage = self.tracker.get_total_usage()

        # Verify totals
        self.assertEqual(total_usage["prompt_tokens"], 300)
        self.assertEqual(total_usage["completion_tokens"], 150)
        self.assertEqual(total_usage["total_tokens"], 450)

    def test_estimate_cost(self):
        """Test estimating cost based on token usage"""
        # Set up some usage data
        mock_result = MagicMock()
        mock_result.usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        }

        # Track usage
        self.tracker.track_usage("cost_test_agent", mock_result, "gpt-4")

        # Estimate cost
        cost_data = self.tracker.estimate_cost("gpt-4")

        # Verify cost was calculated (we don't check specific values as they depend on internal pricing)
        self.assertIn("total_cost", cost_data)
        self.assertTrue(isinstance(cost_data["total_cost"], float))
        self.assertTrue(cost_data["total_cost"] > 0)


if __name__ == "__main__":
    unittest.main()
