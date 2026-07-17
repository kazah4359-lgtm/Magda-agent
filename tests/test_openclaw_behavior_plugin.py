import unittest
from unittest.mock import MagicMock

from magda_agent.learning.openclaw_behavior_plugin import OpenClawBehaviorPlugin
from magda_agent.learning.openclaw_rl_rollout import OpenClawRLTrajectoryRolloutV6


class TestOpenClawBehaviorPlugin(unittest.TestCase):
    def setUp(self):
        self.mock_rollout = MagicMock(spec=OpenClawRLTrajectoryRolloutV6)
        self.plugin = OpenClawBehaviorPlugin(rollout=self.mock_rollout)
        self.user_id = 123

    def test_before_write_dict_context(self):
        context = {"skill_id": "test_skill", "content": "test_content"}
        result = self.plugin.before_write(context, self.user_id)

        self.assertEqual(result, context)
        self.mock_rollout.record_step.assert_called_once_with(str(self.user_id), "test_skill", "test_content")

    def test_before_write_object_context(self):
        class DummyContext:
            skill_id = "obj_skill"
            content = "obj_content"

        context = DummyContext()
        result = self.plugin.before_write(context, self.user_id)

        self.assertEqual(result, context)
        self.mock_rollout.record_step.assert_called_once_with(str(self.user_id), "obj_skill", "obj_content")

    def test_before_write_string_context(self):
        context = "just some string"
        result = self.plugin.before_write(context, self.user_id)

        self.assertEqual(result, context)
        self.mock_rollout.record_step.assert_called_once_with(str(self.user_id), "unknown_skill", "just some string")

    def test_on_context_update_dict_with_reward(self):
        context = {"reward": 0.75}
        self.plugin.on_context_update(context, self.user_id)

        self.mock_rollout.process_delayed_reward.assert_called_once_with(str(self.user_id), 0.75)

    def test_on_context_update_object_with_reward(self):
        class DummyContext:
            reward = 0.5

        context = DummyContext()
        self.plugin.on_context_update(context, self.user_id)

        self.mock_rollout.process_delayed_reward.assert_called_once_with(str(self.user_id), 0.5)

    def test_on_context_update_no_reward(self):
        context = {"other_key": "value"}
        self.plugin.on_context_update(context, self.user_id)

        self.mock_rollout.process_delayed_reward.assert_not_called()

    def test_on_context_update_invalid_reward(self):
        context = {"reward": "not_a_number"}
        self.plugin.on_context_update(context, self.user_id)

        self.mock_rollout.process_delayed_reward.assert_not_called()

if __name__ == '__main__':
    unittest.main()
