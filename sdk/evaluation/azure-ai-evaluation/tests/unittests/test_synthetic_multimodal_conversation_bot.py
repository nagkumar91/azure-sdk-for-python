import pytest
import copy
from unittest.mock import AsyncMock, patch
from azure.ai.evaluation.simulator._conversation import MultiModalConversationBot, ConversationRole, ConversationTurn
from azure.ai.evaluation._http_utils import AsyncHttpPipeline
from azure.ai.evaluation._model_tools import RAIClient

@pytest.mark.unittest
class TestMultiModalConversationBot:
    @pytest.mark.asyncio
    async def test_multi_modal_generate_response_no_images(self):
        bot = MultiModalConversationBot(
            callback=AsyncMock(return_value={"messages":[{"content":"some response","role":"assistant"}]}),
            user_template="",
            user_template_parameters={},
            rai_client=RAIClient("some-endpoint"),
            role=ConversationRole.ASSISTANT,
            model=AsyncMock(),
            conversation_template="",
            instantiation_parameters={},
        )
        session = AsyncHttpPipeline()
        conversation_history = [ConversationTurn(role=ConversationRole.USER, message="Hello assistant")]
        response, chat_protocol_msg, time_taken, result = await bot.generate_response(session, conversation_history, 5)
        assert response["samples"][0] == "some response"
        assert "messages" in chat_protocol_msg

    @pytest.mark.asyncio
    async def test_multi_modal_generate_response_image_in_text(self):
        bot = MultiModalConversationBot(
            callback=AsyncMock(return_value={"messages":[{"content":"image content response","role":"assistant"}]}),
            user_template="",
            user_template_parameters={},
            rai_client=RAIClient("some-endpoint"),
            role=ConversationRole.ASSISTANT,
            model=AsyncMock(),
            conversation_template="",
            instantiation_parameters={},
        )
        with patch.object(bot, "_to_multi_modal_content", return_value=[{"type": "image_url", "image_url":{"url":"data:image/png;base64,xyz"}}]):
            session = AsyncHttpPipeline()
            conversation_history = [ConversationTurn(role=ConversationRole.USER, message="Here: {image:test.jpg}")]
            response, chat_protocol_msg, time_taken, result = await bot.generate_response(session, conversation_history, 5)
            assert "image_url" in chat_protocol_msg["messages"][0]["content"][0]
