# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from typing import List, Any, Dict, Tuple, Callable
import copy
import time
import re
import jinja2

from azure.ai.evaluation._http_utils import AsyncHttpPipeline
from .._types import ConversationTurn
from ..._model_tools import RAIClient
from ..._model_tools._template_handler import TemplateParameters
from ._conversation_bot_base import ConversationBot


class MultiModalConversationBot(ConversationBot):
    """MultiModal Conversation bot that uses a user provided callback to generate responses.

    :param callback: The callback function to use to generate responses.
    :type callback: Callable
    :param user_template: The template to use for the request.
    :type user_template: str
    :param user_template_parameters: The template parameters to use for the request.
    :type user_template_parameters: Dict
    :param args: Optional arguments to pass to the parent class.
    :type args: Any
    :param kwargs: Optional keyword arguments to pass to the parent class.
    :type kwargs: Any
    """

    def __init__(
        self,
        callback: Callable,
        user_template: str,
        user_template_parameters: TemplateParameters,
        rai_client: RAIClient,
        *args,
        **kwargs,
    ) -> None:
        self.callback = callback
        self.user_template = user_template
        self.user_template_parameters = user_template_parameters
        self.rai_client = rai_client

        super().__init__(*args, **kwargs)

    async def generate_response(
        self,
        session: AsyncHttpPipeline,
        conversation_history: List[Any],
        max_history: int,
        turn_number: int = 0,
    ) -> Tuple[dict, dict, float, dict]:
        previous_prompt = conversation_history[-1]
        chat_protocol_message = await self._to_chat_protocol(conversation_history, self.user_template_parameters)

        # replace prompt with {image.jpg} tags with image content data.
        conversation_history.pop()
        conversation_history.append(
            ConversationTurn(
                role=previous_prompt.role,
                name=previous_prompt.name,
                message=chat_protocol_message["messages"][0]["content"],
                full_response=previous_prompt.full_response,
                request=chat_protocol_message,
            )
        )
        msg_copy = copy.deepcopy(chat_protocol_message)
        result = {}
        start_time = time.time()
        result = await self.callback(msg_copy)
        end_time = time.time()
        if not result:
            result = {
                "messages": [{"content": "Callback did not return a response.", "role": "assistant"}],
                "finish_reason": ["stop"],
                "id": None,
                "template_parameters": {},
            }

        time_taken = end_time - start_time
        try:
            response = {
                "samples": [result["messages"][-1]["content"]],
                "finish_reason": ["stop"],
                "id": None,
            }
        except Exception as exc:
            msg = "User provided callback does not conform to chat protocol standard."
            raise EvaluationException(
                message=msg,
                internal_message=msg,
                target=ErrorTarget.CALLBACK_CONVERSATION_BOT,
                category=ErrorCategory.INVALID_VALUE,
                blame=ErrorBlame.USER_ERROR,
            ) from exc

        return response, chat_protocol_message, time_taken, result

    async def _to_chat_protocol(self, conversation_history, template_parameters):  # pylint: disable=unused-argument
        messages = []

        for _, m in enumerate(conversation_history):
            if "image:" in m.message:
                content = await self._to_multi_modal_content(m.message)
                messages.append({"content": content, "role": m.role.value})
            else:
                messages.append({"content": m.message, "role": m.role.value})

        return {
            "template_parameters": template_parameters,
            "messages": messages,
            "$schema": "http://azureml/sdk-2-0/ChatConversation.json",
        }

    async def _to_multi_modal_content(self, text: str) -> list:
        split_text = re.findall(r"[^{}]+|\{[^{}]*\}", text)
        messages = [
            text.strip("{}").replace("image:", "").strip() if text.startswith("{") else text for text in split_text
        ]
        contents = []
        for msg in messages:
            if msg.startswith("image_understanding/"):
                encoded_image = await self.rai_client.get_image_data(msg)
                contents.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                )
            else:
                contents.append({"type": "text", "text": msg})
        return contents
    
    