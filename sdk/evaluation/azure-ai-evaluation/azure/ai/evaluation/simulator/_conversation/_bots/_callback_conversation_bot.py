# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from typing import Any, Dict, List, Tuple, Union
import copy
import time
import jinja2

from azure.ai.evaluation._http_utils import AsyncHttpPipeline
from ..types import ConversationTurn
from ..._model_tools._template_handler import TemplateParameters
from .conversation_bot_base import ConversationBot


class CallbackConversationBot(ConversationBot):
    """Conversation bot that uses a user provided callback to generate responses.

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
        *args,
        **kwargs,
    ) -> None:
        self.callback = callback
        self.user_template = user_template
        self.user_template_parameters = user_template_parameters

        super().__init__(*args, **kwargs)

    async def generate_response(
        self,
        session: AsyncHttpPipeline,
        conversation_history: List[Any],
        max_history: int,
        turn_number: int = 0,
    ) -> Tuple[dict, dict, float, dict]:
        chat_protocol_message = self._to_chat_protocol(
            self.user_template, conversation_history, self.user_template_parameters
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

        return response, {}, time_taken, result

    # Bug 3354264: template is unused in the method - is this intentional?
    def _to_chat_protocol(self, template, conversation_history, template_parameters):  # pylint: disable=unused-argument
        messages = []

        for _, m in enumerate(conversation_history):
            messages.append({"content": m.message, "role": m.role.value})

        return {
            "template_parameters": template_parameters,
            "messages": messages,
            "$schema": "http://azureml/sdk-2-0/ChatConversation.json",
        }
