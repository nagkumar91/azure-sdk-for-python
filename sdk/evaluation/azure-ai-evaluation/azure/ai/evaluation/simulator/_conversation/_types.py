# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .constants import ConversationRole


@dataclass
class ConversationTurn:
    """Class to represent a turn in a conversation.

    A "turn" involves only one exchange between the user and the chatbot.

    :param role: The role of the participant in the conversation. 
        Accepted values are "user" and "assistant".
    :type role: ~azure.ai.evaluation.simulator._conversation.constants.ConversationRole
    :param name: The name of the participant in the conversation.
    :type name: Optional[str]
    :param message: The message exchanged in the conversation. 
        Defaults to an empty string.
    :type message: str
    :param full_response: The full response.
    :type full_response: Optional[Any]
    :param request: The request.
    :type request: Optional[Any]
    """

    role: "ConversationRole"
    name: Optional[str] = None
    message: str = ""
    full_response: Optional[Dict[str, Any]] = None
    request: Optional[Any] = None

    def to_openai_chat_format(self, reverse: bool = False) -> Dict[str, str]:
        """Convert the conversation turn to the OpenAI chat format.

        OpenAI chat format is a dictionary with two keys: "role" and "content".

        :param reverse: Whether to reverse the conversation turn. Defaults to False.
        :type reverse: bool
        :return: The conversation turn in the OpenAI chat format.
        :rtype: Dict[str, str]
        """
        if reverse is False:
            return {"role": self.role.value, "content": self.message}
        if self.role == ConversationRole.ASSISTANT:
            return {
                "role": ConversationRole.USER.value,
                "content": self.message
            }
        return {
            "role": ConversationRole.ASSISTANT.value,
            "content": self.message
        }

    def to_annotation_format(self, turn_number: int) -> Dict[str, Any]:
        """Convert the conversation turn to an annotation format.

        Annotation format is a dictionary with the following keys:
        - "turn_number": The turn number.
        - "response": The response.
        - "actor": The actor.
        - "request": The request.
        - "full_json_response": The full JSON response.

        :param turn_number: The turn number.
        :type turn_number: int
        :return: The conversation turn in the annotation format.
        :rtype: Dict[str, Any]
        """
        return {
            "turn_number": turn_number,
            "response": self.message,
            "actor": self.role.value if self.name is None else self.name,
            "request": self.request,
            "full_json_response": self.full_response,
        }

    def __str__(self) -> str:
        return f"({self.role.value}): {self.message}"
