# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import logging
import jinja2
import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from azure.ai.evaluation._http_utils import AsyncHttpPipeline
from azure.ai.evaluation._exceptions import ErrorBlame, ErrorCategory, ErrorTarget, EvaluationException

# For type hints:
from ..._model_tools import LLMBase, OpenAIChatCompletionsModel
# or from azure.ai.evaluation.simulator._model_tools import LLMBase, OpenAIChatCompletionsModel
from .._constants import ConversationRole
from .._types import ConversationTurn
from ..._model_tools._template_handler import TemplateParameters


class ConversationBot:
    """
    A conversation chat bot with a specific name, persona and a sentence that can be used as a conversation starter.

    :param role: The role of the bot in the conversation, either "user" or "assistant".
    :type role: ~azure.ai.evaluation.simulator._conversation.constants.ConversationRole
    :param model: The LLM model to use for generating responses.
    :type model: Union[
        ~azure.ai.evaluation.simulator._model_tools.LLMBase,
        ~azure.ai.evaluation.simulator._model_tools.OpenAIChatCompletionsModel
    ]
    :param conversation_template: A Jinja2 template describing the conversation to generate the prompt for the LLM
    :type conversation_template: str
    :param instantiation_parameters: A dictionary of parameters used to instantiate the conversation template
    :type instantiation_parameters: Dict[str, str]
    """

    def __init__(
        self,
        *,
        role: ConversationRole,
        model: Union[LLMBase, OpenAIChatCompletionsModel],
        conversation_template: str,
        instantiation_parameters: TemplateParameters,
    ) -> None:
        self.role = role
        self.conversation_template_orig = conversation_template
        self.conversation_template: jinja2.Template = jinja2.Template(
            conversation_template, undefined=jinja2.StrictUndefined
        )
        self.persona_template_args = instantiation_parameters
        if self.role == ConversationRole.USER:
            self.name: str = cast(str, self.persona_template_args.get("name", role.value))
        else:
            self.name = cast(str, self.persona_template_args.get("chatbot_name", role.value)) or model.name
        self.model = model

        self.logger = logging.getLogger(repr(self))
        self.conversation_starter: Optional[Union[str, jinja2.Template, Dict]] = None
        if role == ConversationRole.USER:
            if "conversation_starter" in self.persona_template_args:
                print(self.persona_template_args)
                conversation_starter_content = self.persona_template_args["conversation_starter"]
                if isinstance(conversation_starter_content, dict):
                    self.conversation_starter = conversation_starter_content
                    print(f"Conversation starter content: {conversation_starter_content}")
                else:
                    try:
                        self.conversation_starter = jinja2.Template(
                            conversation_starter_content, undefined=jinja2.StrictUndefined
                        )
                        print("Successfully created a Jinja2 template for the conversation starter.")
                    except jinja2.exceptions.TemplateSyntaxError as e:  # noqa: F841
                        print(f"Template syntax error: {e}. Using raw content.")
                        self.conversation_starter = conversation_starter_content
            else:
                self.logger.info(
                    "This simulated bot will generate the first turn as no conversation starter is provided"
                )

    async def generate_response(
        self,
        session: AsyncHttpPipeline,
        conversation_history: List[ConversationTurn],
        max_history: int,
        turn_number: int = 0,
    ) -> Tuple[dict, dict, float, dict]:
        """
        Prompt the ConversationBot for a response.

        :param session: AsyncHttpPipeline to use for the request.
        :type session: AsyncHttpPipeline
        :param conversation_history: The turns in the conversation so far.
        :type conversation_history: List[ConversationTurn]
        :param max_history: Parameters used to query GPT-4 model.
        :type max_history: int
        :param turn_number: Parameters used to query GPT-4 model.
        :type turn_number: int
        :return: The response from the ConversationBot.
        :rtype: Tuple[dict, dict, float, dict]
        """

        # check if this is the first turn and the conversation_starter is not None,
        # return the conversations starter rather than generating turn using LLM
        if turn_number == 0 and self.conversation_starter is not None:
            # if conversation_starter is a dictionary, pass it into samples as is
            if isinstance(self.conversation_starter, dict):
                samples: List[Union[str, jinja2.Template, Dict]] = [self.conversation_starter]
            if isinstance(self.conversation_starter, jinja2.Template):
                samples = [self.conversation_starter.render(**self.persona_template_args)]
            else:
                samples = [self.conversation_starter]
            jailbreak_string = self.persona_template_args.get("jailbreak_string", None)
            if jailbreak_string:
                samples = [f"{jailbreak_string} {samples[0]}"]
            time_taken = 0

            finish_reason = ["stop"]

            parsed_response = {"samples": samples, "finish_reason": finish_reason, "id": None}
            full_response = parsed_response
            return parsed_response, {}, time_taken, full_response

        try:
            prompt = self.conversation_template.render(
                conversation_turns=conversation_history[-max_history:],
                role=self.role.value,
                **self.persona_template_args,
            )
        except Exception:  # pylint: disable=broad-except
            import code

            code.interact(local=locals())

        messages = [{"role": "system", "content": prompt}]

        # The ChatAPI must respond as ASSISTANT, so if this bot is USER, we need to reverse the messages
        if (self.role == ConversationRole.USER) and (isinstance(self.model, (OpenAIChatCompletionsModel))):
            # in here we need to simulate the user, The chatapi only generate turn as assistant and
            # can't generate turn as user
            # thus we reverse all rules in history messages,
            # so that messages produced from the other bot passed here as user messages
            messages.extend([turn.to_openai_chat_format(reverse=True) for turn in conversation_history[-max_history:]])
            prompt_role = ConversationRole.USER.value
        else:
            messages.extend([turn.to_openai_chat_format() for turn in conversation_history[-max_history:]])
            prompt_role = self.role.value

        response = await self.model.get_conversation_completion(
            messages=messages,
            session=session,
            role=prompt_role,
        )

        return response["response"], response["request"], response["time_taken"], response["full_response"]

    def __repr__(self):
        return f"Bot(name={self.name}, role={self.role.name}, model={self.model.__class__.__name__})"
