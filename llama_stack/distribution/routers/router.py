# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import inspect
from types import MethodType
from typing import Any, AsyncGenerator, Dict, List, Tuple
from llama_stack.apis.inference import *  # noqa: F403
from fastapi import Request

from llama_stack.distribution.datatypes import Api, GenericProviderConfig
from llama_stack.distribution.distribution import api_protocols, api_providers
from llama_stack.distribution.utils.dynamic import instantiate_provider
from llama_stack.providers.registry.inference import available_providers
from termcolor import cprint


class Router:
    def __init__(self, api: str, provider_routing_table: Dict[str, Any]) -> None:
        # map of model_id to provider impl
        self.api = api
        self.providers_table = provider_routing_table[api]
        self.protocol = api_protocols()[Api(api)]
        # setattr(self, name, types.MethodType(Foo.bar, foo))

    async def initialize(self) -> None:
        print("initialize")
        all_providers = api_providers()[Api(self.api)]

        # look up and instantiate implementations for each method
        protocol_methods = inspect.getmembers(
            self.protocol, predicate=inspect.isfunction
        )

        self.routing_key_to_impls = {}
        for routing_entry in self.providers_table:
            impl = await instantiate_provider(
                all_providers[routing_entry.provider_id],
                deps=[],
                provider_config=GenericProviderConfig(
                    provider_id=routing_entry.provider_id,
                    config=routing_entry.config,
                ),
            )
            self.routing_key_to_impls[routing_entry.routing_key] = impl

        cprint(self.routing_key_to_impls, "red")

        # for name, method in protocol_methods:
        #     if not hasattr(method, "__webmethod__"):
        #         continue

        #     async def wrapper(self, name, **kwargs):
        #         impl = self.routing_key_to_impls["Meta-Llama3.1-8B-Instruct"]
        #         method = getattr(impl, name)
        #         return await method(**kwargs)

        #     cprint(f"method={method}, name={name}", "red")

        #     setattr(self, name, MethodType(wrapper, self))

    async def shutdown(self) -> None:
        pass

    async def chat_completion(self, *args, **kwargs):
        pass

    # async def completion(self, *args, **kwargs):
    #     pass

    # async def embedding(self, *args, **kwargs):
    #     pass

    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]: ...

    # async def chat_completion(
    #     self,
    #     model: str,
    #     messages: List[Message],
    #     sampling_params: Optional[SamplingParams] = SamplingParams(),
    #     # zero-shot tool definitions as input to the model
    #     tools: Optional[List[ToolDefinition]] = list,
    #     tool_choice: Optional[ToolChoice] = ToolChoice.auto,
    #     tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
    #     stream: Optional[bool] = False,
    #     logprobs: Optional[LogProbConfig] = None,
    # ) -> AsyncGenerator:
    #     async for chunk in self.routing_key_to_impls[model].chat_completion(
    #         model=model,
    #         messages=messages,
    #         sampling_params=sampling_params,
    #         tools=tools,
    #         tool_choice=tool_choice,
    #         tool_prompt_format=tool_prompt_format,
    #         stream=stream,
    #         logprobs=logprobs,
    #     ):
    #         yield chunk

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse: ...
