# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio

from typing import AsyncIterator, Union

from llama_models.llama3.api.datatypes import StopReason
from llama_models.sku_list import resolve_model

from llama_toolchain.models.api import *  # noqa: F403
from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_models.datatypes import CoreModelId, Model

from llama_toolchain.inference.api import Inference
from llama_toolchain.safety.api import Safety

from .config import MetaReferenceImplConfig

DUMMY_MODEL_SPEC = ModelSpec(
    metadata=Model(
        core_model_id=CoreModelId.meta_llama3_8b_instruct,
        is_default_variant=True,
        description_markdown="Llama 3 8b instruct model",
        huggingface_repo="meta-llama/Meta-Llama-3-8B-Instruct",
        # recommended_sampling_params=recommended_sampling_params(),
        model_args={
            "dim": 4096,
            "n_layers": 32,
            "n_heads": 32,
            "n_kv_heads": 8,
            "ffn_dim_multiplier": 1.3,
            "multiple_of": 1024,
            "norm_eps": 1e-05,
            "rope_theta": 500000.0,
            "use_scaled_rope": False,
        },
        pth_file_count=1,
    ),
    providers_spec={
        "inference": [{"provider_type": "meta-reference"}],
    },
)


class MetaReferenceModelsImpl(Models):
    def __init__(
        self,
        config: MetaReferenceImplConfig,
        inference_api: Inference,
        safety_api: Safety,
    ) -> None:
        self.config = config
        self.inference_api = inference_api
        self.safety_api = safety_api

    async def initialize(self) -> None:
        pass

    async def list_models(self) -> ModelsListResponse:
        return ModelsListResponse(models_list=[DUMMY_MODEL_SPEC])

    async def get_model(self, model_id: str) -> ModelsGetResponse:
        return ModelsGetResponse(core_model_spec=DUMMY_MODEL_SPEC)

    async def register_model(
        self, model_id: str, api: str, provider_spec: Dict[str, str]
    ) -> ModelsRegisterResponse:
        return ModelsGetResponse(core_model_spec=DUMMY_MODEL_SPEC)
