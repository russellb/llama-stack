# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Tuple

from llama_stack.distribution.datatypes import Api


async def get_router_impl(api: str, provider_routing_table: Dict[str, Any]):
    from .router import Router

    impl = Router(api, provider_routing_table)
    await impl.initialize()
    return impl
