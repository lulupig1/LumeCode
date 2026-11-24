"""本地成本估算。

价格表只用于报告和趋势分析，不参与计费。默认值取自编写时的公开价格，
可能随 provider 调价而过时；调用方可以通过 ``PricingTable(overrides=...)``
覆盖任意条目。未知模型返回 cost=0 且 ``estimated=False``，避免把“没查到”
误报成“免费”。
"""

from __future__ import annotations

from dataclasses import dataclass

from lumecode.telemetry.models import Usage


@dataclass(frozen=True)
class ModelPrice:
    """每百万 token 的美元价格。"""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_creation_per_mtok: float = 0.0


# key 为模型名的小写子串。匹配时优先最长的 key，避免 "o3" 抢先命中 "o3-mini"。
# 这些是本地估算用的近似值，不是账单依据。
DEFAULT_PRICES: dict[str, ModelPrice] = {
    # Anthropic Claude
    "claude-opus": ModelPrice(15.0, 75.0, 1.50, 18.75),
    "claude-sonnet": ModelPrice(3.0, 15.0, 0.30, 3.75),
    "claude-haiku": ModelPrice(1.0, 5.0, 0.10, 1.25),
    # OpenAI
    "gpt-4.1": ModelPrice(2.0, 8.0, 0.50, 2.00),
    "gpt-4o": ModelPrice(2.50, 10.0, 1.25, 2.50),
    "o3": ModelPrice(2.0, 8.0, 0.50, 2.00),
    "o4-mini": ModelPrice(1.10, 4.40, 0.275, 1.10),
    # DeepSeek（V3 系列价格；V4 如不同请用 overrides 覆盖）
    "deepseek-chat": ModelPrice(0.27, 1.10, 0.07, 0.27),
    "deepseek-reasoner": ModelPrice(0.55, 2.19, 0.14, 0.55),
    "deepseek": ModelPrice(0.27, 1.10, 0.07, 0.27),
}


class PricingTable:
    """按模型名子串查找价格。"""

    def __init__(self, overrides: dict[str, ModelPrice] | None = None) -> None:
        merged = dict(DEFAULT_PRICES)
        if overrides:
            merged.update({key.lower(): value for key, value in overrides.items()})
        # 长 key 优先，保证更具体的模型名先匹配。
        self._entries = sorted(merged.items(), key=lambda item: len(item[0]), reverse=True)

    def lookup(self, model: str | None) -> ModelPrice | None:
        if not model:
            return None
        lowered = model.lower()
        for key, price in self._entries:
            if key in lowered:
                return price
        return None

    def estimate(self, model: str | None, usage: Usage | None) -> tuple[float, bool]:
        """返回 ``(cost_usd, estimated)``。

        ``estimated=False`` 表示模型不在价格表中，此时 cost 为 0。
        """
        if usage is None:
            return 0.0, False
        price = self.lookup(model)
        if price is None:
            return 0.0, False
        cost = (
            usage.input_tokens * price.input_per_mtok
            + usage.output_tokens * price.output_per_mtok
            + usage.cache_read_tokens * price.cache_read_per_mtok
            + usage.cache_creation_tokens * price.cache_creation_per_mtok
        ) / 1_000_000
        return round(cost, 8), True
