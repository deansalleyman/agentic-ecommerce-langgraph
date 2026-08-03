"""Store catalog and the tools the agent uses to shop it.

The catalog is seed data in app/data/products.json. `search_products` is the seam a real
inventory backend would slot into — the agent only ever sees the tool signature.
"""

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from app.schemas import Product

_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "products.json"

CATALOG: list[Product] = [
    Product.model_validate(row) for row in json.loads(_CATALOG_PATH.read_text())
]
_BY_ID: dict[str, Product] = {product.product_id: product for product in CATALOG}
CATEGORIES: list[str] = sorted({product.category for product in CATALOG})

# Results per search. Small on purpose: the whole list would crowd out the conversation,
# and a shortlist the customer can actually hold in their head is the point.
_MAX_RESULTS = 5


def product_by_id(product_id: str) -> Product | None:
    """Plain lookup, for code that needs a product without going through the tool."""
    return _BY_ID.get(product_id)


def _summarize(product: Product) -> dict[str, Any]:
    """The view the model sees — full specs come from get_product."""
    summary: dict[str, Any] = {
        "product_id": product.product_id,
        "name": product.name,
        "brand": product.brand,
        "category": product.category,
        "price_usd": product.price_usd,
        "in_stock": product.in_stock,
        "description": product.description,
    }
    for field in ("weight_lb", "season", "capacity", "temp_rating_c"):
        value = getattr(product, field)
        if value is not None:
            summary[field] = value
    return summary


@tool
def search_products(
    category: str,
    max_price_usd: float | None = None,
    max_weight_lb: float | None = None,
    season: str | None = None,
    min_capacity: int | None = None,
    activity: str | None = None,
) -> list[dict[str, Any]]:
    """Search the camping store catalog for products matching a customer's requirements.

    Use this before recommending anything, so recommendations only ever cite real stock.

    Args:
        category: One of tent, sleeping_bag, sleeping_pad, stove, backpack, headlamp,
            water_filter, insulation.
        max_price_usd: Upper price limit in USD.
        max_weight_lb: Upper weight limit in lb. Matters for backpacking, not car camping.
        season: summer, 3-season or winter. Products rated for harsher conditions than
            asked for are included; lighter-rated ones are not.
        min_capacity: Minimum people (tents) or litres (backpacks).
        activity: Filter to products suited to an activity, e.g. backpacking, winter hiking.

    Returns:
        Up to 5 matching products, cheapest first. Empty list if nothing matches — say so
        rather than inventing an alternative.
    """
    # Harsher-rated gear also serves a milder trip, so treat season as a floor.
    season_rank = {"summer": 0, "3-season": 1, "winter": 2}
    wanted_rank = season_rank.get(season or "", None)

    matches: list[Product] = []
    for product in CATALOG:
        if product.category != category or not product.in_stock:
            continue
        if max_price_usd is not None and product.price_usd > max_price_usd:
            continue
        if max_weight_lb is not None and (
            product.weight_lb is None or product.weight_lb > max_weight_lb
        ):
            continue
        if wanted_rank is not None and (
            product.season is None or season_rank[product.season] < wanted_rank
        ):
            continue
        if min_capacity is not None and (
            product.capacity is None or product.capacity < min_capacity
        ):
            continue
        if activity is not None and activity.lower() not in [
            a.lower() for a in product.activities
        ]:
            continue
        matches.append(product)

    matches.sort(key=lambda p: p.price_usd)
    return [_summarize(product) for product in matches[:_MAX_RESULTS]]


@tool
def get_product(product_id: str) -> dict[str, Any]:
    """Look up the full details of one catalog product by its id.

    Use when comparing a shortlist, or when the customer asks about a specific item.

    Args:
        product_id: The catalog id, exactly as returned by search_products.

    Returns:
        The product's full record, or an error message if the id is not in the catalog.
    """
    product = _BY_ID.get(product_id)
    if product is None:
        return {"error": f"No product with id {product_id!r} in the catalog."}
    return product.model_dump()


catalog_tools = [search_products, get_product]
