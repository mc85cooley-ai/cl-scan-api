# main.py
# LeagueAI Backend – eBay SOLD + ACTIVE ladder pricing (v6.9)
# Python 3.13 compatible

import os
import statistics
import httpx
from fastapi import FastAPI

app = FastAPI()

EBAY_APP_ID = os.getenv("EBAY_APP_ID", "")
UA = "LeagueAI/1.0"

def _normalize_prices(prices):
    prices = [p for p in prices if p > 0]
    if not prices:
        return {}
    prices.sort()
    return {
        "low": min(prices),
        "high": max(prices),
        "avg": round(sum(prices) / len(prices), 2),
        "median": round(statistics.median(prices), 2),
        "count": len(prices),
        "prices": prices
    }

async def _ebay_search(query, sold=True, limit=50):
    url = "https://svcs.ebay.com/services/search/FindingService/v1"
    params = {
        "OPERATION-NAME": "findCompletedItems" if sold else "findItemsAdvanced",
        "SERVICE-VERSION": "1.13.0",
        "SECURITY-APPNAME": EBAY_APP_ID,
        "RESPONSE-DATA-FORMAT": "JSON",
        "keywords": query,
        "paginationInput.entriesPerPage": limit,
    }
    if sold:
        params["itemFilter(0).name"] = "SoldItemsOnly"
        params["itemFilter(0).value"] = "true"

    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": UA}) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        j = r.json()

    root = "findCompletedItemsResponse" if sold else "findItemsAdvancedResponse"
    items = (
        j.get(root, [{}])[0]
        .get("searchResult", [{}])[0]
        .get("item", [])
    )

    prices = []
    for it in items:
        try:
            price = float(it["sellingStatus"][0]["currentPrice"][0]["__value__"])
            prices.append(price)
        except Exception:
            pass

    return prices

async def ebay_ladder(card_name, set_name, number):
    queries = [
        f"{card_name} {set_name} {number}",
        f"{card_name} {set_name} {number.split('/')[0]}",
        f"{card_name} {set_name}",
        f"{card_name} Pokemon card",
    ]
    for q in queries:
        sold_prices = await _ebay_search(q, sold=True)
        if len(sold_prices) >= 5:
            active_prices = await _ebay_search(q, sold=False)
            return {
                "used_query": q,
                "sold": _normalize_prices(sold_prices),
                "active": _normalize_prices(active_prices),
                "source": "ebay",
            }
    return {"available": False, "message": "No sufficient eBay market data found"}

@app.get("/market-context")
async def market_context(name: str, set: str, number: str):
    return await ebay_ladder(name, set, number)
