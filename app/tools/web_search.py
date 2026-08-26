from ddgs import DDGS
from ddgs.exceptions import DDGSException


def web_search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    timelimit: str | None = None,
) -> dict:
    """
    Free, no-API-key web search via DuckDuckGo. Stateless — every call
    is independent and returns everything the caller needs.

    Args:
        query: search string.
        max_results: how many results to return.
        region: DDG region code, e.g. "us-en", "uk-en", "wt-wt" (no region).
        safesearch: "on", "moderate", or "off".
        timelimit: restrict by recency — "d" (day), "w" (week),
            "m" (month), "y" (year), or None for no limit.

    Returns:
        {
            "query": str,
            "results": [
                {"title": str, "url": str, "snippet": str},
                ...
            ],
            "error": str | None,
        }
    """
    try:
        raw_results = DDGS().text(
            query,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            max_results=max_results,
        )
    except DDGSException as e:
        return {"query": query, "results": [], "error": str(e)}

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in raw_results
    ]

    return {"query": query, "results": results, "error": None}


if __name__ == "__main__":
    result = web_search("Anthropic Claude", max_results=3)
    for r in result["results"]:
        print(r["title"])
        print(r["url"])
        print(r["snippet"])
        print()