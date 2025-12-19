import aiohttp
import asyncio
import pandas as pd
from tqdm import tqdm

BASE_URL = "http://drugcombdb.denglab.org:8888"
PAGE_SIZE = 200
MAX_PAGES = 6000       # ~ complete DB
CONCURRENCY = 40

OUTPUT_RAW = "raw_integration.parquet"
sem = asyncio.Semaphore(CONCURRENCY)

async def fetch_page(session, page):
    async with sem:
        url = f"{BASE_URL}/integration/list"
        params = {"page": page, "size": PAGE_SIZE}
        async with session.get(url, params=params, timeout=30) as r:
            if r.status != 200:
                return []
            return (await r.json()).get("list", [])

async def main():
    rows = []

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(session, p) for p in range(1, MAX_PAGES + 1)]

        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            data = await fut
            for r in data:
                rows.append({
                    "block_id": r.get("blockId"),
                    "drugA": r.get("drugName1"),
                    "drugB": r.get("drugName2"),
                    "cell_line": r.get("cellName"),
                    "tissue": r.get("tissue"),
                    "synergy_score": r.get("synergyScore"),
                    "source": r.get("source")
                })

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_RAW)
    print("Saved:", OUTPUT_RAW)

if __name__ == "__main__":
    asyncio.run(main())
