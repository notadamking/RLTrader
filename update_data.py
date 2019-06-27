import asyncio
import ssl
import pandas as pd
import os

final_fmt = '%Y-%m-%d %H:%M'
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://www.cryptodatadownload.com/cdd/Coinbase_BTCUSD_1h.csv"
url2 = "https://www.cryptodatadownload.com/cdd/Coinbase_BTCUSD_d.csv"


async def get_daily(url: str, fmt: str, fn: str):
    c = pd.read_csv(url, header=1)
    c = c.dropna(thresh=2)
    c.columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'VolumeFrom', 'VolumeTo']
    c['Date'] = pd.to_datetime(c['Date'], format=fmt)
    c['Date'] = c['Date'].dt.strftime(final_fmt)

    p = os.path.join('data', 'input', fn)
    c.to_csv(p, index=False)

    return c


async def get_pages(url1: str, url2: str):
    tasks = [get_daily(url1, '%Y-%m-%d %I-%p', 'coinbase-1h-btc-usd.csv'), get_daily(url2, '%Y-%m-%d', 'coinbase-1d-btc-usd.csv')]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)  # also FIRST_EXCEPTION and ALL_COMPLETED (default)
    print('>> done: ', done)
    print('>> pending: ', pending)  # will be empty if using default return_when setting


loop = asyncio.get_event_loop()
loop.run_until_complete(get_pages(url, url2))
loop.close()
