import asyncio
import aiohttp
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pymongo import MongoClient
import spacy


client = MongoClient("mongodb://localhost:27017/")
db = client["disaster_db"]
collection = db["events"]


apis = {
    "news": "https://newsapi.org/v2/everything?q=disaster&apiKey=6cfb3ad6433845bea794f002fa1913e3",
    "reliefweb": "https://api.reliefweb.int/v1/disasters?limit=100",
    "gdacs": "https://www.gdacs.org/xml/rss.xml",
    "usgs": "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&orderby=time"
}

nlp = spacy.load("en_core_web_sm")


DISASTER_TERMS = {
    "earthquake", "flood", "cyclone", "hurricane", "wildfire",
    "tsunami", "volcano", "eruption", "typhoon", "storm", "landslide",
    "tornado", "disaster", "avalanche", "drought"
}


def is_disaster_event(text):
    if not text:
        return False
    doc = nlp(text)

    # Lemmatization-based keyword check (fast broad filter)
    lemmas = {token.lemma_.lower() for token in doc}
    if any(term in lemmas for term in DISASTER_TERMS):
        return True

    # Named Entity Recognition check (contextual check)
    for ent in doc.ents:
        # Check entities labeled as EVENT or ORG (organizations involved in disasters)
        if ent.label_ in {"EVENT", "ORG"}:
            # See if any disaster term is part of the entity text
            if any(term in ent.text.lower() for term in DISASTER_TERMS):
                return True

    return False

def hybrid_filter(events):
    filtered = []
    for evt in events:
        text = (evt.get("title") or "") + " " + (evt.get("summary") or "")
        if is_disaster_event(text):
            filtered.append(evt)
    return filtered

def normalize_news(data):
    articles = json.loads(data).get("articles", [])
    return [
        {
            "source": "newsapi",
            "title": art.get("title"),
            "date": art.get("publishedAt"),
            "location": None,
            "severity": "medium",
            "summary": art.get("description"),
            "inserted_at": datetime.utcnow()
        }
        for art in articles
    ]

def normalize_reliefweb(data):
    items = json.loads(data).get("data", [])
    results = []
    for item in items:
        fields = item.get("fields", {})
        results.append({
            "source": "reliefweb",
            "title": fields.get("name"),
            "date": fields.get("date", {}).get("created"),
            "location": fields.get("country", [{}])[0].get("name") if fields.get("country") else None,
            "severity": fields.get("status"),
            "summary": fields.get("description"),
            "inserted_at": datetime.utcnow()
        })
    return results

def normalize_gdacs(data):
    try:
        root = ET.fromstring(data)
        items = []
        for item in root.findall("./channel/item"):
            items.append({
                "source": "gdacs",
                "title": item.findtext("title"),
                "date": item.findtext("pubDate"),
                "location": None,
                "severity": "high",
                "summary": item.findtext("description"),
                "inserted_at": datetime.utcnow()
            })
        return items
    except Exception as e:
        print("[gdacs] parse error:", e)
        return []

def normalize_usgs(data):
    features = json.loads(data).get("features", [])
    return [
        {
            "source": "usgs",
            "title": feat["properties"].get("title"),
            "date": datetime.utcfromtimestamp(feat["properties"]["time"] / 1000).isoformat() if feat["properties"].get("time") else None,
            "location": feat["properties"].get("place"),
            "severity": f"Magnitude {feat['properties'].get('mag')}",
            "summary": feat["properties"].get("title"),
            "inserted_at": datetime.utcnow()
        }
        for feat in features
    ]


normalizers = {
    "news": normalize_news,
    "reliefweb": normalize_reliefweb,
    "gdacs": normalize_gdacs,
    "usgs": normalize_usgs,
}


async def fetch(session, name, url):
    try:
        async with session.get(url) as response:
            raw = await response.text()
            print(f"[{name}] Status {response.status}, length {len(raw)}")
            return name, raw
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return name, None

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, name, url) for name, url in apis.items()]
        raw_results = await asyncio.gather(*tasks)

        all_normalized = []
        for name, raw in raw_results:
            if raw and name in normalizers:
                parsed = normalizers[name](raw)
                print(f"[{name}] normalized {len(parsed)} items")

                disaster_events = hybrid_filter(parsed)
                print(f"[{name}] hybrid-filtered {len(disaster_events)} disaster items")

                if disaster_events:
                    collection.insert_many(disaster_events)

                all_normalized.extend(disaster_events)

        return all_normalized

if __name__ == "__main__":
    normalized_data = asyncio.run(main())
    print(f"Inserted {len(normalized_data)} disaster events into MongoDB")
