"""Quick script to inspect .ez file structure."""
import zipfile
import json
import sys

f = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\griff\Desktop\SyncLayerTest\SyncLayerTest.ez"

with zipfile.ZipFile(f) as z:
    data = json.loads(z.read("project.json"))

    for di in data["data_items"][:5]:
        print("---")
        print(f"  name: {di['name']}")
        print(f"  type: {di['type']}")
        print(f"  event_count: {di.get('event_count', '?')}")
        layers = di.get("layers", [])
        print(f"  layers: {len(layers)}")
        for layer in layers[:3]:
            events = layer.get("events", [])
            print(f"    layer '{layer.get('name', '?')}': {len(events)} events")
            if events:
                e = events[0]
                print(f"      sample event keys: {list(e.keys())}")
                print(f"      sample: {json.dumps(e)[:200]}")
