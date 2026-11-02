import urllib.request
import json

url = "https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/metric_depth/README.md"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    readme = resp.read().decode()

print(readme[:3000])

url_dir = "https://api.github.com/repos/DepthAnything/Depth-Anything-V2/contents/metric_depth/depth_anything_v2"
req_dir = urllib.request.Request(url_dir, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req_dir) as resp:
    files = json.loads(resp.read().decode())
print("\nFiles in depth_anything_v2:")
for f in files:
    print(f"{f['name']}: {f.get('download_url') or f['url']}")
