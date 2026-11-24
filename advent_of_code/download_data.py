# A script to download input data from AoC urls.

import sys
from pathlib import Path
from urllib.parse import urlparse
import re
import requests
import os


url = sys.argv[1]  # https://adventofcode.com/2015/day/1/input
parsed_url = urlparse(url)

assert parsed_url.scheme == "https", "Must be https!"
assert parsed_url.netloc == "adventofcode.com"

_path_matcher = r"/(?P<year>\d{4})/day/(?P<day>\d{1,2})/input"
match = re.fullmatch(_path_matcher, parsed_url.path)
if not match:
    raise ValueError(f"Bad AoC path {parsed_url.path} could not extract year \
                     and day!")

year = match.groupdict()["year"]
day = match.groupdict()["day"]

dl_path = Path(year) /day
dl_path.mkdir(parents=True, exist_ok=True)
dl_file = dl_path / "input"

print("Downloading ...")
_header = {
    'User-Agent': "priyamtejaswin's python script",
    'From': 'priyam.tejaswin@gmail.com'
}
response = requests.get(
    url, timeout=30, headers=_header,
    cookies={"session": os.environ["AOC_SESSION"]}
)
response.raise_for_status()
dl_file.write_bytes(response.content)
print("Done!")