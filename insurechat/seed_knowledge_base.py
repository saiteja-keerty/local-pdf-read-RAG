"""
Seed the knowledge_base/insurance_glossary.md by fetching public URLs (optional).
This is designed to be run once with network access.
"""
import requests
from pathlib import Path

KB = Path(__file__).parent / 'knowledge_base' / 'insurance_glossary.md'
KB.parent.mkdir(exist_ok=True)

SOURCES = [
    'https://www.healthcare.gov/what-are-premiums-deductibles-copayments-and-coinsurance/',
]

out = []
for url in SOURCES:
    try:
        r = requests.get(url, timeout=10)
        if r.ok:
            out.append(f"# Source: {url}\n\n"+r.text[:20000])
    except Exception as e:
        print('Failed', url, e)

KB.write_text('\n\n'.join(out), encoding='utf8')
print('Wrote', KB)
