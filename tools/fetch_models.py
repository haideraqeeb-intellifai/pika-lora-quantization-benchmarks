"""List or explicitly download the pinned model manifest. Never commits model files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--kind', choices=['int8', 'bf16', 'both'], default='int8')
parser.add_argument('--dest', type=Path, default=ROOT / 'experiments/pika-h3/models')
parser.add_argument('--download', action='store_true')
args = parser.parse_args()
records = json.loads((ROOT / 'model-manifest.json').read_text())['assets']
selected = [r for r in records if r['kind'] == 'common' or args.kind == 'both' or r['kind'] == args.kind]
print(json.dumps({'assets': selected, 'total_bytes': sum(r['bytes'] for r in selected)}, indent=2))
if not args.download:
    print('Listing only; pass --download to retrieve these weights.')
    raise SystemExit(0)
for record in selected:
    target = args.dest / record['local_path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if record.get('repository'):
            from huggingface_hub import hf_hub_download
            cached = Path(hf_hub_download(record['repository'], record['filename'],
                          revision=record['revision'], cache_dir=args.dest / '.cache'))
            target.symlink_to(cached.resolve())
        else:
            temporary = target.with_suffix(target.suffix + '.partial')
            with urllib.request.urlopen(record['url'], timeout=90) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            temporary.rename(target)
    assert target.stat().st_size == record['bytes'], f'Size mismatch: {target.name}'
    with target.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == record['sha256'], f'Hash mismatch: {target.name}'
    print('Verified ' + str(target))
