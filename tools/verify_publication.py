"""Verify the isolated publication, without loading any GPU model."""
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
inventory = json.loads((ROOT / 'PUBLICATION_INVENTORY.json').read_text())
for record in inventory['included']:
    data = (ROOT / record['published']).read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], record['published']

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.targets = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {'src', 'href'} and value:
                self.targets.append(value)

broken, secrets = [], []
python_count = html_count = media_count = 0
files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts]
for path in files:
    assert not path.is_symlink(), path
    assert path.stat().st_size < 100 * 1024 * 1024, path
    assert path.suffix.lower() not in {'.safetensors', '.pth', '.pt', '.ckpt', '.gguf', '.onnx', '.bin'}, path
    if path.suffix == '.py':
        ast.parse(path.read_text(), filename=str(path)); python_count += 1
    if path.suffix == '.json':
        json.loads(path.read_text())
    if path.suffix == '.html':
        parser = Links(); parser.feed(path.read_text())
        for target in parser.targets:
            if target.startswith(('#', 'data:', 'mailto:', 'http:', 'https:', 'javascript:')) or '${' in target:
                continue
            resolved = (path.parent / unquote(urlsplit(target).path)).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                broken.append({'file': str(path.relative_to(ROOT)), 'target': target})
        html_count += 1
    if path.suffix == '.mp4':
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))
        assert any(s['codec_type'] == 'video' for s in probe['streams']), path
        assert any(s['codec_type'] == 'audio' for s in probe['streams']), path
        media_count += 1
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            assert not any(Path(n).suffix in {'.safetensors','.pt','.pth','.ckpt','.gguf'} or Path(n).name.startswith('.env') for n in z.namelist()), path
    if path.suffix.lower() in {'.py','.mjs','.js','.json','.jsonl','.html','.md','.log','.txt','.yml','.yaml'}:
        data = path.read_text(errors='replace')
        patterns = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{30,}',
                    r'AKIA[0-9A-Z]{16}', r'hf_[A-Za-z0-9]{24,}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
        for pattern in patterns:
            if re.search(pattern, data):
                secrets.append({'file': str(path.relative_to(ROOT)), 'pattern': pattern})
assert not broken, broken
assert not secrets, secrets
for report in ['pika-fp8-five-video-comparison-20261102','pika-preloaded-five-video-comparison-20261102']:
    base = ROOT / 'experiments/pika-h3' / report
    hashes = json.loads((base / 'artifact-integrity.json').read_text())
    # Reports use a path -> digest mapping or per-entry digest records.
    if isinstance(hashes, dict):
        for relative, digest in hashes.items():
            if isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest):
                assert hashlib.sha256((base / relative).read_bytes()).hexdigest() == digest, relative
summary = {'snapshot_local': inventory['snapshot_local'], 'source_files_verified': len(inventory['included']),
           'python_files_parsed': python_count, 'html_files_checked': html_count,
           'videos_with_audio_probed': media_count, 'broken_static_links': broken,
           'credential_pattern_matches': secrets, 'model_weight_files': 0,
           'easycache_terminal_runs_at_cutoff': len(inventory['easycache_snapshot_terminal_runs']),
           'total_files_before_validation_record': len(files)}
(ROOT / 'PUBLICATION_VALIDATION.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
