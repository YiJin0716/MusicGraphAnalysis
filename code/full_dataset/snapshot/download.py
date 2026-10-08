"""Download all official raw_30s AcousticBrainz archives, with SHA-256 validation."""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path, expected=None):
    if path.exists() and (expected is None or digest(path) == expected):
        return
    for attempt in range(5):
        try:
            with requests.get(url, timeout=(30, 90), stream=True) as response:
                response.raise_for_status()
                temp = path.with_suffix(path.suffix + '.partial')
                with temp.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if expected and digest(temp) != expected:
                raise ValueError(f'Checksum mismatch: {path.name}')
            temp.replace(path)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    state_path = ROOT / 'source_metadata/download_manifest.json'
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
    else:
        state = {}
        for name, repository, branch in [('upstream', 'MTG/mtg-jamendo-dataset', 'master'),
                                         ('pipeline', 'YiJin0716/MusicGraphAnalysis', 'main')]:
            response = requests.get(f'https://api.github.com/repos/{repository}/commits/{branch}', timeout=60)
            response.raise_for_status()
            state[name] = {'repository': repository, 'commit': response.json()['sha']}
        state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    base = f"https://raw.githubusercontent.com/{state['upstream']['repository']}/{state['upstream']['commit']}/"
    files = ['data/raw_30s.tsv', 'data/raw_30s_cleantags_50artists.tsv',
             'data/autotagging_moodtheme.tsv', 'data/raw.meta.tsv',
             'data/download/raw_30s_acousticbrainz_sha256_tars.txt',
             'data/download/raw_30s_acousticbrainz_sha256_tracks.txt',
             'data/download/autotagging_moodtheme_acousticbrainz_sha256_tracks.txt',
             'README.md', 'LICENSE', 'CITATION.bib', 'audio_licenses.txt']
    metadata = []
    for name in files:
        path = ROOT / 'source_metadata' / Path(name).name
        fetch(base + name, path)
        metadata.append({'url': base + name, 'file': str(path.relative_to(ROOT)), 'sha256': digest(path), 'bytes': path.stat().st_size})
        print(f'Metadata verified: {path.name}', flush=True)
    pipeline = state['pipeline']
    for name in ['code/build_graph.py', 'code/requirements.txt', 'source_data/tracks.jsonl.gz']:
        target = ROOT / 'code' / ('reference_build_graph.py' if name.endswith('build_graph.py') else 'pilot_' + Path(name).name)
        fetch(f"https://raw.githubusercontent.com/{pipeline['repository']}/{pipeline['commit']}/{name}", target)
    checks = dict((line.split()[1], line.split()[0]) for line in (ROOT / 'source_metadata/raw_30s_acousticbrainz_sha256_tars.txt').read_text().splitlines())

    def archive(item):
        name, sha = item
        url = f'https://cdn.freesound.org/mtg-jamendo/raw_30s/acousticbrainz/{name}'
        path = ROOT / 'archives' / name
        fetch(url, path, sha)
        print(f'Archive verified: {name} ({path.stat().st_size:,} bytes)', flush=True)
        return {'file': name, 'url': url, 'sha256': sha, 'bytes': path.stat().st_size}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        archives = list(pool.map(archive, sorted(checks.items())))
    state.update(metadata=metadata, archives=archives, complete=True)
    state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    print(f'Download COMPLETE: {len(archives)} archives, {sum(a["bytes"] for a in archives):,} bytes', flush=True)


if __name__ == '__main__':
    main()
