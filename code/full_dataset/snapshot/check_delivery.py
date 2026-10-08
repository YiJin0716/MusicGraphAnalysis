"""Check final reports, completed experiments, and cross-experiment coverage."""
import json
from html.parser import HTMLParser

from run_experiments import ROOT, write_json
from download import digest


class Links(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value and not value.startswith(('http:', 'https:', '#')):
                if not (self.path.parent / value).exists():
                    raise ValueError(f'Missing report link: {self.path}: {value}')


def main():
    subsets = {}
    for name in ['moodtheme', 'full']:
        out = ROOT / 'experiments' / name
        completed = json.loads((out / 'COMPLETE.json').read_text(encoding='utf-8'))
        assert completed['independently_validated']
        validation = json.loads((out / 'validation.json').read_text(encoding='utf-8'))
        assert validation['passed']
        assert validation['validator_sha256'] in {digest(ROOT / 'code' / filename) for filename in ['validate.py', 'validate_blocked.py', 'validate_streamed.py']}
        assert len(list((out / 'graphs').glob('k*.npz'))) == 63
        assert len(list((out / 'components').glob('k*.npz'))) == 63
        subsets[name] = {json.loads(line)['track_id'] for line in (out / 'nodes.jsonl').read_text(encoding='utf-8').splitlines()}
    assert subsets['moodtheme'] < subsets['full']
    reports = [ROOT / 'comparison.html'] + [ROOT / 'experiments' / name / 'report.html' for name in ['moodtheme', 'full']]
    for path in reports:
        Links(path).feed(path.read_text(encoding='utf-8'))
    write_json(ROOT / 'delivery_validation.json', dict(passed=True, reports_checked=3,
               total_saved_graphs=126, total_component_files=126, moodtheme_subset_of_full=True,
               both_experiments_independently_validated=True,
               total_files=sum(p.is_file() for p in ROOT.rglob('*')),
               total_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())))
    print('FINAL DELIVERY CHECK PASSED', flush=True)


if __name__ == '__main__':
    main()
