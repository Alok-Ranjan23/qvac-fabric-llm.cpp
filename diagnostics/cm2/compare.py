"""Run matched GPU inputs, report attribution without disguising test failures."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path.cwd()
output = root / 'cm2-results'
output.mkdir(exist_ok=True)
corpus = output / 'corpus'
corpus.mkdir(exist_ok=True)
count = int(os.environ.get('CM2_CORPUS_COUNT', '1024'))
backend = os.environ.get('CM2_DIAGNOSTIC_BACKEND', 'Vulkan0')
results = []

def digest():
    files = list(corpus.glob('*.bin'))
    assert len(files) == count, (len(files), count)
    h = hashlib.sha256()
    for f in sorted(files, key=lambda p: int(p.stem)):
        h.update(f.name.encode() + b'\0' + f.read_bytes())
    return h.hexdigest()

for label, tree, mode in [('base-capture', 'base', 'capture'),
                          ('head-replay-1', 'head', 'replay'),
                          ('base-replay', 'base', 'replay'),
                          ('head-replay-2', 'head', 'replay')]:
    env = dict(os.environ, CM2_CORPUS_COUNT=str(count), CM2_CORPUS_DIR=str(corpus),
               CM2_CORPUS_MODE=mode, GGML_N_THREADS='1')
    env.pop('GGML_VK_DISABLE_COOPMAT2', None)
    cmd = [str(root / tree / 'build-cm2/bin/test-backend-ops'), '-b', backend, '-o', 'SET_ROWS', '-j', '2']
    log = output / (label + '.log')
    with log.open('w') as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=240)
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', log.read_text())
    if backend == 'Vulkan0':
        assert 'matrix cores: NV_coopmat2' in text, 'Did not exercise NV_coopmat2'
        assert 'NVIDIA RTX 4000 SFF Ada Generation' in text, 'Unexpected GPU'
    assert f'/{count} tests passed' in text, 'Missing complete test summary'
    outcomes = {}
    for line in text.splitlines():
        m = re.search(r'SET_ROWS\([^\n]*corpus=(\d+)\):.*\b(OK|FAIL)\b', line)
        if m:
            outcomes[int(m[1])] = m[2]
    assert len(outcomes) == count, ('Missing or unsupported cases', len(outcomes), count)
    failed = sorted(i for i, status in outcomes.items() if status == 'FAIL')
    assert p.returncode == (1 if failed else 0), ('Unexpected exit', p.returncode)
    result = dict(label=label, exit=p.returncode, count=count, failed_corpus_ids=failed, corpus_sha256=digest())
    results.append(result)
    print(json.dumps(result), flush=True)
    (output / 'partial-results.json').write_text(json.dumps(results, indent=2))

assert len({r['corpus_sha256'] for r in results}) == 1, 'Corpus changed between runs'
base_fail = set(results[0]['failed_corpus_ids'])
head_fail = set(results[1]['failed_corpus_ids'])
base_repeat = set(results[2]['failed_corpus_ids'])
head_repeat = set(results[3]['failed_corpus_ids'])
if base_fail and base_fail == head_fail == base_repeat == head_repeat:
    verdict = 'SAME_FAILURE_ON_BASE_AND_HEAD: reproduced pre-existing failure on identical inputs'
elif not any(r['failed_corpus_ids'] for r in results):
    verdict = 'INCONCLUSIVE: original randomized failure did not reproduce in this corpus'
elif head_fail and not base_fail and not base_repeat and head_fail == head_repeat:
    verdict = 'HEAD_ONLY_FAILURE: investigate PR regression before team review'
else:
    verdict = 'INCONCLUSIVE: base/head failure sets or repeats differ'
report = dict(base_sha='f5ddcebbddb0bf1c87b60e5ca81c720d1c5cd5af',
              head_sha='5ceeacf03396914fb805fdc48d8c0c3b6f1a934d', verdict=verdict, runs=results,
              limitation='Bounded exact-case attribution, not full GPU suite or production qualification; original CI tensor bytes were not saved.')
(output / 'comparison.json').write_text(json.dumps(report, indent=2))
print(verdict)
if 'GITHUB_STEP_SUMMARY' in os.environ:
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
        f.write('```json\n' + json.dumps(report, indent=2) + '\n```\n')
