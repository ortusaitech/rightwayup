#!/usr/bin/env python3
"""[lead-push] Pod-side fetcher for the pinned backbone weights in rotlab.model_x.SPECS, with provenance receipts.
Anonymous HTTPS only (every repo is public and ungated; no token or storage credential on the pod).

  python -m rotlab.fetch_weights pins                                  # pinned table (no network)
  python -m rotlab.fetch_weights check  pe_s tips_b b b_reg            # Hub API + card + licence texts only, no weights
  python -m rotlab.fetch_weights fetch  pe_s tips_b [--root DIR] [--verify-upstream] [--keep-upstream]
  python -m rotlab.fetch_weights verify pe_s tips_b [--root DIR]       # offline: re-hash vs pins + receipt, strict load
Default root: model_x.EXT (/workspace/rotation-data/quarantine/external-weights; env ROTLAB_EXT_WEIGHTS).
Hub endpoint: https://huggingface.co (env HF_ENDPOINT or --endpoint; tests use a file:// mirror).

fetch, per arch; fail closed: any mismatch prints REFUSE, exits 2 and leaves no unverified file at the destination:
 1. Hub API at the pinned revision: the revision resolves to itself; repo public and not gated; licence tag and model-card
    YAML licence = apache-2.0; every pinned file's size + LFS SHA-256 equal the pin (checked BEFORE downloading).
 2. Licence texts at pinned GitHub commits: SHA-256 must equal the pin; the licence sentence quoted in PROV must occur in
    the (whitespace-normalised) text. Saved beside the weights.
 3. model.safetensors, README.md (model card) and config.json from <endpoint>/<repo>/resolve/<revision>/<file>, streamed
    to <file>.part and hashed on the fly: weights must equal the pinned size + SHA-256; card and config must equal the git
    blob id the Hub lists at that revision. Then fsync + atomic rename. An existing file is reused only if it verifies
    (never overwritten).
 4. Strict load (timm load_state_dict strict=True) into the architecture; records timm/torch versions and the SHA-256 of
    the timm source file that defines the architecture (the code that interprets the weights).
 5. --verify-upstream (pe_s, tips_b): download the ORIGINAL release (facebook/PE-Spatial-S16-512 .pt; google/tipsv2-b14
    safetensors) at its pinned revision + SHA-256, convert it with timm's own checkpoint filter and require the same keys,
    shapes and bit-identical values as the timm copy. The upstream file is deleted afterwards unless --keep-upstream.
 6. Receipt PROVENANCE.<arch>.json + .sha256 beside the weights (never overwritten; a re-run writes a timestamped one):
    repo, pinned + resolved revision, head revision at access time, file list with sizes / SHA-256 / expected values,
    raw API JSON hashes, model-card URL + hash + YAML licence, licence-text URLs + hashes, upstream parity, code versions,
    tool hashes, access time (UTC) and host.
"""
from __future__ import annotations

import argparse, datetime, hashlib, inspect, json, os, re, socket, sys, time, urllib.error, urllib.request
from pathlib import Path

from rotlab import model_x as mx

SCHEMA = 'rotlab.fetch_weights/1'
GH = 'https://raw.githubusercontent.com'
UA = {'User-Agent': 'rotlab-fetch-weights/1'}
DINOV2_LIC = [
    dict(name='LICENSE.dinov2', url=f'{GH}/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE',
         sha256='600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7', quote='Apache License Version 2.0, January 2004'),
    dict(name='README.dinov2.md', url=f'{GH}/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md',
         sha256='d1bc2e9686522bbd66ed6123dc81eb36b3a98e8ea9a2ca97778f54a1c641c9e1',
         quote='DINOv2 code and model weights are released under the Apache License 2.0.'),
]
TIMM_CODE_LIC = dict(name='LICENSE.timm', url=f'{GH}/huggingface/pytorch-image-models/107fdc039527165f01faf83356ca56fcc09396de/LICENSE',
                     sha256='71b111620fa32c17a80ccd6db2da759d31a591e497d8b5f382174f59d1b59d49', quote='Apache License Version 2.0, January 2004')

# Provenance pins per arch (runtime pins -- timm name, repo, revision, weight SHA-256/size -- live in model_x.SPECS).
PROV = {
    's': dict(licence_texts=DINOV2_LIC, upstream=None, lineage='Meta DINOv2 ViT-S/14 (LVD-142M), timm copy of the official release'),
    'b': dict(licence_texts=DINOV2_LIC, upstream=None, lineage='Meta DINOv2 ViT-B/14 (LVD-142M), timm copy of the official release'),
    'l': dict(licence_texts=DINOV2_LIC, upstream=None, lineage='Meta DINOv2 ViT-L/14 (LVD-142M), timm copy of the official release'),
    'b_reg': dict(licence_texts=DINOV2_LIC, upstream=None,
                  lineage='Meta DINOv2 ViT-B/14 with 4 registers (LVD-142M; arXiv 2309.16588), timm copy of the official release'),
    'pe_s': dict(
        licence_texts=[
            dict(name='LICENSE.PE', url=f'{GH}/facebookresearch/perception_models/3e352cca660658d4b5c90f42a7808b11469e4c66/LICENSE.PE',
                 sha256='c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4', quote='Apache License Version 2.0, January 2004'),
            dict(name='README.pe.md', url=f'{GH}/facebookresearch/perception_models/3e352cca660658d4b5c90f42a7808b11469e4c66/apps/pe/README.md',
                 sha256='61ff35d490199b1328fca9eaac3461a873f9d7dd03c7e72a6814cff2cc5d6bfb',
                 quote='All checkpoints released on this page, unless otherwise specified, are released with the [Apache 2.0 license]'),
        ],
        upstream=dict(repo='facebook/PE-Spatial-S16-512', revision='3a1d1419e8b19ba1c473a6e9b095e78699ddd7a7', file='PE-Spatial-S16-512.pt',
                      sha256='2d8d63fdd816555cd223e4d535ecde432250846cf4381675b3f31f29cab893a6', size=87981467, fmt='torch',
                      filter='timm.models.eva:checkpoint_filter_fn'),
        lineage='Meta Perception Encoder PE-Spatial S/16 @512 (arXiv 2504.13181; distilled from PE-Spatial G/14; MetaCLIP-curated '
                '5.4B image-text pairs + SAM 2.1 mask teacher), timm conversion of facebook/PE-Spatial-S16-512',
        notes=['timm config.json pretrained_cfg.license reads "custom" (timm hub placeholder); the HF licence tag, the model-card '
               'YAML, the upstream facebook repo tag and the PE README all state Apache-2.0.']),
    'tips_b': dict(
        licence_texts=[
            dict(name='LICENSE.tips', url=f'{GH}/google-deepmind/tips/bf10a73765b048edfbfcdf1f7e6ca8292542baae/LICENSE',
                 sha256='cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30', quote='Apache License Version 2.0, January 2004'),
            dict(name='README.tips.md', url=f'{GH}/google-deepmind/tips/bf10a73765b048edfbfcdf1f7e6ca8292542baae/README.md',
                 sha256='61ddf33ddf4c88b2460168930697bf0c52ccc4b9c1b1ee5fe470120f64fb954d',
                 quote='All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except '
                       'in compliance with the Apache 2.0 license.'),
        ],
        upstream=dict(repo='google/tipsv2-b14', revision='ed1e4dc6b74bf3935ae099e9d5eb30fa96528454', file='model.safetensors',
                      sha256='c19aec1a777c381cc3d351b4882c56fc3041505c8f5b97535816a92dce3d20c5', size=783830160, fmt='safetensors',
                      filter='timm.models.vision_transformer:checkpoint_filter_fn'),
        lineage='Google DeepMind TIPSv2 B/14 (arXiv 2604.12012; distilled from TIPSv2 g/14; WebLI 116M filtered subset with '
                'PaliGemma / Gemini 1.5 Flash synthetic captions), timm conversion of google/tipsv2-b14 vision encoder',
        notes=['google-deepmind/tips README: software Apache-2.0, "All other materials" CC-BY-4.0; if the checkpoint counts as '
               'material, CC-BY-4.0 (commercial use and redistribution with attribution) applies. Credit Google in NOTICE either way.']),
}
WANT_LICENCE = 'apache-2.0'


class Refuse(Exception):
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git_blob_id(b):
    return hashlib.sha1(b'blob %d\0' % len(b) + b).hexdigest()


def norm_ws(s):
    return ' '.join(s.split())


def http(url, dest=None, retries=4, timeout=120):
    """GET -> bytes, or stream to dest (returns (sha256, size)). 4xx (and file:// misses) are not retried."""
    retries = 1 if url.startswith('file:') else retries
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                if dest is None:
                    return r.read()
                h, n = hashlib.sha256(), 0
                with open(dest, 'wb') as f:
                    while True:
                        b = r.read(1 << 22)
                        if not b:
                            break
                        f.write(b); h.update(b); n += len(b)
                    f.flush(); os.fsync(f.fileno())
                return h.hexdigest(), n
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 or i == retries - 1:
                raise Refuse(f'HTTP {e.code} for {url}')
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if i == retries - 1:
                raise Refuse(f'network error for {url}: {e}')
        time.sleep(2 ** i)


def endpoint(a):
    return (getattr(a, 'endpoint', None) or os.environ.get('HF_ENDPOINT') or 'https://huggingface.co').rstrip('/')


def card_licence(text):
    m = re.match(r'^---\s*\n(.*?)\n---', text, re.S)
    if not m:
        return None
    for line in m.group(1).splitlines():
        if line.strip().startswith('license:'):
            return line.split(':', 1)[1].strip().strip('"\'')
    return None


def dest_dir(arch, root):
    return Path(root) / Path(mx.SPECS[arch].weights).relative_to(mx.EXT0).parent


def atomic_write(path, data):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- checks
def hub_check(arch, ep, repo=None, revision=None):
    """Hub API at the pinned revision -> (info, tree, raw bytes, head sha). Refuses on any pin/licence/gating mismatch."""
    sp = mx.SPECS[arch]; repo = repo or sp.repo; revision = revision or sp.revision
    raw_info = http(f'{ep}/api/models/{repo}/revision/{revision}')
    raw_tree = http(f'{ep}/api/models/{repo}/tree/{revision}')
    info, tree = json.loads(raw_info), json.loads(raw_tree)
    if info.get('sha') != revision:
        raise Refuse(f'{repo}: revision {revision} resolves to {info.get("sha")}')
    if info.get('gated') not in (False, None) or info.get('private'):
        raise Refuse(f'{repo}: gated={info.get("gated")} private={info.get("private")}')
    tags = [t for t in info.get('tags', []) if t.startswith('license:')]
    card_lic = (info.get('cardData') or {}).get('license')
    if tags != [f'license:{WANT_LICENCE}'] or card_lic != WANT_LICENCE:
        raise Refuse(f'{repo}: licence tags {tags}, cardData {card_lic!r} (want {WANT_LICENCE})')
    head = None
    try:
        head = json.loads(http(f'{ep}/api/models/{repo}')).get('sha')
    except Refuse:
        pass
    return info, {e['path']: e for e in tree if e.get('type', 'file') == 'file'}, raw_info, raw_tree, head


def expect_lfs(tree, fname, sha, size, repo):
    e = tree.get(fname)
    if e is None:
        raise Refuse(f'{repo}: {fname} not listed at the pinned revision')
    lfs = e.get('lfs') or {}
    if lfs.get('oid') != sha or int(e.get('size', -1)) != size:
        raise Refuse(f'{repo}: {fname} Hub LFS sha/size {lfs.get("oid")}/{e.get("size")} != pin {sha}/{size}')
    return e


def licence_texts(arch, save_dir=None):
    out = []
    for L in PROV[arch]['licence_texts'] + [TIMM_CODE_LIC]:
        b = http(L['url']); h = sha256_bytes(b)
        if h != L['sha256']:
            raise Refuse(f'{arch}: licence text {L["url"]} sha256 {h} != pin {L["sha256"]}')
        if norm_ws(L['quote']) not in norm_ws(b.decode('utf-8', 'replace')):
            raise Refuse(f'{arch}: licence statement not found in {L["url"]}: {L["quote"]!r}')
        rec = dict(name=L['name'], url=L['url'], sha256=h, size=len(b), quote=L['quote'])
        if save_dir is not None:
            atomic_write(Path(save_dir) / L['name'], b); rec['saved_as'] = L['name']
        out.append(rec)
    return out


def small_file(ep, repo, rev, tree, fname, dest=None):
    """Non-LFS file (card, config): bytes must hash to the git blob id listed at the revision."""
    e = tree.get(fname)
    if e is None:
        raise Refuse(f'{repo}: {fname} not listed at {rev}')
    if dest is not None and dest.exists():
        b = dest.read_bytes()
        if git_blob_id(b) != e.get('oid'):
            raise Refuse(f'{dest} exists and differs from {repo}@{rev}:{fname}; refusing to overwrite')
    else:
        b = http(f'{ep}/{repo}/resolve/{rev}/{fname}')
        if git_blob_id(b) != e.get('oid'):
            raise Refuse(f'{repo}: {fname} git blob id {git_blob_id(b)} != Hub {e.get("oid")}')
        if dest is not None:
            atomic_write(dest, b)
    return b, dict(path=fname, size=len(b), sha256=sha256_bytes(b), git_blob_id=e.get('oid'),
                   url=f'{ep}/{repo}/resolve/{rev}/{fname}')


def big_file(ep, repo, rev, fname, sha, size, dest):
    """LFS file streamed to dest.part; size + sha256 must equal the pin; reuse an existing dest only if it verifies."""
    url = f'{ep}/{repo}/resolve/{rev}/{fname}'
    if dest.exists():
        if dest.stat().st_size != size or mx.sha256_file(dest) != sha:
            raise Refuse(f'{dest} exists and does not match the pin; refusing to overwrite')
        return dict(path=dest.name, size=size, sha256=sha, expected_sha256=sha, url=url, reused_existing=True)
    part = Path(str(dest) + '.part')
    try:
        h, n = http(url, part)
        if h != sha or n != size:
            raise Refuse(f'{repo}: downloaded {fname} sha256/size {h}/{n} != pin {sha}/{size}')
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()
    return dict(path=dest.name, size=n, sha256=h, expected_sha256=sha, url=url, reused_existing=False)


def strict_load(arch, weights):
    """Load through timm at the native resolution (no pos-embed resample), then require the model's state dict to have
    exactly the file's keys and bit-identical tensors (independent of timm's own strict flag)."""
    import timm, torch
    from safetensors.torch import load_file
    sp = mx.SPECS[arch]
    native = timm.models.get_pretrained_cfg(sp.timm).input_size[-1]
    m = timm.create_model(sp.timm, pretrained=True, num_classes=0, img_size=native, pretrained_cfg_overlay=dict(file=str(weights)),
                          **dict(sp.kwargs))
    ref, sd = load_file(str(weights)), m.state_dict()
    missing, unexpected = sorted(set(sd) - set(ref)), sorted(set(ref) - set(sd))
    if missing or unexpected or not all(torch.equal(sd[k], ref[k].to(sd[k].dtype)) for k in ref):
        raise Refuse(f'{arch}: weights do not load bit-exactly into {sp.timm}: missing {missing[:5]}, unexpected {unexpected[:5]}')
    src = inspect.getsourcefile(type(m))
    return dict(ok=True, tensors=len(ref), native_px=native, params=int(sum(p.numel() for p in m.parameters())), timm_class=type(m).__name__,
                timm_version=timm.__version__, torch_version=torch.__version__,
                timm_source=str(Path(src).relative_to(Path(timm.__file__).parent.parent)), timm_source_sha256=mx.sha256_file(src))


def upstream_parity(arch, ep, weights, work_dir, keep=False):
    """Original release -> timm's own filter -> must equal the timm copy bit for bit."""
    import importlib, timm, torch
    from safetensors.torch import load_file
    u = PROV[arch]['upstream']; sp = mx.SPECS[arch]
    info, tree, raw_info, raw_tree, head = hub_check(arch, ep, u['repo'], u['revision'])
    expect_lfs(tree, u['file'], u['sha256'], u['size'], u['repo'])
    card, card_rec = small_file(ep, u['repo'], u['revision'], tree, 'README.md')
    up_dir = Path(work_dir) / 'upstream'; up_dir.mkdir(exist_ok=True)
    f = up_dir / Path(u['file']).name
    rec = big_file(ep, u['repo'], u['revision'], u['file'], u['sha256'], u['size'], f)
    try:
        sd = torch.load(f, map_location='cpu', weights_only=True) if u['fmt'] == 'torch' else load_file(str(f))
        mod, fn = u['filter'].split(':')
        native = timm.models.get_pretrained_cfg(sp.timm).input_size[-1]
        model = timm.create_model(sp.timm, pretrained=False, num_classes=0, img_size=native)
        conv = getattr(importlib.import_module(mod), fn)(sd, model)
        ref = load_file(str(weights))
        ka, kb = set(conv), set(ref)
        diff = 0.0; shapes_ok = True
        for k in sorted(ka & kb):
            a, b = conv[k], ref[k]
            if tuple(a.shape) != tuple(b.shape):
                shapes_ok = False; continue
            diff = max(diff, float((a.to(b.dtype).float() - b.float()).abs().max()) if a.numel() else 0.0)
        res = dict(repo=u['repo'], revision=u['revision'], head_revision_at_access=head, api_revision_json_sha256=sha256_bytes(raw_info),
                   card=dict(url=f'https://huggingface.co/{u["repo"]}/blob/{u["revision"]}/README.md', sha256=card_rec['sha256'],
                             yaml_licence=card_licence(card.decode('utf-8', 'replace'))),
                   file=rec, converter=f'{u["filter"]} (timm {timm.__version__})', tensors_compared=len(ka & kb),
                   keys_only_in_upstream_conversion=sorted(ka - kb), keys_only_in_timm_copy=sorted(kb - ka),
                   shapes_equal=shapes_ok, max_abs_diff=diff)
        res['identical'] = not (ka ^ kb) and shapes_ok and diff == 0.0
        if not res['identical']:
            raise Refuse(f'{arch}: upstream conversion differs from the timm copy: {json.dumps({k: res[k] for k in ("keys_only_in_upstream_conversion", "keys_only_in_timm_copy", "shapes_equal", "max_abs_diff")})}')
        return res
    finally:
        if not keep and f.exists():
            f.unlink()
        if up_dir.exists() and not any(up_dir.iterdir()):
            up_dir.rmdir()


# ---------------------------------------------------------------- commands
def do_check(arch, a):
    sp = mx.SPECS[arch]; ep = endpoint(a)
    info, tree, raw_info, raw_tree, head = hub_check(arch, ep)
    w = expect_lfs(tree, Path(sp.weights).name, sp.sha256, sp.size, sp.repo)
    card, card_rec = small_file(ep, sp.repo, sp.revision, tree, 'README.md')
    if card_licence(card.decode('utf-8', 'replace')) != WANT_LICENCE:
        raise Refuse(f'{sp.repo}: model-card YAML licence {card_licence(card.decode())!r}')
    lic = licence_texts(arch)
    u = PROV[arch]['upstream']; up = None
    if u:
        uinfo, utree, _, _, uhead = hub_check(arch, ep, u['repo'], u['revision'])
        expect_lfs(utree, u['file'], u['sha256'], u['size'], u['repo'])
        up = dict(repo=u['repo'], revision=u['revision'], head=uhead, file=u['file'], lfs_ok=True)
    return dict(arch=arch, repo=sp.repo, revision=sp.revision, head_revision=head, pinned_is_head=head == sp.revision,
                last_modified=info.get('lastModified'), weights_lfs_sha256=w['lfs']['oid'], weights_size=w['size'],
                card_sha256=card_rec['sha256'], licence_texts=[(L['name'], L['sha256']) for L in lic], upstream=up, status='OK')


def do_fetch(arch, a):
    sp = mx.SPECS[arch]; ep = endpoint(a); root = a.root or mx.EXT
    d = dest_dir(arch, root); d.mkdir(parents=True, exist_ok=True)
    t_access = now()
    info, tree, raw_info, raw_tree, head = hub_check(arch, ep)
    wname = Path(sp.weights).name
    expect_lfs(tree, wname, sp.sha256, sp.size, sp.repo)
    lic = licence_texts(arch, d)
    card, card_rec = small_file(ep, sp.repo, sp.revision, tree, 'README.md', d / 'README.md')
    if card_licence(card.decode('utf-8', 'replace')) != WANT_LICENCE:
        raise Refuse(f'{sp.repo}: model-card YAML licence {card_licence(card.decode())!r}')
    cfg, cfg_rec = small_file(ep, sp.repo, sp.revision, tree, 'config.json', d / 'config.json')
    wrec = big_file(ep, sp.repo, sp.revision, wname, sp.sha256, sp.size, d / wname)
    for nm, raw in ((f'hub-api.{arch}.json', raw_info), (f'hub-tree.{arch}.json', raw_tree)):
        atomic_write(d / nm, raw)
    load = strict_load(arch, d / wname)
    up = upstream_parity(arch, ep, d / wname, d, a.keep_upstream) if (a.verify_upstream and PROV[arch]['upstream']) else \
        dict(checked=False, reason='not requested' if PROV[arch]['upstream'] else 'no separate upstream pin (timm hub copy is the accepted source)')
    cfg_lic = (json.loads(cfg).get('pretrained_cfg') or {}).get('license')
    rec = dict(
        schema=SCHEMA, arch=arch, status=sp.status, timm_model=sp.timm, lineage=PROV[arch]['lineage'],
        input_normalisation=dict(mean=sp.mean, std=sp.std, note='rotlab.model_x maps the ImageNet-normalised pipeline to this in-model'),
        hub=dict(endpoint=ep, repo=sp.repo, revision_pinned=sp.revision, revision_resolved=info.get('sha'), head_revision_at_access=head,
                 last_modified=info.get('lastModified'), gated=info.get('gated'), private=info.get('private'),
                 licence_tags=[t for t in info.get('tags', []) if t.startswith('license:')], card_data_licence=(info.get('cardData') or {}).get('license'),
                 api_revision_json=dict(saved_as=f'hub-api.{arch}.json', sha256=sha256_bytes(raw_info)),
                 api_tree_json=dict(saved_as=f'hub-tree.{arch}.json', sha256=sha256_bytes(raw_tree))),
        files=[wrec, dict(card_rec, saved_as='README.md'), dict(cfg_rec, saved_as='config.json')],
        model_card=dict(url=f'https://huggingface.co/{sp.repo}/blob/{sp.revision}/README.md', sha256=card_rec['sha256'],
                        yaml_licence=card_licence(card.decode('utf-8', 'replace'))),
        config_pretrained_cfg_licence=cfg_lic, licence_texts=lic, notes=PROV[arch].get('notes', []),
        upstream=up, strict_load=load,
        tool=dict(fetch_weights_sha256=mx.sha256_file(__file__), model_x_sha256=mx.sha256_file(mx.__file__), python=sys.version.split()[0]),
        accessed_utc=t_access, completed_utc=now(), host=socket.gethostname())
    out = d / f'PROVENANCE.{arch}.json'
    if out.exists():
        out = d / f'PROVENANCE.{arch}.{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json'
    data = json.dumps(rec, indent=1).encode()
    atomic_write(out, data); atomic_write(Path(str(out) + '.sha256'), f'{sha256_bytes(data)}  {out.name}\n'.encode())
    return dict(arch=arch, status='OK', dir=str(d), receipt=out.name, receipt_sha256=sha256_bytes(data), weights_sha256=wrec['sha256'],
                upstream_identical=up.get('identical'), strict_load=load['ok'])


def do_verify(arch, a):
    sp = mx.SPECS[arch]; d = dest_dir(arch, a.root or mx.EXT); w = d / Path(sp.weights).name
    if not w.is_file() or w.stat().st_size != sp.size or mx.sha256_file(w) != sp.sha256:
        raise Refuse(f'{w}: missing or not the pinned file')
    rs = sorted(d.glob(f'PROVENANCE.{arch}*.json'))
    if not rs:
        raise Refuse(f'{d}: no PROVENANCE.{arch}*.json receipt')
    for r in rs:
        side = Path(str(r) + '.sha256')
        if not side.exists() or side.read_text().split()[0] != mx.sha256_file(r):
            raise Refuse(f'{r}: receipt sidecar missing or mismatched')
        rec = json.loads(r.read_text())
        if rec['files'][0]['sha256'] != sp.sha256 or rec['hub']['revision_pinned'] != sp.revision:
            raise Refuse(f'{r}: receipt does not bind the current pin')
    for L in PROV[arch]['licence_texts'] + [TIMM_CODE_LIC]:
        f = d / L['name']
        if not f.exists() or mx.sha256_file(f) != L['sha256']:
            raise Refuse(f'{f}: licence text missing or changed')
    return dict(arch=arch, status='OK', receipts=[r.name for r in rs], strict_load=strict_load(arch, w)['ok'])


def cmd_pins(a):
    print('| key | status | timm model | HF repo @ revision | weights SHA-256 | bytes | input mean/std | upstream original |')
    print('|---|---|---|---|---|---|---|---|')
    for k, sp in mx.SPECS.items():
        u = PROV[k]['upstream']
        print(f'| {k} | {sp.status} | `{sp.timm}` | `{sp.repo}` @ `{sp.revision[:12]}` | `{sp.sha256[:16]}…` | {sp.size} | '
              f'{sp.mean}/{sp.std} | {"`" + u["repo"] + "` @ `" + u["revision"][:12] + "`" if u else "-"} |')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='rotlab.fetch_weights', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['pins', 'check', 'fetch', 'verify']); ap.add_argument('archs', nargs='*')
    ap.add_argument('--root', help=f'weights root (default {mx.EXT})'); ap.add_argument('--endpoint', help='Hub endpoint (default https://huggingface.co)')
    ap.add_argument('--verify-upstream', action='store_true'); ap.add_argument('--keep-upstream', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'pins':
        return cmd_pins(a)
    bad = [k for k in a.archs if k not in mx.SPECS]
    if bad or not a.archs:
        raise SystemExit(f'archs required, from {sorted(mx.SPECS)}; unknown: {bad}')
    fn = dict(check=do_check, fetch=do_fetch, verify=do_verify)[a.cmd]; rc = 0
    for k in a.archs:
        try:
            print(json.dumps(fn(k, a)), flush=True)
        except Refuse as e:
            print(f'REFUSE {k}: {e}', flush=True); rc = 2
    if rc:
        sys.exit(rc)


if __name__ == '__main__':
    main()
