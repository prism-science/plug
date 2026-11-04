'''
Fetch pdb entries from rcsb and keep the one chain a dataset entity is about.

An entry is a complex and only one of its chains is the protein a row scores,
so that chain is written out on its own. raw/ holds the downloaded entry and
chains/ the extracted chain; both are keyed by content and never rewritten, so
a graph rebuilt tomorrow compares the same atoms it compared today.
'''

import gzip
import json
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from . import config as c
from .sifts import THREE as _THREE

# the standard twenty plus the modified residues that turn up as HETATM in real entries.
# selenomethionine is the common one: drop it and a whole chain reads as gaps.
THREE = {**_THREE, "MSE": "M", "SEC": "U", "PYL": "O"}

# past this length the chain picker stops aligning and compares lengths instead. difflib
# is quadratic and a handful of entries in pdbbind are ribosomes.
MAX_ALIGN = 3000

def _say(verbose, *a):
    if verbose: print(*a, flush=True)

def fetch(pdb_id, tries=3):
    '''
    one entry as text, from the cache if it is there and from rcsb if it is not.
    the legacy pdb file is preferred because it is smaller; entries too large for
    that format only exist as mmcif and come back as one.
    '''
    pdb_id = pdb_id.strip().lower()
    raw = c.rcsb_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for ext, kind in ((".pdb.gz", "pdb"), (".cif.gz", "cif")):
        dst = raw / f"{pdb_id}{ext}"
        if dst.exists() and dst.stat().st_size:
            try:
                return kind, gzip.decompress(dst.read_bytes()).decode("utf-8", "ignore")
            except (OSError, EOFError):
                # a download cut off mid-stream leaves a file that is there but not gzip
                dst.unlink()
    for ext, kind in ((".pdb.gz", "pdb"), (".cif.gz", "cif")):
        url = f"https://files.rcsb.org/download/{pdb_id}{ext}"
        for _ in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    blob = r.read()
            except urllib.error.HTTPError as e:
                # 404 means this format does not exist for this entry, so try the next one
                if e.code == 404: break
                continue
            except Exception:
                continue
            try:
                text = gzip.decompress(blob).decode("utf-8", "ignore")
            except (OSError, EOFError):
                continue
            # written only once it has decompressed, so the cache never holds a stub
            (raw / f"{pdb_id}{ext}").write_bytes(blob)
            return kind, text
    return None, None

def chains_from_pdb(text):
    '''
    every chain of a legacy pdb entry, as its observed sequence and its atom
    lines. only the first model is read, and only the first altloc, so an nmr
    ensemble and a disordered side chain each come back as one structure.
    '''
    seqs, lines, seen = defaultdict(list), defaultdict(list), set()
    for ln in text.splitlines():
        if ln.startswith("ENDMDL"): break
        if not ln.startswith(("ATOM  ", "HETATM")): continue
        if ln[16] not in (" ", "A"): continue
        ch, res = ln[21], ln[17:20].strip().upper()
        # waters, ions and the ligand itself are not part of the fold
        if ln.startswith("HETATM") and res not in THREE: continue
        lines[ch].append(ln)
        rid = ln[22:27]
        if ln[12:16].strip() == "CA" and (ch, rid) not in seen:
            seen.add((ch, rid))
            seqs[ch].append(THREE.get(res, "X"))
    return {k: ("".join(seqs[k]), lines[k]) for k in lines if seqs[k]}

def _atom_site(text):
    '''
    the atom_site loop of an mmcif entry, as column names and rows. written out
    rather than pulled from a parser so plug keeps no mmcif dependency.
    '''
    it = iter(text.splitlines())
    cols, rows = [], []
    # the column names, which run until the first line that is not one of them
    for ln in it:
        if ln.startswith("_atom_site."):
            cols.append(ln.strip().split(".", 1)[1])
            break
    if not cols: return [], []
    for ln in it:
        if ln.startswith("_atom_site."):
            cols.append(ln.strip().split(".", 1)[1])
            continue
        # the loop ends at the next category or the blank line before it. reading past it
        # would take the first line of whatever comes next as an atom
        if ln.startswith("#") or not ln.strip(): break
        rows.append(ln.split())
    return cols, rows

def chains_from_cif(text):
    '''
    the same for mmcif, read by column name and written back out as pdb atom lines
    so foldseek sees one format however the entry arrived. residue and atom
    numbers are renumbered per chain, since a chain lifted out of a large entry
    can carry numbers too wide for the columns.
    '''
    cols, rows = _atom_site(text)
    if not cols or not rows: return {}
    ix = {k: i for i, k in enumerate(cols)}
    need = ("group_PDB", "label_atom_id", "label_comp_id", "auth_asym_id", "auth_seq_id",
            "Cartn_x", "Cartn_y", "Cartn_z", "label_alt_id", "pdbx_PDB_model_num")
    if any(k not in ix for k in need): return {}
    wide = max(ix[k] for k in need)
    seqs, lines, seen, model0 = defaultdict(list), defaultdict(list), set(), None
    for r in rows:
        if len(r) <= wide: continue
        m = r[ix["pdbx_PDB_model_num"]]
        if model0 is None: model0 = m
        if m != model0: continue
        if r[ix["label_alt_id"]] not in (".", "?", "A"): continue
        res, ch = r[ix["label_comp_id"]].upper(), r[ix["auth_asym_id"]]
        if r[ix["group_PDB"]] == "HETATM" and res not in THREE: continue
        atom, rid = r[ix["label_atom_id"]].strip('"'), r[ix["auth_seq_id"]]
        try:
            x, y, z = (float(r[ix[k]]) for k in ("Cartn_x", "Cartn_y", "Cartn_z"))
        except ValueError:
            continue
        if atom == "CA" and (ch, rid) not in seen:
            seen.add((ch, rid))
            seqs[ch].append(THREE.get(res, "X"))
        n = len(lines[ch]) + 1
        try: num = int(float(rid))
        except ValueError: num = len(seen)
        lines[ch].append(
            f"ATOM  {n % 100000:5d} {atom[:4]:^4s}{res[:3]:>3s} {ch[0]}{num % 10000:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
    return {k: ("".join(seqs[k]), lines[k]) for k in lines if seqs[k]}

def pick(chains, seq):
    '''
    which chain of an entry the benchmark is about: the one whose observed
    sequence best matches the sequence the benchmark carries. an entry with one
    chain needs no deciding and skips the alignment.
    '''
    if len(chains) == 1:
        k = next(iter(chains))
        return k, 1.0
    best, score = None, -1.0
    for k, (cseq, _) in chains.items():
        if not seq or max(len(seq), len(cseq)) > MAX_ALIGN:
            # too long to align: fall back to how close the lengths are
            r = min(len(seq), len(cseq)) / max(len(seq), len(cseq), 1)
        else:
            r = SequenceMatcher(None, seq, cseq, autojunk=False).ratio()
        if r > score: best, score = k, r
    return best, score

def _slug(key):
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(key))

def chain_file(bench, key, pdb_id, seq):
    '''
    the path to this entity's single chain structure, extracting it first if it is
    not already cached.
    '''
    out = c.rcsb_dir / "chains" / f"{bench}__{_slug(key)}.pdb"
    if out.exists() and out.stat().st_size: return out, "cached"
    kind, text = fetch(pdb_id)
    if text is None: return None, "no entry"
    chains = chains_from_pdb(text) if kind == "pdb" else chains_from_cif(text)
    if not chains: return None, f"no chains ({kind})"
    ch, score = pick(chains, seq)
    out.parent.mkdir(parents=True, exist_ok=True)
    # written to a neighbour and moved, so a killed run never leaves half a structure
    tmp = out.with_suffix(f".{_slug(pdb_id)}.part")
    tmp.write_text("\n".join(chains[ch][1]) + "\nEND\n")
    tmp.replace(out)
    return out, f"ok {ch} {score:.2f}"

def entities(cls):
    '''
    what each of this benchmark's entities points at. the first row naming an
    entity decides, since every row about one protein carries the same entry and
    the same sequence.
    '''
    if not isinstance(cls.key, str):
        raise TypeError(f"{cls.name} is keyed on a pair, which has no single entry")
    out = {}
    for it in cls.rows():
        k = it[cls.key]
        if k in out: continue
        pdb = str(it.get("pdb") or "").strip().lower()
        if pdb: out[k] = (pdb, str(it[cls.seq]).strip().upper())
    return out

def structures(cls, workers=None, verbose=True):
    '''
    one structure file per entity of a benchmark that names pdb entries, ready for
    the structure view. downloads run in parallel and everything is cached, so the
    first call is slow and every call after it is not. a manifest of what resolved
    is written beside the chains, and a shortfall raises rather than returning a
    graph that is checked on shape in some places and not others.
    '''
    ents = entities(cls)
    if not ents:
        raise FileNotFoundError(f"{cls.name} names no pdb entries, so it has no structures")
    c.rcsb_dir.mkdir(parents=True, exist_ok=True)
    _say(verbose, f"  {cls.name}: {len(ents):,} entities over "
                  f"{len({p for p, _ in ents.values()}):,} pdb entries -> {c.rcsb_dir}")

    jobs = [(k, p, s) for k, (p, s) in ents.items()]
    paths, why, fetched = {}, {}, 0
    with ThreadPoolExecutor(max_workers=workers or c.rcsb_workers) as pool:
        run = lambda j: (j[0], *chain_file(cls.name, j[0], j[1], j[2]))
        for i, (k, path, state) in enumerate(pool.map(run, jobs), 1):
            if path is None: why[k] = state
            else:
                paths[k] = path
                fetched += state != "cached"
            if verbose and i % 500 == 0:
                _say(True, f"    ... {i:,}/{len(jobs):,} entities, {len(why):,} unresolved")

    (c.rcsb_dir / f"{cls.name}_manifest.json").write_text(json.dumps(
        {"benchmark": cls.name, "entities": len(ents), "resolved": len(paths),
         "newly_extracted": fetched, "missing": why}, indent=1, sort_keys=True) + "\n")
    cov = len(paths) / len(ents)
    _say(verbose, f"  {cls.name}: {len(paths):,}/{len(ents):,} structures "
                  f"({cov:.3%}), {fetched:,} newly extracted")
    if cov < c.rcsb_min_cov:
        raise FileNotFoundError(
            f"{cls.name}: only {len(paths):,} of {len(ents):,} entities resolved to a "
            f"structure ({cov:.3%} < config.rcsb_min_cov={c.rcsb_min_cov}). the unresolved ones "
            f"would be checked on sequence alone while the rest are checked on both, so "
            f"the graph is not what its name says. see "
            f"{c.rcsb_dir / f'{cls.name}_manifest.json'} for which entries failed, and "
            f"lower config.rcsb_min_cov on purpose if the shortfall is obsolete entries.")
    return paths
