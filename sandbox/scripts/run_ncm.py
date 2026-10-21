"""Device-generic NCM generation + scoring (CPU / MPS / CUDA), single process.

A port of scripts/eval_shard.py + scripts/eval_merge.py that takes --device
instead of assuming CUDA, imports the MPS area-interpolate shim, keeps the
float64 frequency-penalty sampler on CPU, and places the reference LMs used
for perplexity on the chosen device. Output pickle has the same keys as the
upstream shard files, so scripts/eval_merge.py can also consume it.

Examples (from the repo root or anywhere; NCM_ROOT defaults to ../.. of here):
  python sandbox/scripts/run_ncm.py --config config/eval/sudoku-easy.yaml \
      --checkpoint-dir sandbox/artifacts/sudoku/easy --data-dir sandbox/data/sudoku/easy \
      --device mps --n 200
  python sandbox/scripts/run_ncm.py --config config/eval/tinystories.yaml \
      --checkpoint-dir sandbox/artifacts/tinystories --device mps --n 16 --seeds 0 --print 4
  python sandbox/scripts/run_ncm.py --config ... --score-only sandbox/eval/sudoku-easy/mps.pkl
"""
import argparse, math, os, pickle, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.environ.get("NCM_ROOT", os.path.join(HERE, "..", "..")))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import torch
import mps_compat  # noqa: F401  MPS shim; no-op on CPU/CUDA
from omegaconf import OmegaConf


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------- scoring

def score(CFG, out, dev):
    hyps, refs = out["hyps"], out["refs"]
    metric = str(CFG.eval.metric)
    print(f"\n{len(hyps)} {CFG.task} outputs, metric={metric}, device={dev}\n")

    if metric == "exact":
        def cells(t):
            return t.replace("|", "")

        def valid(t):
            g = cells(t)
            if len(g) != 81 or not g.isdigit() or "0" in g:
                return False
            rows_ = [g[i * 9:(i + 1) * 9] for i in range(9)]
            cols_ = ["".join(r[j] for r in rows_) for j in range(9)]
            boxes = ["".join(rows_[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3))
                     for br in (0, 3, 6) for bc in (0, 3, 6)]
            return all(sorted(x) == list("123456789") for x in rows_ + cols_ + boxes)

        n = len(hyps)
        exact_n = sum(h == r for h, r in zip(hyps, refs))
        cell = np.mean([np.mean([a == b for a, b in zip(cells(h), cells(r))]) for h, r in zip(hyps, refs)])
        valid_n = sum(valid(h) for h in hyps)
        print(f"{'solved %':>9} {'solved':>7} {'cell %':>7} {'valid %':>8}")
        print(f"{100 * exact_n / n:9.2f} {exact_n:4d}/{n} {100 * cell:7.2f} {100 * valid_n / n:8.2f}")
        return {"solved": exact_n / n, "cell": float(cell), "valid": valid_n / n}

    if metric == "bleu":
        import sacrebleu
        b = sacrebleu.corpus_bleu(hyps, [refs])
        bl = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True)
        c = sacrebleu.corpus_chrf(hyps, [refs])
        lr = sum(len(h.split()) for h in hyps) / max(sum(len(r.split()) for r in refs), 1)
        print(f"{'BLEU':>8} {'BLEU(lc)':>9} {'chrF':>7} {'len ratio':>10}")
        print(f"{b.score:8.2f} {bl.score:9.2f} {c.score:7.2f} {lr:10.3f}")
        return {"bleu": b.score, "bleu_lc": bl.score, "chrf": c.score, "len_ratio": lr}

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if metric == "rouge":
        from rouge_score import rouge_scorer
        from evaluation.gen_ppl import generation_perplexity
        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        g2 = AutoTokenizer.from_pretrained("gpt2")
        lm = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).eval()
        R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
        for h, r in zip(hyps, refs):
            v = sc.score(r, h)
            for k in R:
                R[k].append(v[k].fmeasure)
        with torch.no_grad():
            gp = generation_perplexity(lm, g2, [""] * len(hyps), hyps, device=dev,
                                       score_only_generated=True, batch_size=32).mean_ppl
        print(f"{'R-1':>7} {'R-2':>7} {'R-L':>7} {'genPPL':>8} {'words':>7}")
        print(f"{100 * np.mean(R['rouge1']):7.2f} {100 * np.mean(R['rouge2']):7.2f} {100 * np.mean(R['rougeL']):7.2f} "
              f"{gp:8.2f} {np.mean([len(h.split()) for h in hyps]):7.1f}")
        return {k: float(np.mean(v)) for k, v in R.items()} | {"genppl": gp}

    seeds, toks, raw_ids = out.get("seeds", []), out.get("toks", []), out.get("raw_ids", [])
    ref = str(CFG.eval.ref_model)
    rtok = AutoTokenizer.from_pretrained(ref)
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rlm = AutoModelForCausalLM.from_pretrained(ref).to(dev).eval()

    if metric == "genppl":
        from evaluation.gen_ppl import generation_perplexity

        def entropy(ids):
            c, m = Counter(ids), len(ids)
            return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0

        per = []
        for seed in dict.fromkeys(seeds):
            sel = [k for k, s_ in enumerate(seeds) if s_ == seed]
            gen = [hyps[k] for k in sel]
            with torch.no_grad():
                ppl = generation_perplexity(rlm, rtok, [""] * len(gen), gen, device=dev, batch_size=32).mean_ppl
            ent = float(np.mean([entropy(toks[k]) for k in sel if toks[k]]))
            per.append((seed, ppl, ent))
    elif metric == "genppl_flm":
        import torch.nn.functional as F
        gen_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id))

        def corpus_ppl(texts):
            enc = rtok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True, max_length=128)
            ids, attn = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
            nll_sum = tok_sum = 0.0
            with torch.no_grad():
                for i in range(0, ids.shape[0], 16):
                    s_, m_ = ids[i:i + 16], attn[i:i + 16]
                    logits = rlm(s_, attention_mask=m_)[0].transpose(-1, -2)
                    nll = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
                    first_eos = (s_ == rtok.eos_token_id).cumsum(-1) == 1
                    valid = (first_eos[..., 1:] + (s_ != rtok.eos_token_id)[..., 1:]).to(nll.dtype)
                    nll_sum += float((nll * valid).sum()); tok_sum += float(valid.sum())
            return float(np.exp(nll_sum / tok_sum))

        def entropy(ids):
            t = torch.tensor(ids)
            return float(torch.special.entr(torch.unique(t, return_counts=True)[1].float() / t.numel()).sum())

        per = []
        for seed in dict.fromkeys(seeds):
            sel = [k for k, s_ in enumerate(seeds) if s_ == seed]
            texts = gen_tok.batch_decode([raw_ids[k] for k in sel])
            per.append((seed, corpus_ppl(texts), float(np.mean([entropy(raw_ids[k]) for k in sel]))))
    else:
        sys.exit(f"unknown metric {metric}")

    print(f"{'seed':>5} {'genPPL':>8} {'entropy':>8}")
    for seed, ppl, ent in per:
        print(f"{seed:5d} {ppl:8.2f} {ent:8.4f}")
    g = np.array([p[1] for p in per]); e = np.array([p[2] for p in per])
    sd = g.std(ddof=1) if len(g) > 1 else 0.0
    print(f"\n{'mean':>5} {g.mean():8.2f} {e.mean():8.4f}   (+/- {sd:.2f} over {len(g)} seeds, {len(hyps)} samples)")
    return {"genppl": float(g.mean()), "entropy": float(e.mean()), "genppl_sd": float(sd)}


# ----------------------------------------------------------------------------- generation

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cpu, mps, cuda, cuda:1 ...")
    ap.add_argument("--split", default="")
    ap.add_argument("--n", type=int, default=0, help="first n inputs (conditional) or samples per seed (unconditional)")
    ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--seeds", default="", help="comma-separated seeds for unconditional tasks (default: config)")
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--freq", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--adj", default=None)
    ap.add_argument("--l0", default="", help="unigram (default), active, uniform, or smooth:<eps>")
    ap.add_argument("--out", default="", help="output pickle (default sandbox/eval/<task>/<device>.pkl)")
    ap.add_argument("--print", type=int, default=0, help="print the first k outputs")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--score-only", default="", help="skip generation and score this pickle")
    a = ap.parse_args()

    CFG = OmegaConf.load(a.config)
    dev = pick_device(a.device)
    out_path = a.out or os.path.join(HERE, "..", "eval", str(CFG.task), f"{dev.replace(':', '')}.pkl")

    if a.score_only:
        out = pickle.load(open(a.score_only, "rb"))
        score(CFG, out, dev)
        return

    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    G = CFG.generation
    ck = a.checkpoint_dir or os.path.join(HERE, "..", "artifacts", *str(CFG.task).split("-"))
    dd = a.data_dir or os.path.join(HERE, "..", "data", *str(CFG.task).split("-"))
    pick = lambda cli, key, cast, default: cast(cli if cli is not None else G.get(key, default))
    temp = pick(a.temp, "temperature", float, 1.0)
    top_k = pick(a.top_k, "top_k", int, 1)
    top_p = pick(a.top_p, "top_p", float, 0.0)
    cfg = pick(a.cfg, "cfg_scale", float, 0.0)
    freq = pick(a.freq, "frequency_penalty", float, 0.0)
    mu = pick(a.mu, "frequency_deficit", float, 0.0)
    adj = str(a.adj if a.adj is not None else G.get("adjacency_penalty", "") or "")
    split = a.split or CFG.eval.get("split", "test")
    n = a.n or int(CFG.eval.get("n_docs", 0))
    bs = a.bs or int(CFG.eval.batch_size)

    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if CFG.checkpoints.get(name):
            CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
    t0 = time.time()
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.to(dev).eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(dev)
    sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()
    print(f"[run_ncm] {CFG.task} loaded on {dev} in {time.time() - t0:.1f}s "
          f"(temp={temp} top_k={top_k} top_p={top_p} cfg={cfg} freq={freq} mu={mu} adj={adj or '-'})", flush=True)
    P = int(CFG.model.prefix_len)
    EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id

    def load_allowance(spec):  # kept on CPU: only used by the float64 sampler below
        if ":" in spec:
            path, ref = spec.split(":", 1)
            obj = torch.load(os.path.join(ck, path), map_location="cpu", weights_only=False)
            for k in ref.split("."):
                obj = obj[k]
            return obj.double()
        return torch.load(os.path.join(ck, spec), map_location="cpu").double()

    ALLOW = load_allowance(str(G.frequency_allowance)) if G.get("frequency_allowance") else None
    L0_MODE = a.l0 or "unigram"
    L0_P = load_allowance(str(G.level0_unigram)).float() if G.get("level0_unigram") else None
    if L0_P is not None and L0_MODE == "active":
        L0_P = (L0_P > 0).float() / (L0_P > 0).sum()
    elif L0_P is not None and L0_MODE == "uniform":
        L0_P = torch.ones_like(L0_P) / L0_P.numel()
    elif L0_P is not None and L0_MODE.startswith("smooth:"):
        eps = float(L0_MODE.split(":", 1)[1])
        L0_P = (1.0 - eps) * L0_P + eps * torch.ones_like(L0_P) / L0_P.numel()
    UNCOND = not CFG.data or not CFG.data.get(split)
    ADJ = [float(v) for v in adj.split(",")] if adj else None

    if UNCOND:
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [int(x) for x in CFG.eval.seeds]
        n = a.n or int(CFG.eval.n_samples)
        items, refs, idxs, src_tok = [], [], [], None
    else:
        items = pickle.load(open(os.path.join(dd, CFG.data[split]), "rb"))[:n]
        refs = pickle.load(open(os.path.join(dd, CFG.data[f"{split}_refs"]), "rb"))[:n]
        assert len(items) == len(refs), f"{len(items)} inputs vs {len(refs)} references"
        idxs = list(range(len(items)))
        src_tok = None
        if isinstance(items[0], str):
            mid = CFG.tokenizer.pre_tokenizer.get("model_id")
            if mid:
                from transformers import AutoTokenizer
                src_tok = AutoTokenizer.from_pretrained(str(mid))
            else:
                src_tok = lambda s: {"input_ids": tk.pre_tokenizer.encode(s)}

    def prefix(item):
        ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
        return (ids[:P] + [PAD] * P)[:P]

    def pick_tokens(logits):
        if freq <= 0 and not ADJ:
            return logits.argmax(-1)
        B, T, V = logits.shape
        lg = logits.detach().cpu().double()  # MPS has no float64; this loop is cheap on CPU
        allow = (ALLOW[:V] * T)[None, :] if ALLOW is not None else None
        al = ADJ[0] if ADJ else 0.0
        aw = int(ADJ[1]) if ADJ and len(ADJ) > 1 else 1
        ad = ADJ[2] if ADJ and len(ADJ) > 2 else 1.0
        cnt = torch.zeros(B, V, dtype=torch.float64)
        out = torch.empty(B, T, dtype=torch.long)
        hist = []
        for t in range(T):
            step = lg[:, t, :].clone()
            if freq > 0:
                excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
                step -= freq * excess
                if mu > 0 and allow is not None:
                    step += mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
            for j, h in enumerate(reversed(hist[-aw:])):
                step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64))
            p = step.argmax(-1)
            out[:, t] = p
            hist.append(p)
            cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64))
        return out

    CUT_AT_EOS = bool(G.get("cut_at_eos", True))

    def dec(ids):
        ids = list(ids)
        if CUT_AT_EOS and EOS is not None and EOS in ids:
            ids = ids[:ids.index(EOS)]
        return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)

    RAW = []

    def run(x, l0=None):
        with torch.no_grad():
            g = net.generate(l0, x, temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=cfg,
                             cfg_null_eos=EOS if EOS is not None else 0)
            _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
            ids = pick_tokens(tk.vqvae.decoder(zl[-1])).tolist()
        RAW.extend(ids)
        return [dec(r) for r in ids]

    hyps, sample_seed = [], []
    t0 = time.time()
    if UNCOND:
        L0 = tk.vqvae.vq.levels[0]
        for seed in seeds:
            torch.manual_seed(seed)
            for s in range(0, n, bs):
                B = min(bs, n - s)
                l0 = (torch.multinomial(L0_P, B * L0, replacement=True).view(B, L0).to(dev)
                      if L0_P is not None else None)
                hyps += run(torch.full((B, 1), EOS if EOS is not None else 0, device=dev), l0)
                sample_seed += [seed] * B
            print(f"  seed {seed}: {n} samples ({time.time() - t0:.0f}s)", flush=True)
        toks = [tk.pre_tokenizer.encode(h) for h in hyps]
        out = {"idxs": list(range(len(hyps))), "hyps": hyps, "refs": [""] * len(hyps),
               "seeds": sample_seed, "toks": toks, "raw_ids": RAW}
    else:
        for s in range(0, len(idxs), bs):
            batch = idxs[s:s + bs]
            hyps += run(torch.tensor([prefix(items[i]) for i in batch], device=dev))
            if (s // bs) % 10 == 0:
                print(f"  {s + len(batch)}/{len(idxs)} ({time.time() - t0:.0f}s)", flush=True)
        out = {"idxs": idxs, "hyps": hyps,
               "refs": [r if isinstance(r, str) else dec(r) for r in (refs[i] for i in idxs)]}
    sync(dev)
    el = time.time() - t0
    out["meta"] = {"device": dev, "seconds": el, "torch": torch.__version__, "config": a.config,
                   "temp": temp, "top_k": top_k, "top_p": top_p, "cfg": cfg, "freq": freq, "mu": mu, "adj": adj}
    print(f"[run_ncm] generated {len(hyps)} outputs on {dev} in {el:.1f}s ({el / max(len(hyps), 1) * 1000:.0f} ms/output)")

    for i in range(min(a.print, len(hyps))):
        print(f"\n--- output {i} ---\n{hyps[i]}")
        if out["refs"][i]:
            print(f"--- reference {i} ---\n{out['refs'][i]}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pickle.dump(out, open(out_path, "wb"))
    print(f"[run_ncm] wrote {out_path}")
    if not a.no_score:
        score(CFG, out, dev)


if __name__ == "__main__":
    main()
