//! tsumuji run  --model DIR --ids FILE [--new N] [--spec K] [--ctx N (0: fill VRAM)] [--logits OUT] [--profile]
//! tsumuji eval --model DIR --out DIR IDS...
//! tsumuji kl   --ref DIR --got DIR --vocab N NAME...
//!
//! `run` feeds the prompt ids in steps of up to 8 (TSJ_CHUNK), then generates `--new`
//! tokens greedily; `--logits` writes every prompt position's logits as f32 [positions][vocab].
//! `eval` does that for several id files with one model load, writing <out>/<stem>.f32.
//! `kl` compares two such logit files per name: KL(ref || got), top-1 agreement.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tsumuji_engine::engine::{Engine, N_MAX, Sampling};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

const USAGE: &str = "usage: tsumuji run --model DIR --ids FILE [--new N] [--spec K] [--ctx N] [--logits OUT]\n       tsumuji bench --model DIR [--steps N] [--n N | --per-n]\n       tsumuji eval --model DIR --out DIR IDS...\n       tsumuji kl --ref DIR --got DIR --vocab N NAME...\n       tsumuji specbench --model DIR [--new N] [--k 1,2,3,auto] [--dump DIR] IDS...";

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| {
        argv.iter()
            .position(|a| a == k)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    // Positional arguments: everything after the subcommand that is not an option or its value.
    let mut pos = Vec::new();
    let mut i = 1;
    while i < argv.len() {
        if argv[i].starts_with("--") {
            i += 2;
        } else {
            pos.push(argv[i].clone());
            i += 1;
        }
    }
    let t0 = Instant::now();
    let open = |ctx: usize| {
        tsumuji_engine::open(
            &PathBuf::from(opt("--model").ok_or("--model DIR")?),
            ctx,
            |m| eprintln!("[{:6.1} s] {m}", t0.elapsed().as_secs_f64()),
        )
    };
    match argv.first().map(String::as_str) {
        Some("run") => {
            let ids = tsumuji_engine::read_ids(&PathBuf::from(opt("--ids").ok_or("--ids FILE")?))?;
            let new: usize = opt("--new")
                .map_or(Ok(0), |v| v.parse())
                .map_err(|e| format!("--new: {e}"))?;
            let ctx: usize = opt("--ctx")
                .map_or(Ok(8192), |v| v.parse())
                .map_err(|e| format!("--ctx: {e}"))?;
            // --ctx 0: as many tokens as fit in free VRAM (Engine::ctx reports the choice).
            let mut eng = open(if ctx == 0 {
                0
            } else {
                ctx.max(ids.len() + new)
            })?;
            eprintln!("context: {} tokens", eng.ctx);
            eng.set_sampling(sampling_opts(&opt)?)?;
            eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f64());
            if argv.iter().any(|a| a == "--profile") {
                eng.profile = Some(
                    tsumuji_engine::profile::Profiler::new(16 * eng.dims.layers + 32)
                        .map_err(|e| e.to_string())?,
                );
            }
            prompt(
                &mut eng,
                &ids,
                opt("--logits").map(PathBuf::from).as_deref(),
            )?;
            let spec: usize = opt("--spec")
                .map_or(Ok(0), |v| v.parse())
                .map_err(|e| format!("--spec: {e}"))?;
            generate(&mut eng, new, spec)?;
            if let Some(p) = &eng.profile {
                eprint!("{}", p.report());
            }
            Ok(())
        }
        Some("eval") => {
            let out = PathBuf::from(opt("--out").ok_or("--out DIR")?);
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            let files: Vec<PathBuf> = pos.iter().map(PathBuf::from).collect();
            let seqs: Vec<Vec<u32>> = files
                .iter()
                .map(|f| tsumuji_engine::read_ids(f))
                .collect::<Result<_, _>>()?;
            let mut eng = open(seqs.iter().map(Vec::len).max().unwrap_or(1))?;
            for (f, ids) in files.iter().zip(&seqs) {
                eng.reset()?;
                let stem = f.file_stem().unwrap().to_string_lossy();
                prompt(&mut eng, ids, Some(&out.join(format!("{stem}.f32"))))?;
            }
            Ok(())
        }
        Some("bench") => {
            let steps: usize = opt("--steps")
                .map_or(Ok(200), |v| v.parse())
                .map_err(|e| format!("--steps: {e}"))?;
            let mut eng = open(steps + 64)?;
            let n: usize = opt("--n")
                .map_or(Ok(1), |v| v.parse())
                .map_err(|e| format!("--n: {e}"))?;
            if argv.iter().any(|a| a == "--per-n") {
                bench_per_n(&mut eng, steps)
            } else {
                bench(&mut eng, steps, n.clamp(1, N_MAX))
            }
        }
        Some("specbench") => {
            let new: usize = opt("--new")
                .map_or(Ok(512), |v| v.parse())
                .map_err(|e| format!("--new: {e}"))?;
            let ks: Vec<usize> = opt("--k")
                .unwrap_or_else(|| "1,2,3,4".into())
                .split(',')
                .map(|v| {
                    if v == "auto" {
                        Ok(usize::MAX)
                    } else {
                        v.parse().map_err(|e| format!("--k: {e}"))
                    }
                })
                .collect::<Result<_, _>>()?;
            let files: Vec<PathBuf> = pos.iter().map(PathBuf::from).collect();
            let seqs: Vec<Vec<u32>> = files
                .iter()
                .map(|f| tsumuji_engine::read_ids(f))
                .collect::<Result<_, _>>()?;
            let longest = seqs.iter().map(Vec::len).max().unwrap_or(1);
            let mut eng = open(longest + new + 16)?;
            eng.set_sampling(sampling_opts(&opt)?)?;
            specbench(
                &mut eng,
                &files,
                &seqs,
                new,
                &ks,
                opt("--dump").map(PathBuf::from).as_deref(),
            )
        }
        Some("detcheck") => {
            let ids = tsumuji_engine::read_ids(&PathBuf::from(opt("--ids").ok_or("--ids FILE")?))?;
            let mut eng = open(ids.len() + 8)?;
            detcheck(&mut eng, &ids)
        }
        Some("kl") => {
            let (r, g) = (
                PathBuf::from(opt("--ref").ok_or("--ref DIR")?),
                PathBuf::from(opt("--got").ok_or("--got DIR")?),
            );
            let v: usize = opt("--vocab")
                .ok_or("--vocab N")?
                .parse()
                .map_err(|e| format!("--vocab: {e}"))?;
            println!(
                "{:<8} {:>6} {:>10} {:>10} {:>10} {:>8}",
                "name", "pos", "KL mean", "KL p99", "KL max", "top-1"
            );
            for name in &pos {
                kl_report(
                    name,
                    &r.join(format!("{name}.f32")),
                    &g.join(format!("{name}.f32")),
                    v,
                )?;
            }
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

/// Prompt tokens per step: TSJ_CHUNK (1..=N_MAX), N_MAX by default. Any chunking gives the same
/// logits bit for bit (tests/tiny_e2e.rs).
fn chunk() -> usize {
    std::env::var("TSJ_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(N_MAX)
        .clamp(1, N_MAX)
}

fn prompt(eng: &mut Engine, ids: &[u32], logits: Option<&Path>) -> Result<(), String> {
    let mut w = match logits {
        Some(p) => Some(std::io::BufWriter::new(
            std::fs::File::create(p).map_err(|e| format!("{}: {e}", p.display()))?,
        )),
        None => None,
    };
    let t = Instant::now();
    let n = chunk();
    let whole = w.is_none() && n == N_MAX;
    if whole {
        eng.prefill(ids)?; // also gives the MTP layer the lookahead token of each chunk
    }
    for part in ids.chunks(n).filter(|_| !whole) {
        eng.step_tokens(part)?;
        if let Some(w) = w.as_mut() {
            let bytes: Vec<u8> = eng
                .logits_all()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            w.write_all(&bytes).map_err(|e| e.to_string())?;
        }
    }
    eng.synchronize()?;
    let s = t.elapsed().as_secs_f64();
    eprintln!(
        "prompt: {} tokens in {s:.2} s ({:.1} tok/s, {n} per step)",
        ids.len(),
        ids.len() as f64 / s
    );
    Ok(())
}

/// --temp T [--top-k K] [--top-p P] [--seed S]: sampling (default greedy); with --temp, top_k
/// and top_p default to the model's generation_config values (20, 0.95).
fn sampling_opts(opt: &dyn Fn(&str) -> Option<String>) -> Result<Sampling, String> {
    let num = |k: &str| {
        opt(k)
            .map(|v| v.parse::<f64>().map_err(|e| format!("{k}: {e}")))
            .transpose()
    };
    let Some(t) = num("--temp")? else {
        return Ok(Sampling::default());
    };
    Ok(Sampling {
        temperature: t as f32,
        top_k: num("--top-k")?.unwrap_or(20.0) as u32,
        top_p: num("--top-p")?.unwrap_or(0.95) as f32,
        seed: num("--seed")?.unwrap_or(1.0) as u64,
    })
}

/// Speculative decoding against plain decoding (greedy or sampled), one model load: per prompt,
/// one token per step, then MTP speculation with each draft count; the tokens must be the same.
/// `--dump DIR`: also writes each prompt's plain greedy (or sampled) continuation to DIR/<stem>.out.
fn specbench(
    eng: &mut Engine,
    files: &[PathBuf],
    seqs: &[Vec<u32>],
    new: usize,
    ks: &[usize],
    dump: Option<&Path>,
) -> Result<(), String> {
    let decode =
        |eng: &mut Engine, ids: &[u32], k: usize| -> Result<(Vec<u32>, f64, usize), String> {
            eng.reset()?;
            eng.prefill(ids)?;
            eng.synchronize()?;
            let t = Instant::now();
            let mut out = Vec::with_capacity(new + 8);
            let mut steps = 0;
            if k == 0 {
                out.push(eng.next_token()?);
                while out.len() < new {
                    eng.step(None)?;
                    out.push(eng.next_token()?);
                    steps += 1;
                }
            } else {
                while out.len() < new {
                    out.extend(if k == usize::MAX {
                        eng.spec_auto()?
                    } else {
                        eng.spec_step(k)?
                    });
                    steps += 1;
                }
            }
            out.truncate(new);
            Ok((out, t.elapsed().as_secs_f64(), steps))
        };
    println!(
        "{:<12} {:>6} {:>9} {:>11} {:>8}",
        "prompt", "drafts", "tok/s", "tok/step", "same"
    );
    for (f, ids) in files.iter().zip(seqs) {
        let name = f.file_stem().unwrap().to_string_lossy();
        let (base, s0, _) = decode(eng, ids, 0)?;
        if let Some(dir) = dump {
            let text: Vec<String> = base.iter().map(u32::to_string).collect();
            std::fs::write(dir.join(format!("{name}.out")), text.join(" ") + "\n")
                .map_err(|e| e.to_string())?;
        }
        println!(
            "{name:<12} {:>6} {:>9.1} {:>11.2} {:>8}",
            0,
            new as f64 / s0,
            1.0,
            "-"
        );
        for &k in ks {
            let (got, s, steps) = decode(eng, ids, k)?;
            let same = if got == base { "yes" } else { "NO" };
            let label = if k == usize::MAX {
                "auto".to_string()
            } else {
                k.to_string()
            };
            println!(
                "{name:<12} {label:>6} {:>9.1} {:>11.2} {same:>8}",
                new as f64 / s,
                new as f64 / steps as f64
            );
        }
    }
    Ok(())
}

/// Greedy generation: one token per step, or with `spec` > 0 speculative steps whose MTP drafts
/// `spec` tokens (same tokens, fewer steps).
fn generate(eng: &mut Engine, new: usize, spec: usize) -> Result<(), String> {
    if new == 0 {
        return Ok(());
    }
    let mut out = Vec::with_capacity(new);
    let t = Instant::now();
    let mut steps = 0usize;
    if spec == 0 {
        let mut next = eng.next_token()?;
        for i in 0..new {
            out.push(next);
            if i + 1 < new {
                eng.step(None)?;
                next = eng.next_token()?;
                steps += 1;
            }
        }
    } else {
        while out.len() < new {
            out.extend(eng.spec_step(spec)?);
            steps += 1;
        }
        out.truncate(new);
    }
    let s = t.elapsed().as_secs_f64();
    let how = if spec > 0 {
        format!(
            ", {spec} drafts: {:.2} tokens per step",
            new as f64 / steps as f64
        )
    } else {
        String::new()
    };
    eprintln!(
        "generated {new} tokens in {s:.2} s ({:.1} tok/s{how})",
        new as f64 / s
    );
    println!(
        "{}",
        out.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
    );
    Ok(())
}

fn read_f32(p: &Path) -> Result<Vec<f32>, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn kl_report(name: &str, rp: &Path, gp: &Path, v: usize) -> Result<(), String> {
    let (r, g) = (read_f32(rp)?, read_f32(gp)?);
    if r.len() != g.len() || r.len() % v != 0 {
        return Err(format!(
            "{name}: sizes differ ({} vs {} floats)",
            r.len(),
            g.len()
        ));
    }
    let n = r.len() / v;
    let mut kls = Vec::with_capacity(n);
    let mut top1 = 0;
    for p in 0..n {
        let (a, b) = (&r[p * v..(p + 1) * v], &g[p * v..(p + 1) * v]);
        kls.push(kl(a, b));
        let am = |x: &[f32]| {
            x.iter()
                .enumerate()
                .fold(0, |bi, (i, &y)| if y > x[bi] { i } else { bi })
        };
        top1 += usize::from(am(a) == am(b));
    }
    if let Some(th) = std::env::var("TSJ_KL_ABOVE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        let big: Vec<String> = kls
            .iter()
            .enumerate()
            .filter(|(_, k)| **k > th)
            .map(|(p, k)| format!("{p}:{k:.1e}"))
            .collect();
        eprintln!(
            "{name}: {} positions above {th:e}: {}",
            big.len(),
            big.join(" ")
        );
    }
    let mean = kls.iter().sum::<f64>() / n as f64;
    let mut sorted = kls.clone();
    sorted.sort_by(f64::total_cmp);
    let p99 = sorted[(n * 99 / 100).min(n - 1)];
    println!(
        "{name:<8} {n:>6} {mean:>10.3e} {p99:>10.3e} {:>10.3e} {:>7.2}%",
        sorted[n - 1],
        100.0 * top1 as f64 / n as f64
    );
    Ok(())
}

/// KL(softmax(reference) || softmax(got)) in f64.
fn kl(reference: &[f32], got: &[f32]) -> f64 {
    let lse = |x: &[f32]| {
        let m = x
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(f64::from(b)));
        m + x
            .iter()
            .map(|&y| (f64::from(y) - m).exp())
            .sum::<f64>()
            .ln()
    };
    let (lr, lg) = (lse(reference), lse(got));
    reference
        .iter()
        .zip(got)
        .map(|(&a, &b)| {
            let (la, lb) = (f64::from(a) - lr, f64::from(b) - lg);
            la.exp() * (la - lb)
        })
        .sum()
}

/// Step time by token count N (graph replays, the same N tokens every step): what verifying
/// N - 1 drafts costs (docs/m3.md).
fn bench_per_n(eng: &mut Engine, steps: usize) -> Result<(), String> {
    println!(
        "{:<3} {:>9} {:>10} {:>10}",
        "N", "ms/step", "ms/token", "vs N=1"
    );
    let mut base = 0.0;
    for n in 1..=N_MAX {
        let toks: Vec<u32> = (0..n as u32).map(|i| 1000 + 7 * i).collect();
        eng.reset()?;
        for _ in 0..4 {
            eng.step_tokens(&toks)?;
        }
        eng.synchronize()?;
        let reps = (steps / n).max(8);
        let t = Instant::now();
        for _ in 0..reps {
            eng.step_tokens(&toks)?;
        }
        eng.synchronize()?;
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        if n == 1 {
            base = ms;
        }
        println!(
            "{n:<3} {ms:>9.3} {:>10.3} {:>9.2}x",
            ms / n as f64,
            ms / base
        );
    }
    Ok(())
}

/// Step time on the device (graph replays without host reads), in full and with each op group
/// left out: the difference is what that group costs inside the PDL pipeline. N = 1 decodes
/// greedily; N > 1 steps the same N tokens every time (a verification window's cost).
fn bench(eng: &mut Engine, steps: usize, n: usize) -> Result<(), String> {
    const GROUPS: &[&str] = &[
        "grdown", "grup", "mixin", "mixmid", "mixout", "inject", "router", "experts", "shared",
        "combine", "lm_head", "ple", "gdnmid", "qsamid",
    ];
    let toks: Vec<u32> = (0..n as u32).map(|i| 1000 + 7 * i).collect();
    let reps = (steps / n).max(8);
    let mut run = |skip: &[&str]| -> Result<f64, String> {
        eng.set_skip(skip);
        eng.reset()?;
        let one = |eng: &mut Engine| {
            if n == 1 {
                eng.step(None)
            } else {
                eng.step_tokens(&toks)
            }
        };
        eng.step_tokens(&toks)?;
        for _ in 0..16 / n {
            one(eng)?;
        }
        eng.synchronize()?;
        let t = Instant::now();
        for _ in 0..reps {
            one(eng)?;
        }
        eng.synchronize()?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / reps as f64)
    };
    let full = run(&[])?;
    println!(
        "full step (N = {n}): {full:.3} ms ({:.1} tok/s)",
        n as f64 * 1e3 / full
    );
    println!("{:<10} {:>9} {:>9}", "without", "ms/step", "saves");
    for g in GROUPS {
        let ms = run(&[g])?;
        println!("{g:<10} {ms:>9.3} {:>9.3}", full - ms);
    }
    let gemv_only = run(&["grup", "mixmid", "inject", "combine", "router", "experts"])?;
    println!(
        "{:<10} {gemv_only:>9.3} {:>9.3}  (grdown, mixin, mixout, lm_head, ple only)",
        "gemv-only",
        full - gemv_only
    );
    let none = run(GROUPS)?;
    println!(
        "{:<10} {none:>9.3} {:>9.3}  (begin + embed only)",
        "nothing",
        full - none
    );
    eng.set_skip(&[]);
    Ok(())
}

/// Hunts for nondeterminism: runs the prompt several times per configuration (all kernels with
/// PDL, then each kernel without it) and reports which configurations give identical logits.
fn detcheck(eng: &mut Engine, ids: &[u32]) -> Result<(), String> {
    let run = |eng: &mut Engine| -> Result<u64, String> {
        eng.reset()?;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &t in ids {
            eng.step(Some(t))?;
            for v in eng.logits()? {
                h = (h ^ u64::from(v.to_bits())).wrapping_mul(0x100_0000_01b3);
            }
        }
        Ok(h)
    };
    let report = |eng: &mut Engine, label: &str, n: usize| -> Result<bool, String> {
        let hs: Vec<u64> = (0..n).map(|_| run(eng)).collect::<Result<_, _>>()?;
        let same = hs.iter().all(|&h| h == hs[0]);
        println!(
            "{label:<28} {}",
            if same { "deterministic" } else { "DIFFERS" }
        );
        Ok(same)
    };
    report(eng, "all PDL", 4)?;
    for name in eng.kernel_names() {
        eng.set_no_pdl(std::slice::from_ref(&name));
        report(eng, &format!("without PDL: {name}"), 3)?;
    }
    eng.set_no_pdl(&[]);
    Ok(())
}
