//! Bounded concurrent image-caption client. One append-only result per image, resumable by key.
use darask_json::{Value as V, quote};
use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{Read, Write},
    net::TcpStream,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
type Res<T> = Result<T, String>;
#[derive(Clone)]
struct Config {
    endpoint: String,
    model: String,
    prompt: String,
    jobs: usize,
    retries: usize,
    max_tokens: usize,
    detail: String,
    out: PathBuf,
}
fn files(path: &Path, out: &mut Vec<PathBuf>) -> Res<()> {
    if path.is_file() {
        if path.extension().and_then(|x| x.to_str()).is_some_and(|s| {
            ["png", "jpg", "jpeg", "webp", "bmp", "ppm", "pgm", "pnm"]
                .contains(&s.to_ascii_lowercase().as_str())
        }) {
            out.push(path.canonicalize().map_err(|e| e.to_string())?);
        }
        return Ok(());
    }
    for entry in fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.file_type().map_err(|e| e.to_string())?.is_symlink() {
            continue;
        }
        files(&entry.path(), out)?;
    }
    Ok(())
}
use darask_text::base64::encode as base64;
fn key(path: &Path, c: &Config) -> Res<String> {
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    let modified = meta
        .modified()
        .map_err(|e| e.to_string())?
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let s = format!(
        "{}\0{}\0{}\0{}\0{}\0{}\0{}\0{}",
        path.display(),
        meta.len(),
        modified,
        c.endpoint,
        c.model,
        c.prompt,
        c.max_tokens,
        c.detail
    );
    let hash = s.bytes().fold(0xcbf29ce484222325u64, |n, b| {
        (n ^ b as u64).wrapping_mul(0x100000001b3)
    });
    Ok(format!("{hash:016x}"))
}
fn endpoint(url: &str) -> Res<(String, String)> {
    let url = url
        .strip_prefix("http://")
        .ok_or("--endpoint must be an http:// inference server URL")?;
    let (host, path) = url.split_once('/').unwrap_or((url, ""));
    if host.is_empty() || host.contains(['\r', '\n', '@']) || path.contains(['\r', '\n']) {
        return Err("invalid endpoint".into());
    }
    let path = path.trim_end_matches('/');
    let path = if path.ends_with("/chat/completions") {
        format!("/{path}")
    } else if path.is_empty() {
        "/v1/chat/completions".into()
    } else {
        format!("/{path}/chat/completions")
    };
    Ok((host.into(), path))
}
fn post(c: &Config, body: &str, route: Option<&str>) -> Res<V> {
    let (host, mut path) = endpoint(&c.endpoint)?;
    if let Some(route) = route {
        path = format!(
            "{}{route}",
            path.strip_suffix("/chat/completions")
                .ok_or("invalid API base path")?
        );
    }
    let address = if host.contains(':') {
        host.clone()
    } else {
        format!("{host}:80")
    };
    let mut stream = TcpStream::connect(&address).map_err(|e| format!("connect {address}: {e}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(1800)))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(60)))
        .map_err(|e| e.to_string())?;
    write!(stream,"POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",body.len()).map_err(|e|e.to_string())?;
    let mut response = Vec::new();
    stream
        .take(8 << 20)
        .read_to_end(&mut response)
        .map_err(|e| e.to_string())?;
    let end = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("incomplete HTTP response")?;
    let header = std::str::from_utf8(&response[..end]).map_err(|e| e.to_string())?;
    let status = header
        .lines()
        .next()
        .and_then(|s| s.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or("invalid HTTP status")?;
    let data = &response[end + 4..];
    // All native non-streaming endpoints send Content-Length. Never accept truncated output.
    if let Some(n) = header
        .lines()
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .map(|(_, v)| v.trim().parse::<usize>())
        .transpose()
        .map_err(|e| e.to_string())?
        && data.len() != n
    {
        return Err("truncated/oversized caption response".into());
    }
    let body = std::str::from_utf8(data).map_err(|e| e.to_string())?;
    if status != 200 {
        return Err(format!(
            "HTTP {status}: {}",
            body.chars().take(1200).collect::<String>()
        ));
    }
    darask_json::parse(body).map_err(|e| e.to_string())
}
fn caption(path: &Path, c: &Config) -> Res<(String, V)> {
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    if meta.len() > 24 << 20 {
        return Err("image exceeds 24 MiB".into());
    }
    let data = fs::read(path).map_err(|e| e.to_string())?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png")
        .to_ascii_lowercase();
    let mime = if ext == "jpg" { "jpeg" } else { &ext };
    let url = format!("data:image/{mime};base64,{}", base64(&data));
    let body = format!(
        r#"{{"model":{},"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":{},"detail":{}}}}},{{"type":"text","text":{}}}]}}],"max_tokens":{},"temperature":0,"enable_thinking":false,"stream":false}}"#,
        quote(&c.model),
        quote(&url),
        quote(&c.detail),
        quote(&c.prompt),
        c.max_tokens
    );
    let mut last = String::new();
    for attempt in 0..=c.retries {
        match post(c, &body, None).and_then(|v| {
            let choice = v
                .get("choices")
                .and_then(V::array)
                .map_err(|e| e.to_string())?
                .first()
                .ok_or("empty choices")?;
            let text = choice
                .get("message")
                .and_then(|v| v.get("content"))
                .and_then(V::string)
                .map_err(|e| e.to_string())?
                .to_string();
            if text.trim().is_empty() {
                return Err("empty caption".into());
            }
            Ok((text, v))
        }) {
            Ok(v) => return Ok(v),
            Err(e) => last = e,
        }
        if attempt < c.retries {
            std::thread::sleep(Duration::from_millis(250 * (1 << attempt.min(5))));
        }
    }
    Err(last)
}
pub fn run(engine: &str, args: &[String]) -> Res<i32> {
    let (port, model) = match engine {
        "hayate" => (7870, "glm53"),
        "kogarashi" => (8080, "kogarashi-ornith"),
        "tsumuji" => (8090, "qwen38"),
        _ => return Err("batch engine: hayate|kogarashi|tsumuji".into()),
    };
    let mut c = Config {
        endpoint: format!("http://127.0.0.1:{port}/v1"),
        model: model.into(),
        prompt: "Describe this image accurately and concisely.".into(),
        jobs: 2,
        retries: 2,
        max_tokens: 256,
        detail: "auto".into(),
        out: PathBuf::from("captions.jsonl"),
    };
    let mut input = None;
    let (pairs, remainder) = args.as_chunks::<2>();
    for p in pairs {
        let value = &p[1];
        match p[0].as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--out" => c.out = PathBuf::from(value),
            "--endpoint" => c.endpoint = value.clone(),
            "--model" => c.model = value.clone(),
            "--prompt" => c.prompt = value.clone(),
            "--jobs" => c.jobs = value.parse().map_err(|_| "--jobs integer")?,
            "--retry" => c.retries = value.parse().map_err(|_| "--retry integer")?,
            "--max-tokens" => c.max_tokens = value.parse().map_err(|_| "--max-tokens integer")?,
            "--detail" => c.detail = value.clone(),
            k => return Err(format!("unknown batch option {k}")),
        }
    }
    if !remainder.is_empty() {
        return Err("each batch option requires a value".into());
    }
    if !(1..=16).contains(&c.jobs)
        || c.retries > 10
        || c.max_tokens == 0
        || !["auto", "low", "high"].contains(&c.detail.as_str())
    {
        return Err(
            "jobs: 1..16; retry: 0..10; max-tokens: positive; detail: auto|low|high".into(),
        );
    }
    endpoint(&c.endpoint)?;
    let mut paths = Vec::new();
    files(
        &input.ok_or("batch needs --input FILE_OR_DIRECTORY")?,
        &mut paths,
    )?;
    paths.sort();
    paths.dedup();
    if let Some(parent) = c.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut out = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(&c.out)
        .map_err(|e| e.to_string())?;
    out.try_lock()
        .map_err(|e| format!("cannot lock result file (another batch may be running): {e}"))?;
    let mut complete = HashSet::new();
    {
        use std::io::BufRead;
        let mut reader = std::io::BufReader::new(out.try_clone().map_err(|e| e.to_string())?);
        let mut offset = 0u64;
        let mut line = String::new();
        let mut row_number = 0;
        loop {
            line.clear();
            if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                break;
            }
            row_number += 1;
            let row = match darask_json::parse(&line) {
                Ok(row) => row,
                Err(_) if !line.ends_with('\n') => {
                    out.set_len(offset).map_err(|e| e.to_string())?;
                    eprintln!("recovered an interrupted final result row");
                    break;
                }
                Err(e) => return Err(format!("result line {row_number}: {e}")),
            };
            if row.optional("status").and_then(|v| v.string().ok()) == Some("ok")
                && let Some(k) = row.optional("key").and_then(|v| v.string().ok())
            {
                complete.insert(k.to_string());
            }
            offset += line.len() as u64;
            if !line.ends_with('\n') {
                writeln!(out).map_err(|e| e.to_string())?;
                break;
            }
        }
    }
    let all = paths.len();
    let work = paths
        .into_iter()
        .map(|p| key(&p, &c).map(|k| (p, k)))
        .collect::<Res<Vec<_>>>()?
        .into_iter()
        .filter(|(_, k)| !complete.contains(k))
        .collect::<Vec<_>>();
    eprintln!(
        "images={all}, queued={}, already complete={}, jobs={}",
        work.len(),
        all - work.len(),
        c.jobs
    );
    if work.is_empty() {
        return Ok(0);
    }
    if engine == "hayate" {
        // Hayate normally loads on an explicit Studio action; batch users only need serve + batch.
        eprintln!("preparing Hayate model…");
        post(
            &c,
            &format!("{{\"id\":{}}}", quote(&c.model)),
            Some("/models/load"),
        )?;
    }
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::sync_channel(c.jobs * 2);
    let mut failures = 0;
    std::thread::scope(|scope| -> Res<()> {
        for _ in 0..c.jobs.min(work.len()) {
            let (tx, c, work, next) = (tx.clone(), &c, &work, &next);
            scope.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some((path, k)) = work.get(i) else { break };
                    let start = SystemTime::now();
                    let result = caption(path, c);
                    if tx
                        .send((
                            i,
                            path,
                            k,
                            start.elapsed().unwrap_or_default().as_secs_f64(),
                            result,
                        ))
                        .is_err()
                    {
                        break;
                    }
                }
            });
        }
        drop(tx);
        for (count, (i, path, k, seconds, result)) in rx.into_iter().enumerate() {
            let (status, caption, error, response) = match result {
                Ok((text, v)) => ("ok", V::String(text), V::Null, v),
                Err(e) => {
                    failures += 1;
                    ("error", V::Null, V::String(e), V::Null)
                }
            };
            let row = V::Object(vec![
                ("key".into(), V::String(k.clone())),
                ("index".into(), V::Number(i.to_string())),
                ("image".into(), V::String(path.to_string_lossy().into())),
                ("model".into(), V::String(c.model.clone())),
                ("prompt".into(), V::String(c.prompt.clone())),
                ("status".into(), V::String(status.into())),
                ("caption".into(), caption),
                ("error".into(), error),
                ("seconds".into(), V::Number(seconds.to_string())),
                ("response".into(), response),
            ]);
            writeln!(out, "{}", row.dump()).map_err(|e| e.to_string())?;
            out.flush().map_err(|e| e.to_string())?;
            eprintln!(
                "[{}/{}] {status} {} ({seconds:.2}s)",
                count + 1,
                work.len(),
                path.display()
            );
        }
        Ok(())
    })?;
    eprintln!(
        "saved {}, errors={failures}; rerun the same command to resume",
        c.out.display()
    );
    Ok(if failures == 0 { 0 } else { 1 })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn two_workers_retry_and_resume() {
        use std::net::TcpListener;
        let dir = std::env::temp_dir().join(format!(
            "darask-batch-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        for i in 0..3 {
            fs::write(dir.join(format!("{i}.png")), b"image bytes").unwrap();
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::scope(|s| {
                    for i in 0..4 {
                        let (mut socket, _) = listener.accept().unwrap();
                        let (active, peak) = (&active, &peak);
                        s.spawn(move || {
                            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(now, Ordering::SeqCst);
                            let mut request = Vec::new();
                            loop {
                                let mut chunk = [0u8; 1024];
                                let n = socket.read(&mut chunk).unwrap();
                                assert_ne!(n, 0, "incomplete request");
                                request.extend_from_slice(&chunk[..n]);
                                if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                                    let headers = std::str::from_utf8(&request[..end]).unwrap();
                                    let length: usize = headers.lines().filter_map(|l| l.split_once(':')).find(|(k,_)| k.eq_ignore_ascii_case("content-length")).unwrap().1.trim().parse().unwrap();
                                    if request.len() >= end+4+length { break; }
                                }
                            }
                            assert!(String::from_utf8_lossy(&request).contains("image_url"));
                            std::thread::sleep(Duration::from_millis(25));
                            let (status, body) = if i == 0 { (503, "busy") } else { (200, r#"{"choices":[{"message":{"content":"A red square."}}]}"#) };
                            write!(socket, "HTTP/1.1 {status} OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
                            active.fetch_sub(1, Ordering::SeqCst);
                        });
                    }
                });
            });
            let args = vec![
                "--input".into(),
                dir.to_string_lossy().into(),
                "--out".into(),
                dir.join("results.jsonl").to_string_lossy().into(),
                "--endpoint".into(),
                endpoint,
                "--jobs".into(),
                "2".into(),
            ];
            assert_eq!(run("kogarashi", &args).unwrap(), 0);
            OpenOptions::new()
                .append(true)
                .open(dir.join("results.jsonl"))
                .unwrap()
                .write_all(b"{\"interrupted")
                .unwrap();
            assert_eq!(run("kogarashi", &args).unwrap(), 0);
        });
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert_eq!(
            fs::read_to_string(dir.join("results.jsonl"))
                .unwrap()
                .lines()
                .count(),
            3
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
