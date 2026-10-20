//! GPU slot protocol v2 (docs/GPU_SHARING.md). `gpu.lock` is the slot for one
//! heavy job at a time (Krea 2 / MOSS / H3 in serve, training, edit tools, batches).
//! Resident chat LLMs stay outside it on serve's VRAM ledger. A process waiting
//! for the slot declares its VRAM need in `gpu.d/<pid>.json` next to the lock.
use crate::json::{self, quote};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Chat models kept resident next to the slot; every other model holds it.
pub const LLMS: [&str; 1] = ["glm53"];
pub fn is_llm(id: &str) -> bool {
    LLMS.contains(&id)
}
/// Headroom on top of a declared need; gpu_slot.py waits for the same.
pub const MARGIN_BYTES: u64 = 1 << 30;
const GIB: f64 = (1u64 << 30) as f64;

#[derive(Clone, Debug, PartialEq)]
pub struct Declaration {
    pub pid: u32,
    pub who: String,
    pub need_gib: f64,
    pub since: u64,
    /// Lower goes first (0 = most urgent); waiters take the slot in
    /// (priority, since) order. Missing means DEFAULT_PRIORITY.
    pub priority: u32,
    /// Short text for Studio's queue display.
    pub label: Option<String>,
}
pub const DEFAULT_PRIORITY: u32 = 5;
/// Priority of serve's own slot work unless the request says otherwise:
/// Studio's interactive generations go before background jobs.
pub const SERVE_PRIORITY: u32 = 1;
pub const AGE_SECONDS: u64 = 30 * 60;
pub const AGE_FLOOR: u32 = 2;
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
/// `priority` (0..=9, default SERVE_PRIORITY) of a media or load request.
pub fn request_priority(v: &json::Value) -> hayate_core::Result<u32> {
    match v.optional("priority") {
        None | Some(json::Value::Null) => Ok(SERVE_PRIORITY),
        Some(p) => p
            .integer()
            .ok()
            .and_then(|p| u32::try_from(p).ok())
            .filter(|p| *p <= 9)
            .ok_or_else(|| hayate_core::Error::InvalidPlan("priority must be 0..=9".into())),
    }
}
pub fn request_label(v: &json::Value) -> hayate_core::Result<Option<String>> {
    match v.optional("label") {
        None | Some(json::Value::Null) => Ok(None),
        Some(l) => Ok(Some(l.string()?.chars().take(120).collect())),
    }
}
impl Declaration {
    fn parse(text: &str) -> Option<Self> {
        let v = json::parse(text).ok()?;
        let d = Self {
            pid: v.get("pid").ok()?.integer().ok()?.try_into().ok()?,
            who: v.get("who").ok()?.string().ok()?.into(),
            need_gib: v.get("need_gib").ok()?.number().ok()?,
            since: v.get("since").ok()?.number().ok()? as u64,
            priority: match v.optional("priority") {
                Some(p) => p.integer().ok()?.try_into().ok()?,
                None => DEFAULT_PRIORITY,
            },
            label: match v.optional("label") {
                None | Some(json::Value::Null) => None,
                Some(l) => Some(l.string().ok()?.into()),
            },
        };
        (d.need_gib.is_finite() && d.need_gib >= 0.).then_some(d)
    }
    /// Aging against starvation: one step up per AGE_SECONDS waited, never
    /// past AGE_FLOOR, so user requests (0/1) stay ahead of aged jobs.
    pub fn effective_priority(&self, now: u64) -> u32 {
        if self.priority <= AGE_FLOOR {
            return self.priority;
        }
        let steps = now.saturating_sub(self.since) / AGE_SECONDS;
        self.priority
            .saturating_sub(steps.min(u64::from(u32::MAX)) as u32)
            .max(AGE_FLOOR)
    }
    /// Free VRAM the declarer waits for: the need plus the margin.
    pub fn need_bytes(&self) -> u64 {
        (self.need_gib * GIB).ceil() as u64 + MARGIN_BYTES
    }
    pub fn json(&self) -> String {
        format!(
            r#"{{"pid":{},"who":{},"need_gib":{},"since":{},"priority":{},"effective_priority":{},"label":{}}}"#,
            self.pid,
            quote(&self.who),
            self.need_gib,
            self.since,
            self.priority,
            self.effective_priority(now()),
            self.label.as_deref().map_or("null".into(), quote)
        )
    }
}

pub fn declarations_dir(lock: &Path) -> PathBuf {
    lock.with_file_name("gpu.d")
}
fn alive(pid: u32) -> bool {
    pid > 0 && Path::new(&format!("/proc/{pid}")).exists()
}
/// Live declarations in slot order (effective priority, then oldest first). Temporary files (`.name`), broken JSON and
/// files of processes that no longer exist are skipped, not deleted.
pub fn declarations(lock: &Path) -> Vec<Declaration> {
    let Ok(dir) = fs::read_dir(declarations_dir(lock)) else {
        return Vec::new();
    };
    let mut out = dir
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.ends_with(".json") && !n.starts_with('.'))
        })
        .filter_map(|e| Declaration::parse(&fs::read_to_string(e.path()).ok()?))
        .filter(|d| alive(d.pid))
        .collect::<Vec<_>>();
    let now = now();
    out.sort_by_key(|d| (d.effective_priority(now), d.since, d.pid));
    out
}
/// Whether `pid` may take the flock: its declaration is first, or it has none.
pub fn first_in_line(lock: &Path, pid: u32) -> bool {
    let line = declarations(lock);
    line.first().is_none_or(|d| d.pid == pid) || line.iter().all(|d| d.pid != pid)
}
/// Write `d` as gpu.d/<pid>.json atomically, as gpu_slot.py does.
pub fn declare(lock: &Path, d: &Declaration) -> std::io::Result<PathBuf> {
    let dir = declarations_dir(lock);
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", d.pid));
    let temporary = dir.join(format!(".{}.json.tmp", d.pid));
    fs::write(&temporary, d.json())?;
    fs::rename(&temporary, &path)?;
    Ok(path)
}

/// `flock` rows on `lock` in /proc/locks: (pid, blocked). A blocked row is a
/// process waiting for the slot; the other row is its holder.
pub fn lock_rows(lock: &Path) -> Vec<(u32, bool)> {
    use std::os::unix::fs::MetadataExt;
    let Ok(inode) = fs::metadata(lock).map(|m| m.ino().to_string()) else {
        return Vec::new();
    };
    let Ok(locks) = fs::read_to_string("/proc/locks") else {
        return Vec::new();
    };
    locks
        .lines()
        .filter_map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            let at = fields.iter().position(|f| {
                f.matches(':').count() == 2 && f.rsplit(':').next() == Some(inode.as_str())
            })?;
            if !fields.contains(&"FLOCK") {
                return None;
            }
            let pid = fields.get(at.checked_sub(1)?)?.parse().ok()?;
            Some((pid, fields.contains(&"->")))
        })
        .collect()
}

/// Whether `free` device memory covers a declared `need` (both bytes).
/// An undeclared waiter (`need` None) or an unknown `free` never fits.
pub fn covers(need: Option<u64>, free: Option<u64>) -> bool {
    need.zip(free).is_some_and(|(need, free)| free >= need)
}

/// Elastic VRAM of a resident LLM (GLM-5.3): what it may use now, what it
/// grows back to when the slot is free, and how many experts sit on the GPU.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Budget {
    pub id: &'static str,
    pub budget_bytes: usize,
    pub default_bytes: usize,
    pub experts_gpu: usize,
    pub experts_total: usize,
    /// Decode tok/s of the last reply (0 before the first).
    pub decode_tok_s: f64,
}
impl Budget {
    pub fn json(&self) -> String {
        format!(
            r#"{{"budget_bytes":{},"default_budget_bytes":{},"shrunk":{},"experts_gpu":{},"experts_total":{},"decode_tok_s":{:.2}}}"#,
            self.budget_bytes,
            self.default_bytes,
            self.budget_bytes < self.default_bytes,
            self.experts_gpu,
            self.experts_total,
            self.decode_tok_s
        )
    }
}

/// What the GPU worker last published for `GET /v1/gpu`.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub lock: Option<PathBuf>,
    /// Resident models and the VRAM each added when it was loaded.
    pub models: Vec<(&'static str, usize)>,
    /// cudaMemGetInfo (free, total), refreshed about once a second while idle.
    pub memory: Option<(usize, usize)>,
    /// Media job serve is running.
    pub job: Option<String>,
    /// Elastic LLM budgets, shown on their ledger rows.
    pub budgets: Vec<Budget>,
}

fn holder_who(pid: u32, declared: &[Declaration]) -> Option<String> {
    if pid == std::process::id() {
        return Some("hayate serve".into());
    }
    if let Some(d) = declared.iter().find(|d| d.pid == pid) {
        return Some(d.who.clone());
    }
    let cmdline = fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    let text = String::from_utf8_lossy(&cmdline).replace('\0', " ");
    Some(text.trim().chars().take(200).collect())
}

/// `GET /v1/gpu`: serve's VRAM ledger, the slot holder and the waiters.
pub fn status(s: &Snapshot) -> String {
    let pid = std::process::id();
    let models = s
        .models
        .iter()
        .map(|(id, bytes)| {
            let budget = s
                .budgets
                .iter()
                .find(|b| b.id == *id)
                .map_or(String::new(), |b| format!(r#","budget":{}"#, b.json()));
            format!(
                r#"{{"id":{},"vram_bytes":{bytes},"slot":{}{budget}}}"#,
                quote(id),
                !is_llm(id)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let used = s.models.iter().map(|(_, b)| b).sum::<usize>();
    let job = s.job.as_deref().map_or("null".into(), quote);
    let device = s.memory.map_or("null".into(), |(free, total)| {
        format!(r#"{{"free_bytes":{free},"total_bytes":{total}}}"#)
    });
    let slot = s.lock.as_ref().map_or("null".into(), |lock| {
        let rows = lock_rows(lock);
        let declared = declarations(lock);
        let holder = rows.iter().find(|r| !r.1).map_or("null".into(), |&(p, _)| {
            format!(
                r#"{{"pid":{p},"who":{},"serve":{}}}"#,
                holder_who(p, &declared)
                    .as_deref()
                    .map_or("null".into(), quote),
                p == pid
            )
        });
        format!(
            r#"{{"lock":{},"holder":{holder},"declarations":[{}],"waiters":[{}]}}"#,
            quote(&lock.to_string_lossy()),
            declared
                .iter()
                .map(Declaration::json)
                .collect::<Vec<_>>()
                .join(","),
            rows.iter()
                .filter(|r| r.1)
                .map(|r| r.0.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    });
    format!(
        r#"{{"serve":{{"pid":{pid},"used_bytes":{used},"models":[{models}],"job":{job}}},"device":{device},"slot":{slot}}}"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;

    fn directory(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hayate-gpu-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("gpu.d")).unwrap();
        d
    }
    fn dead_pid() -> u32 {
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let pid = child.id();
        child.wait().unwrap();
        pid
    }

    #[test]
    fn declarations_skip_dead_pids_broken_json_and_temporary_files() {
        let d = directory("declarations");
        let lock = d.join("gpu.lock");
        let me = std::process::id();
        let write = |name: &str, text: &str| fs::write(d.join("gpu.d").join(name), text).unwrap();
        write(
            "1.json",
            r#"{"pid":1,"who":"older","need_gib":2.5,"since":100}"#,
        );
        write(
            &format!("{me}.json"),
            &format!(r#"{{"pid":{me},"who":"hayate-train krea2","need_gib":30,"since":200}}"#),
        );
        let dead = dead_pid();
        write(
            &format!("{dead}.json"),
            &format!(r#"{{"pid":{dead},"who":"dead","need_gib":1,"since":1}}"#),
        );
        write("7.json", r#"{"pid":7,"who":"bro"#);
        write(
            "8.json",
            r#"{"pid":1,"who":"negative","need_gib":-1,"since":1}"#,
        );
        write(
            &format!(".{me}.json.tmp"),
            &format!(r#"{{"pid":{me},"who":"tmp","need_gib":9,"since":0}}"#),
        );
        let got = declarations(&lock);
        assert_eq!(
            got.iter().map(|d| d.who.as_str()).collect::<Vec<_>>(),
            ["older", "hayate-train krea2"]
        );
        assert_eq!(
            (got[0].priority, got[0].label.as_deref()),
            (DEFAULT_PRIORITY, None)
        );
        // A higher priority (lower number) goes first regardless of age.
        write(
            "1.json",
            // Recent: an old priority-7 declaration would have aged past 5.
            &format!(
                r#"{{"pid":1,"who":"older","need_gib":2.5,"since":{},"priority":7,"label":"GLM quant"}}"#,
                now()
            ),
        );
        let got = declarations(&lock);
        assert_eq!(
            got.iter()
                .map(|d| (d.who.as_str(), d.priority))
                .collect::<Vec<_>>(),
            [("hayate-train krea2", DEFAULT_PRIORITY), ("older", 7)]
        );
        assert_eq!(got[1].label.as_deref(), Some("GLM quant"));
        assert!(
            got[1]
                .json()
                .contains(r#""priority":7,"effective_priority":7,"label":"GLM quant""#)
        );
        write(
            "1.json",
            r#"{"pid":1,"who":"older","need_gib":2.5,"since":100}"#,
        );
        let got = declarations(&lock);
        assert_eq!(got[1].need_bytes(), 31 << 30);
        assert_eq!(got[0].need_bytes(), (5 << 29) + MARGIN_BYTES);
        // No directory at all: nothing declared.
        assert!(declarations(&d.join("missing").join("gpu.lock")).is_empty());
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn aging_lifts_background_jobs_but_never_past_user_requests() {
        let d = |priority, waited| Declaration {
            pid: 1,
            who: "x".into(),
            need_gib: 1.,
            since: 1_000_000 - waited,
            priority,
            label: None,
        };
        let now = 1_000_000;
        assert_eq!(d(7, 0).effective_priority(now), 7);
        assert_eq!(d(7, AGE_SECONDS - 1).effective_priority(now), 7);
        assert_eq!(d(7, AGE_SECONDS).effective_priority(now), 6);
        assert_eq!(d(7, 6 * AGE_SECONDS).effective_priority(now), 2);
        assert_eq!(d(9, 100 * AGE_SECONDS).effective_priority(now), 2);
        assert_eq!(d(1, 100 * AGE_SECONDS).effective_priority(now), 1);
        assert_eq!(d(0, 0).effective_priority(now), 0);
        assert!(
            d(7, 0)
                .json()
                .contains(r#""priority":7,"effective_priority":"#)
        );
        assert!(
            Declaration::parse(r#"{"pid":1,"who":"x","need_gib":1,"since":1,"label":null}"#)
                .is_some()
        );
    }

    #[test]
    fn need_is_covered_only_by_enough_known_free_memory() {
        let need = Some(30 << 30);
        assert!(covers(need, Some(30 << 30)));
        assert!(!covers(need, Some((30 << 30) - 1)));
        assert!(!covers(None, Some(u64::MAX)));
        assert!(!covers(need, None));
    }

    #[test]
    fn lock_rows_report_holder_and_blocked_waiter() {
        let d = directory("rows");
        let lock = d.join("gpu.lock");
        let open = || {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&lock)
                .unwrap()
        };
        assert!(lock_rows(&lock).is_empty());
        let holder = open();
        holder.lock().unwrap();
        let me = std::process::id();
        assert_eq!(lock_rows(&lock), [(me, false)]);
        let waiter = std::thread::spawn({
            let file = open();
            move || file.lock().unwrap()
        });
        let start = std::time::Instant::now();
        while lock_rows(&lock).len() < 2 {
            assert!(start.elapsed() < std::time::Duration::from_secs(5));
            std::thread::yield_now();
        }
        assert!(lock_rows(&lock).contains(&(me, true)));
        let status = json::parse(&status(&Snapshot {
            lock: Some(lock.clone()),
            models: vec![("glm53", 20 << 30), ("external-job", 12 << 30)],
            memory: Some((40 << 30, 96 << 30)),
            job: Some("job-1".into()),
            budgets: vec![],
        }))
        .unwrap();
        let serve = status.get("serve").unwrap();
        assert_eq!(
            serve.get("used_bytes").unwrap().integer().unwrap(),
            32 << 30
        );
        let models = serve.get("models").unwrap().array().unwrap();
        assert!(!models[0].get("slot").unwrap().boolean().unwrap());
        assert!(models[1].get("slot").unwrap().boolean().unwrap());
        assert_eq!(serve.get("job").unwrap().string().unwrap(), "job-1");
        let slot = status.get("slot").unwrap();
        let owner = slot.get("holder").unwrap();
        assert!(owner.get("serve").unwrap().boolean().unwrap());
        assert_eq!(owner.get("who").unwrap().string().unwrap(), "hayate serve");
        assert_eq!(slot.get("waiters").unwrap().array().unwrap().len(), 1);
        drop(holder);
        waiter.join().unwrap();
        fs::remove_dir_all(d).unwrap();
    }
}
