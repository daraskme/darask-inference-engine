use super::*;
use crate::chat::Chat;
use hayate_llm::session::Timings;

struct CpuBackend {
    resident: bool,
}
impl Backend for CpuBackend {
    fn models(&self) -> Result<String> {
        Ok(format!("{{\"resident\":{}}}", self.resident))
    }
    fn load(&mut self, _: &Value, progress: &mut dyn FnMut(String) -> Result<()>) -> Result<()> {
        progress("{\"stage\":\"weights\",\"done\":0,\"total\":12}".into())?;
        self.resident = true;
        Ok(())
    }
    fn unload(&mut self, _: &Value) -> Result<()> {
        self.resident = false;
        Ok(())
    }
    fn chat(
        &mut self,
        request: &Value,
        emit: &mut dyn FnMut(String) -> Result<()>,
    ) -> Result<String> {
        assert!(self.resident);
        let mut chat = Chat::parse(request)?;
        emit(chat.start())?;
        chat.push("reason</think>hi", emit)?;
        chat.finish(
            &Timings {
                prompt_tokens: 10,
                cached_tokens: 5,
                completion_tokens: 3,
                ..Timings::default()
            },
            emit,
        )
    }
}
fn request(sender: &queue::Sender, method: &str, path: &str, body: &str, sse: bool) -> Vec<Event> {
    let (reply, events) = mpsc::sync_channel(8);
    sender
        .send(Job {
            work: Work::Http(
                http::Request {
                    method: method.into(),
                    path: path.into(),
                    body: body.into(),
                    sse,
                    ..Default::default()
                },
                reply,
            ),
        })
        .unwrap();
    events.into_iter().collect()
}
#[test]
fn serialized_worker_lifecycle_sse_and_json_errors() {
    let (sender, receiver) = queue::channel(8);
    let worker =
        thread::spawn(move || worker_loop(receiver, &mut CpuBackend { resident: false }).unwrap());
    let events = request(&sender, "GET", "/v1/models", "", false);
    assert!(matches!(&events[1], Event::Data(s) if s.contains("false")));
    let events = request(
        &sender,
        "POST",
        "/v1/models/load",
        r#"{"id":"glm53"}"#,
        true,
    );
    assert!(matches!(&events[0], Event::Headers(200, true, 0)));
    assert!(matches!(&events[2], Event::Data(s) if s == "{\"stage\":\"ready\"}"));
    let body = r#"{"model":"glm53","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let events = request(&sender, "POST", "/v1/chat/completions", body, false);
    assert!(matches!(&events[0], Event::Headers(200, true, 0)));
    assert!(matches!(&events[events.len()-2], Event::Data(s) if s == "[DONE]"));
    for event in &events[1..events.len() - 2] {
        if let Event::Data(s) = event {
            json::parse(s).unwrap();
        }
    }
    let events = request(
        &sender,
        "POST",
        "/v1/chat/completions",
        &body.replace("true", "false"),
        false,
    );
    // A non-streaming reply probes the client with unwritten keepalives first.
    assert!(matches!(&events[0], Event::Keepalive));
    let events = events
        .into_iter()
        .filter(|e| !matches!(e, Event::Keepalive))
        .collect::<Vec<_>>();
    assert!(matches!(&events[0], Event::Headers(200, false, _)));
    assert!(
        matches!(&events[1], Event::Data(s) if json::parse(s).unwrap().get("object").unwrap().string().unwrap() == "chat.completion")
    );
    request(
        &sender,
        "POST",
        "/v1/models/unload",
        r#"{"id":"glm53"}"#,
        false,
    );
    let events = request(&sender, "GET", "/v1/models", "", false);
    assert!(matches!(&events[1], Event::Data(s) if s.contains("false")));
    let events = request(&sender, "POST", "/v1/models/load", "{", false);
    assert!(matches!(&events[0], Event::Headers(400, false, _)));
    let events = request(&sender, "GET", "/missing", "", false);
    assert!(matches!(&events[0], Event::Headers(404, false, _)));
    drop(sender);
    worker.join().unwrap();
}

#[test]
fn removed_models_and_media_routes_are_rejected() {
    let (sender, receiver) = queue::channel(8);
    let worker = thread::spawn(move || {
        worker_loop(receiver, &mut CpuBackend { resident: false }).unwrap();
    });
    for id in ["qwen38", "gemma4", "krea2", "moss", "h3"] {
        let body = format!(r#"{{"id":"{id}"}}"#);
        for route in ["/v1/models/load", "/v1/models/unload"] {
            let events = request(&sender, "POST", route, &body, false);
            assert!(matches!(events[0], Event::Headers(400, false, _)));
        }
        let body = format!(r#"{{"model":"{id}"}}"#);
        let events = request(&sender, "POST", "/v1/chat/completions", &body, false);
        assert!(matches!(events[0], Event::Headers(400, false, _)));
    }
    for (method, path) in [
        ("POST", "/v1/images/generations"),
        ("POST", "/v1/videos"),
        ("GET", "/v1/jobs/1/events"),
        ("DELETE", "/v1/jobs/1"),
    ] {
        let events = request(&sender, method, path, "{}", false);
        assert!(matches!(events[0], Event::Headers(404, false, _)));
    }
    drop(sender);
    worker.join().unwrap();
}

#[test]
fn chat_disconnect_stops_producer_and_worker_continues() {
    struct StreamBackend {
        produced: usize,
    }
    impl Backend for StreamBackend {
        fn models(&self) -> Result<String> {
            Ok("{}".into())
        }
        fn load(&mut self, _: &Value, _: &mut dyn FnMut(String) -> Result<()>) -> Result<()> {
            Ok(())
        }
        fn unload(&mut self, _: &Value) -> Result<()> {
            Ok(())
        }
        fn chat(
            &mut self,
            _: &Value,
            emit: &mut dyn FnMut(String) -> Result<()>,
        ) -> Result<String> {
            for _ in 0..1000 {
                // Also exercise keepalives for hidden reasoning/tool fragments.
                emit(String::new())?;
                self.produced += 1;
            }
            Ok("{}".into())
        }
    }
    let (sender, receiver) = queue::channel(2);
    let worker = thread::spawn(move || {
        let mut backend = StreamBackend { produced: 0 };
        worker_loop(receiver, &mut backend).unwrap();
        backend.produced
    });
    let (reply, events) = mpsc::sync_channel(8);
    sender
        .send(Job {
            work: Work::Http(
                http::Request {
                    method: "POST".into(),
                    path: "/v1/chat/completions".into(),
                    body: r#"{"model":"glm53","messages":[{"role":"user","content":"hi"}],"stream":true}"#.into(),
                    sse: false,
                    ..Default::default()
                },
                reply,
            ),
        })
        .unwrap();
    assert!(matches!(
        events.recv().unwrap(),
        Event::Headers(200, true, 0)
    ));
    drop(events);
    assert!(matches!(
        request(&sender, "GET", "/v1/models", "", false)[0],
        Event::Headers(200, false, _)
    ));
    drop(sender);
    assert!(worker.join().unwrap() < 1000);
}
/// GLM-5.3 on the ledger: 60 GiB resident outside the slot, elastic.
struct Elastic {
    glm: usize,
    default: usize,
}
impl Backend for Elastic {
    fn models(&self) -> Result<String> {
        Ok(format!("{}", self.glm >> 30))
    }
    fn load(&mut self, _: &Value, _: &mut dyn FnMut(String) -> Result<()>) -> Result<()> {
        Ok(())
    }
    fn unload(&mut self, _: &Value) -> Result<()> {
        Ok(())
    }
    fn ledger(&self) -> Vec<(&'static str, usize)> {
        vec![("glm53", self.glm)]
    }
    fn memory(&self) -> Option<(usize, usize)> {
        Some(((96 << 30) - self.glm, 96 << 30))
    }
    fn shrink_llm(&mut self, bytes: usize) -> Result<usize> {
        // Experts move in whole GiB images.
        let freed = bytes.div_ceil(1 << 30) << 30;
        self.glm -= freed;
        Ok(freed)
    }
    fn grow_llm(&mut self, free: usize) -> Result<usize> {
        let taken = (self.default - self.glm).min(free);
        self.glm += taken;
        Ok(taken)
    }
    fn budgets(&self) -> Vec<gpu::Budget> {
        vec![gpu::Budget {
            id: "glm53",
            budget_bytes: self.glm,
            default_bytes: self.default,
            experts_gpu: self.glm >> 30,
            experts_total: 12096,
            decode_tok_s: 10.,
        }]
    }
    fn chat(&mut self, _: &Value, _: &mut dyn FnMut(String) -> Result<()>) -> Result<String> {
        Ok("{}".into())
    }
}

#[test]
fn glm_shrinks_for_a_declared_need_and_grows_back_when_the_slot_is_free() {
    let dir = std::env::temp_dir().join(format!("hayate-elastic-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("gpu.d")).unwrap();
    let (sender, receiver) = queue::channel(8);
    let mut gpu = GpuLock::new(Some(dir.join("gpu.lock")), Some(Duration::from_millis(100)));
    let worker = thread::spawn(move || {
        worker_loop_locked(
            receiver,
            &mut Elastic {
                glm: 60 << 30,
                default: 60 << 30,
            },
            &mut gpu,
        )
    });
    let wait = |gib: &str| {
        let start = std::time::Instant::now();
        while sender.models().as_deref() != Some(gib) {
            assert!(start.elapsed() < Duration::from_secs(10), "{gib}");
            thread::sleep(Duration::from_millis(20));
        }
    };
    wait("60");
    // 36 GiB free, a waiter needs 50 + 1: GLM gives back 15 GiB.
    let pid = std::process::id();
    let declaration = dir.join("gpu.d").join(format!("{pid}.json"));
    std::fs::write(
        &declaration,
        format!(r#"{{"pid":{pid},"who":"hayate-train","need_gib":50,"since":1}}"#),
    )
    .unwrap();
    wait("45");
    let status = json::parse(&gpu::status(&sender.gpu().unwrap())).unwrap();
    let glm = &status
        .get("serve")
        .unwrap()
        .get("models")
        .unwrap()
        .array()
        .unwrap()[0];
    let budget = glm.get("budget").unwrap();
    assert!(budget.get("shrunk").unwrap().boolean().unwrap());
    assert_eq!(
        budget.get("budget_bytes").unwrap().integer().unwrap(),
        45 << 30
    );
    // The job (another process) runs: declaration gone, slot held. No growth.
    let mut job = std::process::Command::new("flock")
        .arg(dir.join("gpu.lock"))
        .args(["sleep", "1"])
        .spawn()
        .unwrap();
    let start = std::time::Instant::now();
    while !gpu::lock_rows(&dir.join("gpu.lock")).contains(&(job.id(), false)) {
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    std::fs::remove_file(&declaration).unwrap();
    thread::sleep(Duration::from_millis(500));
    assert_eq!(sender.models().unwrap(), "45");
    // Slot free, nobody waits: back to the default budget.
    job.wait().unwrap();
    wait("60");

    drop(sender);
    worker.join().unwrap().unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}
