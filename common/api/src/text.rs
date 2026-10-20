//! Output text processing between the detokenizer and the client: stop strings, and splitting
//! `<think>…</think>` into `reasoning_content` the way the Qwen3 reasoning parsers of SGLang and
//! vLLM do. Both hold back only as much text as a match could still need, so streaming and
//! non-streaming responses carry the same text.

/// Finds the first stop string in the output; holds back a tail that could start one.
#[derive(Debug, Default)]
pub struct StopStrings {
    stops: Vec<String>,
    held: String,
}

impl StopStrings {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            held: String::new(),
        }
    }

    /// Text that is safe to release, and whether a stop string was found (the text before it is
    /// released; the stop string and anything after are dropped).
    pub fn push(&mut self, s: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (s.to_string(), false);
        }
        self.held.push_str(s);
        if let Some(pos) = self
            .stops
            .iter()
            .filter_map(|st| self.held.find(st.as_str()))
            .min()
        {
            let out = self.held[..pos].to_string();
            self.held.clear();
            return (out, true);
        }
        // Keep the longest suffix that is a proper prefix of some stop string.
        let keep = self
            .held
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| self.stops.iter().any(|st| st.starts_with(&self.held[i..])))
            .unwrap_or(self.held.len());
        let out = self.held[..keep].to_string();
        self.held.drain(..keep);
        (out, false)
    }

    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

const THINK_END: &str = "</think>";
const THINK_START: &str = "<think>";

/// Splits generated text into reasoning and answer. With thinking on, the prompt ends inside
/// `<think>\n`, so the output starts as reasoning until `</think>`. Reasoning is trimmed at both
/// ends, the answer at its start (as the template trims them when they are fed back).
#[derive(Debug)]
pub struct ThinkSplit {
    in_reasoning: bool,
    held: String,
    /// Something non-blank was released in the current section.
    started: bool,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Delta {
    pub reasoning: String,
    pub content: String,
}

impl Delta {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty()
    }
}

impl ThinkSplit {
    pub fn new(reasoning_first: bool) -> Self {
        Self {
            in_reasoning: reasoning_first,
            held: String::new(),
            started: false,
        }
    }

    pub fn push(&mut self, s: &str) -> Delta {
        let mut d = Delta::default();
        self.held.push_str(s);
        loop {
            if !self.in_reasoning {
                let t = if self.started {
                    &self.held[..]
                } else {
                    self.held.trim_start()
                };
                if !t.is_empty() {
                    self.started = true;
                    d.content.push_str(t);
                }
                self.held.clear();
                return d;
            }
            if let Some(pos) = self.held.find(THINK_END) {
                let mut r = self.held[..pos].trim_end();
                if !self.started {
                    r = r.trim_start();
                    r = r.strip_prefix(THINK_START).map_or(r, str::trim_start);
                }
                d.reasoning.push_str(r);
                self.held.drain(..pos + THINK_END.len());
                self.in_reasoning = false;
                self.started = false;
                continue;
            }
            // Release up to a partial `</think>` at the end and the blanks before it.
            let partial = self
                .held
                .char_indices()
                .map(|(i, _)| i)
                .find(|&i| THINK_END.starts_with(&self.held[i..]));
            let keep = self.held[..partial.unwrap_or(self.held.len())]
                .trim_end()
                .len();
            let mut out = &self.held[..keep];
            if !self.started {
                out = out.trim_start();
                // A `<think>` the model wrote itself (the prompt has one already).
                if let Some(rest) = out.strip_prefix(THINK_START) {
                    out = rest.trim_start();
                } else if THINK_START.starts_with(out) && !out.is_empty() {
                    return d; // could still become `<think>`
                }
            }
            if !out.is_empty() {
                self.started = true;
                d.reasoning.push_str(out);
            }
            self.held.drain(..keep);
            return d;
        }
    }

    /// The rest at the end of generation (trailing blanks of unfinished reasoning are dropped).
    pub fn finish(&mut self) -> Delta {
        let mut d = Delta::default();
        let rest = std::mem::take(&mut self.held);
        let mut t = if self.in_reasoning {
            rest.trim_end()
        } else {
            &rest[..]
        };
        if !self.started {
            t = t.trim_start();
            if self.in_reasoning {
                t = t.strip_prefix(THINK_START).map_or(t, str::trim_start);
            }
        }
        if self.in_reasoning {
            d.reasoning.push_str(t)
        } else {
            d.content.push_str(t)
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(parts: &[&str], think: bool) -> Delta {
        let mut s = ThinkSplit::new(think);
        let mut all = Delta::default();
        for p in parts {
            let d = s.push(p);
            all.reasoning.push_str(&d.reasoning);
            all.content.push_str(&d.content);
        }
        let d = s.finish();
        all.reasoning.push_str(&d.reasoning);
        all.content.push_str(&d.content);
        all
    }

    #[test]
    fn think_split() {
        let want = Delta {
            reasoning: "a <b> c".into(),
            content: "Answer.\n".into(),
        };
        assert_eq!(run(&["a <b> c\n", "</think>", "\n\nAnswer.\n"], true), want);
        assert_eq!(
            run(
                &[
                    "a", " <b", "> c", "\n", "</th", "ink>\n", "\n", "Answer", ".\n"
                ],
                true
            ),
            want
        );
        assert_eq!(
            run(&["<think>\na <b> c\n</think>\n\nAnswer.\n"], true),
            want
        );
        assert_eq!(
            run(&["Hi"], false),
            Delta {
                reasoning: String::new(),
                content: "Hi".into()
            }
        );
        assert_eq!(
            run(&["still thinking \n"], true),
            Delta {
                reasoning: "still thinking".into(),
                content: String::new()
            }
        );
    }

    #[test]
    fn stops() {
        let mut s = StopStrings::new(vec!["END".into(), "\n\n".into()]);
        assert_eq!(s.push("abc E"), ("abc ".to_string(), false));
        assert_eq!(s.push("Nx"), ("ENx".to_string(), false));
        assert_eq!(s.push("y\n"), ("y".to_string(), false));
        assert_eq!(s.push("\nz"), (String::new(), true));
    }
}
