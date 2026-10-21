//! One HTTP/1.1 request per connection; Content-Length in, close-delimited SSE out.
use hayate_core::{Error, Result};
use std::io::{BufRead, BufReader, Read, Write};

#[derive(Debug, Default)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub sse: bool,
    pub body: String,
    /// Set by the connection thread when the client closes the connection.
    pub gone: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
fn bad(message: &str) -> Error {
    Error::InvalidPlan(format!("HTTP: {message}"))
}
pub fn read(reader: impl Read) -> Result<Request> {
    let mut reader = BufReader::new(reader);
    let mut bytes = 0;
    let mut line = String::new();
    fn line_in(r: &mut impl BufRead, line: &mut String, bytes: &mut usize) -> Result<()> {
        line.clear();
        let n = r
            .take((16385usize.saturating_sub(*bytes)) as u64)
            .read_line(line)?;
        *bytes += n;
        if n == 0 || *bytes > 16384 || !line.ends_with("\r\n") {
            return Err(bad("invalid or oversized headers"));
        }
        Ok(())
    }
    line_in(&mut reader, &mut line, &mut bytes)?;
    let mut words = line.trim_end().split(' ');
    let method = words.next().ok_or_else(|| bad("request line"))?.to_owned();
    let path = words
        .next()
        .ok_or_else(|| bad("request target"))?
        .to_owned();
    if words.next() != Some("HTTP/1.1") || words.next().is_some() || !path.starts_with('/') {
        return Err(bad("request line"));
    }
    let mut length = None;
    let mut host = false;
    let mut sse = false;
    loop {
        line_in(&mut reader, &mut line, &mut bytes)?;
        if line == "\r\n" {
            break;
        }
        let (name, value) = line[..line.len() - 2]
            .split_once(':')
            .ok_or_else(|| bad("header"))?;
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Err(bad("header name"));
        }
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            if length.is_some() || value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad("content-length"));
            }
            length = Some(value.parse::<usize>().map_err(|_| bad("content-length"))?);
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            return Err(bad("transfer-encoding is unsupported"));
        } else if name.eq_ignore_ascii_case("expect") {
            return Err(bad("Expect is unsupported"));
        } else if name.eq_ignore_ascii_case("host") {
            if host || value.is_empty() {
                return Err(bad("host"));
            }
            host = true;
        } else if name.eq_ignore_ascii_case("accept") {
            sse |= value
                .split(',')
                .any(|v| v.trim().split(';').next() == Some("text/event-stream"));
        }
    }
    if !host || (method == "POST" && length.is_none()) {
        return Err(bad("Host and POST Content-Length required"));
    }
    let length = length.unwrap_or(0);
    if length > 64 * 1024 * 1024 {
        return Err(bad("body exceeds 16 MiB"));
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    let body = String::from_utf8(body).map_err(|_| bad("body must be UTF-8"))?;
    Ok(Request {
        method,
        path,
        sse,
        body,
        gone: Default::default(),
    })
}
pub fn headers(mut out: impl Write, status: u16, sse: bool, length: usize) -> Result<()> {
    let reason = match status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    write!(
        out,
        "HTTP/1.1 {status} {reason}\r\nConnection: close\r\nContent-Type: {}\r\n",
        if sse {
            "text/event-stream"
        } else {
            "application/json"
        }
    )?;
    if sse {
        write!(out, "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n")?;
    } else {
        write!(out, "Content-Length: {length}\r\n")?;
    }
    write!(out, "\r\n")?;
    out.flush()?;
    Ok(())
}
pub fn sse(mut out: impl Write, data: &str) -> Result<()> {
    for line in data.split('\n') {
        writeln!(out, "data: {line}")?;
    }
    writeln!(out)?;
    out.flush()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_and_smuggling_rejection() {
        let r = read(&b"POST /v1/models/load HTTP/1.1\r\nHost: localhost\r\ncOnTeNt-LeNgTh: 2\r\nAccept: text/event-stream\r\n\r\n{}"[..]).unwrap();
        assert_eq!(r.body, "{}");
        assert!(r.sse);
        for s in [
            "POST / HTTP/1.1\r\nHost: x\r\n\r\n",
            "GET / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n",
            "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n",
            "GET / HTTP/1.1\r\n\r\n",
            "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{",
        ] {
            assert!(read(s.as_bytes()).is_err(), "{s}");
        }
    }
    #[test]
    fn sse_bytes_and_response_length() {
        let mut out = vec![];
        sse(&mut out, "{\"x\":1}").unwrap();
        sse(&mut out, "[DONE]").unwrap();
        assert_eq!(out, b"data: {\"x\":1}\n\ndata: [DONE]\n\n");
        out.clear();
        headers(&mut out, 200, false, 2).unwrap();
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("Content-Length: 2\r\n\r\n")
        );
    }
}
