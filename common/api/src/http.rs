//! Just enough HTTP/1.1 for an OpenAI-style API: keep-alive, Content-Length and chunked request
//! bodies, `Expect: 100-continue`, fixed-length responses and chunked responses (SSE), and a
//! check for a client that has gone away.

use std::io::{self, Read, Write};
use std::net::TcpStream;

const MAX_HEAD: usize = 64 * 1024;
const MAX_BODY: usize = 64 * 1024 * 1024;

#[derive(Debug)]
pub struct Request {
    pub method: String,
    /// Path without the query string.
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub keep_alive: bool,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// A request that could not be read: (status, message).
pub type HttpError = (u16, String);

pub struct Conn {
    pub stream: TcpStream,
    buf: Vec<u8>,
}

fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "",
    }
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

impl Conn {
    pub fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            buf: Vec::new(),
        }
    }

    fn fill(&mut self) -> io::Result<usize> {
        let mut tmp = [0u8; 16 * 1024];
        let n = self.stream.read(&mut tmp)?;
        self.buf.extend_from_slice(&tmp[..n]);
        Ok(n)
    }

    /// The next request; None when the client closed the connection between requests.
    pub fn read_request(&mut self) -> Result<Option<Request>, HttpError> {
        let io = |e: io::Error| (400, format!("read: {e}"));
        let end = loop {
            if let Some(p) = self.buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break p;
            }
            if self.buf.len() > MAX_HEAD {
                return Err((431, "request head too large".into()));
            }
            if self.fill().map_err(io)? == 0 {
                return if self.buf.is_empty() {
                    Ok(None)
                } else {
                    Err((400, "connection closed mid-request".into()))
                };
            }
        };
        let head = String::from_utf8_lossy(&self.buf[..end]).into_owned();
        self.buf.drain(..end + 4);
        let mut lines = head.split("\r\n");
        let mut first = lines.next().unwrap_or("").split(' ');
        let (method, target, version) = (
            first.next().unwrap_or(""),
            first.next().unwrap_or(""),
            first.next().unwrap_or(""),
        );
        if method.is_empty() || !target.starts_with('/') || !version.starts_with("HTTP/1.") {
            return Err((400, "malformed request line".into()));
        }
        let headers: Vec<(String, String)> = lines
            .filter_map(|l| l.split_once(':'))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect();
        let mut req = Request {
            method: method.to_string(),
            path: target.split('?').next().unwrap_or("").to_string(),
            headers,
            body: Vec::new(),
            keep_alive: true,
        };
        let conn = req
            .header("connection")
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        req.keep_alive = if version == "HTTP/1.0" {
            conn.contains("keep-alive")
        } else {
            !conn.contains("close")
        };
        if req
            .header("expect")
            .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
        {
            self.stream
                .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
                .map_err(io)?;
        }
        if req
            .header("transfer-encoding")
            .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"))
        {
            req.body = self
                .read_chunked()
                .map_err(|e| (400, format!("chunked body: {e}")))?;
        } else if let Some(cl) = req.header("content-length") {
            let n: usize = cl
                .parse()
                .map_err(|_| (400, "bad Content-Length".to_string()))?;
            if n > MAX_BODY {
                return Err((413, "request body too large".into()));
            }
            while self.buf.len() < n {
                if self.fill().map_err(io)? == 0 {
                    return Err((400, "connection closed mid-body".into()));
                }
            }
            req.body = self.buf.drain(..n).collect();
        }
        Ok(Some(req))
    }

    fn read_line(&mut self) -> io::Result<String> {
        loop {
            if let Some(p) = self.buf.windows(2).position(|w| w == b"\r\n") {
                let line = String::from_utf8_lossy(&self.buf[..p]).into_owned();
                self.buf.drain(..p + 2);
                return Ok(line);
            }
            if self.buf.len() > MAX_HEAD || self.fill()? == 0 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "bad chunk line"));
            }
        }
    }

    fn read_chunked(&mut self) -> io::Result<Vec<u8>> {
        let mut body = Vec::new();
        loop {
            let line = self.read_line()?;
            let size = usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad chunk size"))?;
            if size == 0 {
                while !self.read_line()?.is_empty() {} // trailers
                return Ok(body);
            }
            if body.len() + size > MAX_BODY {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "body too large"));
            }
            while self.buf.len() < size + 2 {
                if self.fill()? == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "closed mid-chunk",
                    ));
                }
            }
            body.extend(self.buf.drain(..size));
            self.buf.drain(..2);
        }
    }

    pub fn respond(
        &mut self,
        status: u16,
        content_type: &str,
        body: &[u8],
        keep_alive: bool,
    ) -> io::Result<()> {
        let head = format!(
            "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{CORS}Connection: {}\r\n\r\n",
            reason(status),
            body.len(),
            if keep_alive { "keep-alive" } else { "close" }
        );
        let mut out = head.into_bytes();
        out.extend_from_slice(body);
        self.stream.write_all(&out)
    }

    /// Starts a chunked response (server-sent events).
    pub fn start_stream(&mut self, content_type: &str) -> io::Result<()> {
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n{CORS}Connection: keep-alive\r\n\r\n"
        );
        self.stream.write_all(head.as_bytes())
    }

    pub fn chunk(&mut self, data: &[u8]) -> io::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let mut out = format!("{:x}\r\n", data.len()).into_bytes();
        out.extend_from_slice(data);
        out.extend_from_slice(b"\r\n");
        self.stream.write_all(&out)
    }

    pub fn end_stream(&mut self) -> io::Result<()> {
        self.stream.write_all(b"0\r\n\r\n")
    }

    /// True when the client has closed its side (or the socket failed).
    pub fn peer_closed(&self) -> bool {
        if self.stream.set_nonblocking(true).is_err() {
            return false;
        }
        let mut b = [0u8; 1];
        let r = self.stream.peek(&mut b);
        let _ = self.stream.set_nonblocking(false);
        match r {
            Ok(0) => true,
            Ok(_) => false,
            Err(e) => !matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ),
        }
    }
}
