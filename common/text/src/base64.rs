//! Strict RFC 4648 image transport encoding.
pub fn encode(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = ((c[0] as u32) << 16)
            | ((c.get(1).copied().unwrap_or(0) as u32) << 8)
            | c.get(2).copied().unwrap_or(0) as u32;
        s.push(TABLE[(n >> 18) as usize] as char);
        s.push(TABLE[((n >> 12) & 63) as usize] as char);
        s.push(if c.len() > 1 {
            TABLE[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        s.push(if c.len() > 2 {
            TABLE[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    s
}

pub fn decode(s: &str) -> Result<Vec<u8>, String> {
    if s.len() > 32 << 20 {
        return Err("encoded image exceeds 32 MiB".into());
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits, mut padding) = (0u32, 0, 0);
    if !s.len().is_multiple_of(4) {
        return Err("invalid base64 length".into());
    }
    for b in s.bytes() {
        if b == b'=' {
            padding += 1;
            if padding > 2 {
                return Err("invalid base64 padding".into());
            }
            continue;
        }
        if padding > 0 {
            return Err("base64 data after padding".into());
        }
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err("invalid base64 image".into()),
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    if acc != 0
        || (padding, bits) != (0, 0) && (padding, bits) != (1, 2) && (padding, bits) != (2, 4)
    {
        return Err("invalid base64 padding bits".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_roundtrip() {
        for n in 0..512 {
            let b = (0..n).map(|i| i as u8).collect::<Vec<_>>();
            assert_eq!(decode(&encode(&b)).unwrap(), b);
        }
    }
}
