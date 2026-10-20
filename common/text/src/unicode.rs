//! Unicode pieces the tokenizer and the chat template need: general categories, the two
//! whitespace sets (Oniguruma's `\s` and Python's `str.isspace`), and NFC.

use crate::unicode_data::{CATS, CCC, COMP, DECOMP};
use std::borrow::Cow;
use std::cmp::Ordering;

fn lookup(table: &[(u32, u32, u8)], c: u32) -> u8 {
    let r = table.binary_search_by(|&(lo, hi, _)| {
        if hi < c {
            Ordering::Less
        } else if lo > c {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
    r.map_or(0, |i| table[i].2)
}

pub const L: u8 = 1;
pub const M: u8 = 2;
pub const N: u8 = 4;

/// General category bits: L (letter), M (mark), N (number), or 0.
pub fn category(c: char) -> u8 {
    match c {
        'a'..='z' | 'A'..='Z' => L,
        '0'..='9' => N,
        '\0'..='\u{7f}' => 0,
        _ => lookup(CATS, c as u32),
    }
}

/// Oniguruma's `\s` for Unicode text: \t \n \v \f \r, U+0085, and the Z separators.
pub fn is_space_re(c: char) -> bool {
    matches!(c, '\t'..='\r' | ' ' | '\u{85}' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

/// Python's `str.isspace` (what Jinja's `trim` strips): `\s` plus U+001C..U+001F.
pub fn is_space_py(c: char) -> bool {
    is_space_re(c) || matches!(c, '\u{1c}'..='\u{1f}')
}

/// Python's `str.strip()`.
pub fn py_trim(s: &str) -> &str {
    s.trim_matches(is_space_py)
}

fn ccc(c: char) -> u8 {
    if (c as u32) < 0x300 {
        0
    } else {
        lookup(CCC, c as u32)
    }
}

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const S_COUNT: u32 = L_COUNT * V_COUNT * T_COUNT;

fn decompose(c: char, out: &mut Vec<char>) {
    let cp = c as u32;
    if (S_BASE..S_BASE + S_COUNT).contains(&cp) {
        let s = cp - S_BASE;
        out.push(char::from_u32(L_BASE + s / (V_COUNT * T_COUNT)).unwrap());
        out.push(char::from_u32(V_BASE + (s % (V_COUNT * T_COUNT)) / T_COUNT).unwrap());
        if !s.is_multiple_of(T_COUNT) {
            out.push(char::from_u32(T_BASE + s % T_COUNT).unwrap());
        }
        return;
    }
    if cp >= 0xC0
        && let Ok(i) = DECOMP.binary_search_by_key(&cp, |e| e.0)
    {
        let (_, a, b) = DECOMP[i];
        decompose(char::from_u32(a).unwrap(), out);
        if b != 0 {
            decompose(char::from_u32(b).unwrap(), out);
        }
        return;
    }
    out.push(c);
}

fn compose_pair(a: char, b: char) -> Option<char> {
    let (a, b) = (a as u32, b as u32);
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return char::from_u32(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a)
        && (a - S_BASE).is_multiple_of(T_COUNT)
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&b)
    {
        return char::from_u32(a + (b - T_BASE));
    }
    COMP.binary_search_by(|e| (e.0, e.1).cmp(&(a, b)))
        .ok()
        .map(|i| char::from_u32(COMP[i].2).unwrap())
}

/// Unicode Normalization Form C.
pub fn nfc(s: &str) -> Cow<'_, str> {
    // Nothing below U+0300 decomposes to something different or composes with a predecessor.
    if s.chars().all(|c| (c as u32) < 0x300) {
        return Cow::Borrowed(s);
    }
    let mut d = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose(c, &mut d);
    }
    // Canonical ordering: stable sort of each run of non-starters by combining class.
    let mut i = 0;
    while i < d.len() {
        if ccc(d[i]) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < d.len() && ccc(d[i]) != 0 {
            i += 1;
        }
        d[start..i].sort_by_key(|&c| ccc(c));
    }
    // Canonical composition.
    let mut out: Vec<char> = Vec::with_capacity(d.len());
    let mut starter: Option<usize> = None;
    // Combining class of the last character after the starter (-1: none, so adjacent).
    let mut last: i32 = -1;
    for c in d {
        let cc = i32::from(ccc(c));
        if let Some(si) = starter
            && (last == -1 || last < cc)
            && let Some(x) = compose_pair(out[si], c)
        {
            out[si] = x;
            continue;
        }
        if cc == 0 {
            starter = Some(out.len());
            last = -1;
        } else {
            last = cc;
        }
        out.push(c);
    }
    Cow::Owned(out.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfc_cases() {
        assert_eq!(nfc("e\u{301}"), "é");
        assert_eq!(nfc("か\u{3099}"), "が");
        assert_eq!(nfc("\u{1100}\u{1161}\u{11A8}"), "각");
        assert_eq!(nfc("\u{212B}"), "Å");
        assert_eq!(nfc("a\u{323}\u{302}"), "ậ"); // reordered, then composed twice
        assert_eq!(nfc("\u{958}"), "\u{915}\u{93c}"); // composition exclusion
        assert_eq!(category('あ'), L);
        assert_eq!(category('\u{301}'), M);
        assert_eq!(category('٣'), N);
        assert_eq!(py_trim("\u{1c} x \u{3000}"), "x");
    }
}
