//! Post-processing of OCR readings: licence-plate format validation / correction, and grouping the readings of a
//! series of frames into tracks so that each thing seen is reported once, with the text decided by a weighted vote.
use crate::ocr::Rect;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Text,
    Plate,
}

/// One OCR reading on one frame.
#[derive(Clone, Debug)]
pub struct Reading {
    pub frame: u64,
    pub rect: Rect,
    pub text: String,
    pub conf: f32,
    pub kind: Kind,
    /// Plates: probability of every character in every slot (`PLATE_SLOTS × alphabet`); empty when unknown.
    pub slots: Vec<f32>,
    /// Plates: sharpness of the crop, 0..1 (blurred readings weigh less in the vote).
    pub quality: f32,
}

/// A thing seen on one or more frames, with the voted text.
#[derive(Clone, Debug)]
pub struct Hit {
    pub kind: Kind,
    pub text: String,
    /// 0..1: confidence of the voted text.
    pub conf: f32,
    /// Position on the `best` frame.
    pub rect: Rect,
    /// Position on every frame where it was read (frame, rect).
    pub trail: Vec<(u64, Rect)>,
    pub first: u64,
    pub last: u64,
    /// Frame holding the reading closest to the voted text.
    pub best: u64,
    pub readings: usize,
    pub format: Option<&'static str>,
    /// Other distinct readings (most supported first).
    pub alts: Vec<String>,
}

// ------------------------------------------------------------------ plate formats

/// (name, patterns). A pattern is a sequence of `L` (letter) / `D` (digit) runs with a length or a range, e.g. `L2D3L2`
/// or `D2-3L1-2D1-6`.
const FORMATS: &[(&str, &[&str])] = &[
    ("IT/FR AA 123 AA", &["L2D3L2"]),
    ("UK AA 12 AAA", &["L2D2L3"]),
    ("IT storica", &["L2D5-6"]),
    ("FR vecchia", &["D3L2-3D2"]),
    ("ES 1234 ABC", &["D4L3"]),
    ("PT", &["D2L2D2", "L2D2L2", "D4L2", "L2D4"]),
    ("NL", &["L2D4", "D4L2", "D2L2D2", "L2D2L2", "D2L3D1", "D1L3D2", "L2D3L1", "L1D3L2", "L3D2L1", "L1D2L3", "D1L2D3"]),
    ("DE", &["L2-5D1-4"]),
    ("UK vecchia", &["L1D1-3L3", "L3D1-3L1"]),
    ("IE", &["D2-3L1-2D1-6"]),
    ("CH", &["L2D1-6"]),
    ("BE", &["D1L3D3", "L3D3"]),
];

/// Expands a pattern into every class mask it describes.
fn expand(pattern: &str) -> Vec<String> {
    let b = pattern.as_bytes();
    let (mut masks, mut i) = (vec![String::new()], 0);
    let num = |i: &mut usize| {
        let mut n = 0usize;
        while *i < b.len() && b[*i].is_ascii_digit() {
            n = n * 10 + (b[*i] - b'0') as usize;
            *i += 1;
        }
        n
    };
    while i < b.len() {
        let class = b[i] as char;
        i += 1;
        let lo = num(&mut i);
        let mut hi = lo;
        if i < b.len() && b[i] == b'-' {
            i += 1;
            hi = num(&mut i);
        }
        let mut next = Vec::new();
        for m in &masks {
            for n in lo..=hi {
                next.push(format!("{m}{}", class.to_string().repeat(n)));
            }
        }
        masks = next;
    }
    masks.retain(|m| (3..=9).contains(&m.len()));
    masks
}

fn formats() -> &'static Vec<(&'static str, Vec<String>)> {
    static F: OnceLock<Vec<(&'static str, Vec<String>)>> = OnceLock::new();
    F.get_or_init(|| FORMATS.iter().map(|(name, pats)| (*name, pats.iter().flat_map(|p| expand(p)).collect())).collect())
}

fn mask(s: &str) -> Option<String> {
    s.chars().map(|c| if c.is_ascii_digit() { Some('D') } else if c.is_ascii_alphabetic() { Some('L') } else { None }).collect()
}

/// Name of the national format that `text` (upper-case letters and digits only) matches, if any.
pub fn plate_format(text: &str) -> Option<&'static str> {
    let m = mask(text)?;
    let find = |m: &str| formats().iter().find(|(_, masks)| masks.iter().any(|x| x == m)).map(|(name, _)| *name);
    if let Some(n) = find(&m) {
        return Some(n);
    }
    // German plates may end with an E / H suffix (electric / historic)
    if m.len() >= 4 && m.ends_with("DL") && matches!(text.chars().last(), Some('E' | 'H')) {
        return find(&m[..m.len() - 1]).filter(|n| *n == "DE");
    }
    None
}

fn to_digit(c: char) -> Option<char> {
    Some(match c {
        'O' | 'Q' | 'D' => '0',
        'I' | 'L' => '1',
        'Z' => '2',
        'S' => '5',
        'G' => '6',
        'T' => '7',
        'B' => '8',
        _ => return None,
    })
}

fn to_letter(c: char) -> Option<char> {
    Some(match c {
        '0' => 'O',
        '1' => 'I',
        '2' => 'Z',
        '5' => 'S',
        '6' => 'G',
        '7' => 'T',
        '8' => 'B',
        _ => return None,
    })
}

/// Tries to turn an invalid plate into a valid one by swapping look-alike characters (O/0, I/1, B/8, S/5, ...).
/// Returns the corrected text and the format, using at most two substitutions.
pub fn correct_plate(text: &str) -> Option<(String, &'static str)> {
    let chars: Vec<char> = text.chars().collect();
    let mut best: Option<(usize, String, &'static str)> = None;
    for (name, masks) in formats() {
        for m in masks.iter().filter(|m| m.len() == chars.len()) {
            let (mut out, mut changes) = (String::new(), 0);
            let mut ok = true;
            for (c, want) in chars.iter().zip(m.chars()) {
                let is_digit = c.is_ascii_digit();
                if is_digit == (want == 'D') {
                    out.push(*c);
                } else if let Some(r) = if want == 'D' { to_digit(*c) } else { to_letter(*c) } {
                    out.push(r);
                    changes += 1;
                } else {
                    ok = false;
                    break;
                }
            }
            if ok && changes <= 2 && best.as_ref().is_none_or(|b| changes < b.0) {
                best = Some((changes, out, name));
            }
        }
    }
    best.map(|(_, t, n)| (t, n))
}

// ------------------------------------------------------------ plate probability decoding

const ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
const VOCAB: usize = 37;
const SLOTS: usize = 9;
const PAD: usize = 36;

/// Log-probability handicap of formats that share their shape with a more common one, so ties go to the common one
/// (`BL600BC` is Italian/French, not British).
fn format_penalty(name: &str) -> f32 {
    match name {
        "UK AA 12 AAA" | "UK vecchia" | "IE" | "CH" | "BE" => 0.7,
        _ => 0.0,
    }
}

/// How much worse (in nats of log-probability) a format-conforming reading may be than the unconstrained best one and
/// still be preferred. 3.0 ≈ the format may overrule characters the model was 20× more sure of.
const FORMAT_MARGIN: f32 = 3.0;

/// Decodes per-slot character probabilities into a plate text. The best reading that fits a national format is
/// preferred over the unconstrained one when it is not much less likely, so O/0, I/1, B/8 confusions are settled by the
/// format. Returns the text, a 0..1 confidence and the format when one fits.
pub fn decode_plate(p: &[f32]) -> (String, f32, Option<&'static str>) {
    if p.len() != SLOTS * VOCAB {
        return (String::new(), 0.0, None);
    }
    let ln = |x: f32| (x + 1e-4).ln();
    let slot = |i: usize| &p[i * VOCAB..(i + 1) * VOCAB];
    let argmax = |s: &[f32], lo: usize, hi: usize| (lo..hi).max_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap()).unwrap();

    // unconstrained
    let (mut free, mut free_score, mut free_conf, mut free_n) = (String::new(), 0.0f32, 0.0f32, 0u32);
    for i in 0..SLOTS {
        let idx = argmax(slot(i), 0, VOCAB);
        free_score += ln(slot(i)[idx]);
        if idx != PAD {
            free.push(ALPHABET[idx] as char);
            free_conf += slot(i)[idx];
            free_n += 1;
        }
    }
    let free_conf = if free_n == 0 { 0.0 } else { free_conf / free_n as f32 };

    // best format-conforming reading
    let mut best: Option<(f32, String, f32, &'static str)> = None;
    for (name, masks) in formats() {
        for m in masks {
            let (mut score, mut text, mut conf) = (0.0f32, String::new(), 0.0f32);
            for i in 0..SLOTS {
                match m.as_bytes().get(i) {
                    Some(b'D') => {
                        let idx = argmax(slot(i), 0, 10);
                        score += ln(slot(i)[idx]);
                        conf += slot(i)[idx];
                        text.push(ALPHABET[idx] as char);
                    }
                    Some(_) => {
                        let idx = argmax(slot(i), 10, 36);
                        score += ln(slot(i)[idx]);
                        conf += slot(i)[idx];
                        text.push(ALPHABET[idx] as char);
                    }
                    None => score += ln(slot(i)[PAD]),
                }
            }
            let score = score - format_penalty(name);
            if best.as_ref().is_none_or(|b| score > b.0) {
                best = Some((score, text, conf / m.len() as f32, name));
            }
        }
    }
    if let Some((score, text, conf, name)) = best {
        if score >= free_score - FORMAT_MARGIN {
            return (text, conf, Some(name));
        }
    }
    let format = plate_format(&free);
    (free, free_conf, format)
}

// ---------------------------------------------------------------------- similarity

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            cur.push((prev[j] + (ca != cb) as usize).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// 1.0 = identical, 0.0 = nothing in common.
pub fn similarity(a: &str, b: &str) -> f32 {
    let (a, b): (Vec<char>, Vec<char>) = (a.to_lowercase().chars().collect(), b.to_lowercase().chars().collect());
    let n = a.len().max(b.len());
    if n == 0 {
        1.0
    } else {
        1.0 - levenshtein(&a, &b) as f32 / n as f32
    }
}

// ------------------------------------------------------------------------ tracks

/// Decides the text of a group of readings of the same thing: pick the reading best supported by the others, then vote
/// each character among the readings of the same length (weighted by confidence).
fn vote(rs: &[&Reading], plate: bool) -> (String, f32, usize) {
    // for plates, readings that fit a national format win over the others
    let valid: Vec<&Reading> = if plate { rs.iter().copied().filter(|r| plate_format(&r.text).is_some()).collect() } else { vec![] };
    let pool: &[&Reading] = if valid.is_empty() { rs } else { &valid };
    let support = |i: usize| -> f32 { pool.iter().map(|o| o.conf.max(0.01) * similarity(&pool[i].text, &o.text)).sum() };
    let medoid = (0..pool.len()).max_by(|&a, &b| support(a).partial_cmp(&support(b)).unwrap()).unwrap();
    let len = pool[medoid].text.chars().count();
    let same: Vec<&&Reading> = pool.iter().filter(|r| r.text.chars().count() == len).collect();
    let mut text = String::new();
    for i in 0..len {
        let mut tally: Vec<(char, f32)> = Vec::new();
        for r in &same {
            let c = r.text.chars().nth(i).unwrap();
            match tally.iter_mut().find(|t| t.0 == c) {
                Some(t) => t.1 += r.conf.max(0.01),
                None => tally.push((c, r.conf.max(0.01))),
            }
        }
        text.push(tally.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0);
    }
    let agree: Vec<&&&Reading> = same.iter().filter(|r| similarity(&r.text, &text) >= 0.99).collect();
    let conf = if agree.is_empty() { same.iter().map(|r| r.conf).sum::<f32>() / same.len() as f32 } else { agree.iter().map(|r| r.conf).sum::<f32>() / agree.len() as f32 };
    (text, conf, agree.len())
}

/// Plates: averages the per-character probabilities of every reading (sharp, confident readings weigh more) and
/// decodes the result, so blurred frames add evidence instead of competing as whole strings.
fn vote_plate(rs: &[&Reading]) -> Option<(String, f32, Option<&'static str>)> {
    if rs.iter().any(|r| r.slots.len() != SLOTS * VOCAB) {
        return None;
    }
    let mut acc = vec![0.0f32; SLOTS * VOCAB];
    let mut wsum = 0.0f32;
    for r in rs {
        let w = r.quality.max(0.05) * r.conf.max(0.05);
        wsum += w;
        for (a, p) in acc.iter_mut().zip(&r.slots) {
            *a += w * p;
        }
    }
    acc.iter_mut().for_each(|a| *a /= wsum);
    Some(decode_plate(&acc))
}

struct Track {
    kind: Kind,
    items: Vec<Reading>,
}

impl Track {
    fn last(&self) -> &Reading {
        self.items.last().unwrap()
    }

    /// Where the track should be on `frame` assuming constant velocity (last two readings).
    fn predict(&self, frame: u64) -> (f32, f32) {
        let l = self.last();
        let c = l.rect.center();
        if self.items.len() < 2 {
            return c;
        }
        let p = &self.items[self.items.len() - 2];
        let (pc, df) = (p.rect.center(), (l.frame - p.frame).max(1) as f32);
        let dt = (frame - l.frame) as f32;
        (c.0 + (c.0 - pc.0) / df * dt, c.1 + (c.1 - pc.1) / df * dt)
    }
}

/// Groups readings by position/text over time and reports one `Hit` per group.
/// `max_gap` is how many frames a thing may disappear before a new group is started.
pub fn aggregate(mut readings: Vec<Reading>, max_gap: u64) -> Vec<Hit> {
    readings.sort_by_key(|r| r.frame);
    let mut tracks: Vec<Track> = Vec::new();
    for r in readings {
        let mut best: Option<(usize, f32)> = None;
        for (i, t) in tracks.iter().enumerate() {
            let l = t.last();
            if t.kind != r.kind || r.frame < l.frame || r.frame - l.frame > max_gap || l.frame == r.frame {
                continue;
            }
            let sim = similarity(&l.text, &r.text);
            let iou = l.rect.iou(&r.rect);
            let wmax = l.rect.w().max(r.rect.w()).max(1e-6);
            let c2 = r.rect.center();
            let c1 = l.rect.center();
            let near = ((c1.0 - c2.0).powi(2) + (c1.1 - c2.1).powi(2)).sqrt() < 3.0 * wmax;
            let pc = t.predict(r.frame);
            let on_path = ((pc.0 - c2.0).powi(2) + (pc.1 - c2.1).powi(2)).sqrt() < 0.8 * wmax;
            let ratio = l.rect.w() / r.rect.w().max(1e-6);
            let size_ok = (0.5..=2.0).contains(&ratio);
            let score = if iou > 0.3 {
                iou + sim
            } else if sim >= 0.5 && (near || on_path) && size_ok {
                sim
            } else if on_path && size_ok && sim >= 0.3 {
                0.3 + sim * 0.5
            } else {
                continue;
            };
            if best.is_none_or(|b| score > b.1) {
                best = Some((i, score));
            }
        }
        match best {
            Some((i, _)) => tracks[i].items.push(r),
            None => tracks.push(Track { kind: r.kind, items: vec![r] }),
        }
    }
    let mut hits: Vec<Hit> = tracks
        .iter()
        .map(|t| {
            let refs: Vec<&Reading> = t.items.iter().collect();
            let plate = t.kind == Kind::Plate;
            let (text, conf, format) = match if plate { vote_plate(&refs) } else { None } {
                Some(v) => v,
                None => {
                    let (mut text, conf, _) = vote(&refs, plate);
                    let mut format = None;
                    if plate {
                        format = plate_format(&text);
                        if format.is_none() {
                            if let Some((fixed, name)) = correct_plate(&text) {
                                text = fixed;
                                format = Some(name);
                            }
                        }
                    }
                    (text, conf, format)
                }
            };
            let score = |r: &Reading| similarity(&r.text, &text) * r.conf * r.quality.max(0.05);
            let best = t.items.iter().max_by(|a, b| score(a).partial_cmp(&score(b)).unwrap()).unwrap();
            let mut alts: Vec<(String, usize)> = Vec::new();
            for r in &t.items {
                if r.text != text {
                    match alts.iter_mut().find(|a| a.0 == r.text) {
                        Some(a) => a.1 += 1,
                        None => alts.push((r.text.clone(), 1)),
                    }
                }
            }
            alts.sort_by(|a, b| b.1.cmp(&a.1));
            Hit {
                kind: t.kind,
                text,
                conf: conf.min(1.0),
                rect: best.rect,
                trail: t.items.iter().map(|r| (r.frame, r.rect)).collect(),
                first: t.items.first().unwrap().frame,
                last: t.last().frame,
                best: best.frame,
                readings: t.items.len(),
                format,
                alts: alts.into_iter().take(4).map(|a| a.0).collect(),
            }
        })
        .collect();
    hits.sort_by(|a, b| a.first.cmp(&b.first).then(a.rect.y0.partial_cmp(&b.rect.y0).unwrap()).then(a.rect.x0.partial_cmp(&b.rect.x0).unwrap()));
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rd(frame: u64, x: f32, text: &str, conf: f32, kind: Kind) -> Reading {
        Reading { frame, rect: Rect { x0: x, y0: 100.0, x1: x + 120.0, y1: 140.0 }, text: text.into(), conf, kind, slots: vec![], quality: 1.0 }
    }

    /// Slot probabilities for `text`, `p` on the written character and the rest spread evenly.
    fn slots_for(text: &str, p: f32) -> Vec<f32> {
        let mut v = vec![(1.0 - p) / (VOCAB as f32 - 1.0); SLOTS * VOCAB];
        for i in 0..SLOTS {
            let c = text.as_bytes().get(i).copied().unwrap_or(b'_');
            let idx = ALPHABET.iter().position(|&a| a == c).unwrap();
            v[i * VOCAB + idx] = p;
        }
        v
    }

    #[test]
    fn plate_formats() {
        assert_eq!(plate_format("AB123CD"), Some("IT/FR AA 123 AA"));
        assert_eq!(plate_format("AB12CDE"), Some("UK AA 12 AAA"));
        assert_eq!(plate_format("MI203427"), Some("IT storica"));
        assert_eq!(plate_format("1234BCD"), Some("ES 1234 ABC"));
        assert_eq!(plate_format("BX5072"), Some("PT"));
        assert_eq!(plate_format("LBFP4400"), Some("DE"));
        assert_eq!(plate_format("MAB1234E"), Some("DE"));
        assert_eq!(plate_format("161KE137"), Some("IE"));
        assert!(plate_format("BE672562").is_some()); // same shape as the historic Italian format
        assert_eq!(plate_format("HELLO"), None);
    }

    #[test]
    fn plate_correction() {
        assert_eq!(correct_plate("A8123CD").unwrap().0, "AB123CD");
        assert_eq!(correct_plate("AB1Z3CD").unwrap().0, "AB123CD");
        assert!(correct_plate("XYZXYZXYZ").is_none());
    }

    #[test]
    fn format_settles_lookalikes() {
        // the model is unsure whether slot 3 is a 1 or an I; the format (digits in the middle) decides
        let mut p = slots_for("AB123CD", 0.9);
        let (one, i) = (1, 10 + 8);
        p[3 * VOCAB + one] = 0.4;
        p[3 * VOCAB + i] = 0.5;
        let (text, _, fmt) = decode_plate(&p);
        assert_eq!(text, "AB123CD", "{text}");
        assert!(fmt.is_some());
    }

    #[test]
    fn probabilities_accumulate_over_frames() {
        // each frame alone misreads one character, but together they agree
        let mk = |frame, text: &str| {
            let mut r = rd(frame, 100.0 + frame as f32, text, 0.6, Kind::Plate);
            r.slots = slots_for(text, 0.6);
            r
        };
        let hits = aggregate(vec![mk(1, "AB123CD"), mk(2, "AB128CD"), mk(3, "AB123CD"), mk(4, "A8123CD"), mk(5, "AB123CD")], 3);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "AB123CD");
    }

    #[test]
    fn voting_across_frames() {
        let rs = vec![
            rd(10, 100.0, "AB123CD", 0.9, Kind::Plate),
            rd(11, 110.0, "A8123CD", 0.5, Kind::Plate),
            rd(12, 121.0, "AB123CD", 0.8, Kind::Plate),
            rd(13, 133.0, "AB123C0", 0.4, Kind::Plate),
            rd(10, 700.0, "ZZ999ZZ", 0.9, Kind::Plate),
            rd(10, 400.0, "Via Roma", 0.9, Kind::Text),
            rd(11, 400.0, "Via Rorna", 0.6, Kind::Text),
        ];
        let hits = aggregate(rs, 5);
        assert_eq!(hits.len(), 3, "{hits:?}");
        let p = hits.iter().find(|h| h.kind == Kind::Plate && h.text.starts_with("AB")).unwrap();
        assert_eq!(p.text, "AB123CD");
        assert_eq!((p.first, p.last, p.readings), (10, 13, 4));
        assert_eq!(hits.iter().find(|h| h.kind == Kind::Text).unwrap().text, "Via Roma");
    }

    #[test]
    fn fast_mover_stays_one_track() {
        // the plate jumps 2.5 widths per frame in a straight line: the prediction keeps it in one track
        let rs: Vec<Reading> = (0..6).map(|i| rd(i, 100.0 + 300.0 * i as f32, "AB123CD", 0.8, Kind::Plate)).collect();
        assert_eq!(aggregate(rs, 3).len(), 1);
    }

    #[test]
    fn gap_splits_tracks() {
        let hits = aggregate(vec![rd(1, 100.0, "AB123CD", 0.9, Kind::Plate), rd(50, 100.0, "AB123CD", 0.9, Kind::Plate)], 5);
        assert_eq!(hits.len(), 2);
    }
}
