//! Post-processing of OCR readings: licence-plate format validation / correction, and grouping the readings of a
//! series of frames into tracks so that each thing seen is reported once, with the text decided by a weighted vote.
use crate::ocr::Rect;

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
}

/// A thing seen on one or more frames, with the voted text.
#[derive(Clone, Debug)]
pub struct Hit {
    pub kind: Kind,
    pub text: String,
    /// 0..1: mean confidence of the readings that agree with the voted text.
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

/// (name, masks) — masks use `L` for a letter and `D` for a digit.
const FORMATS: &[(&str, &[&str])] = &[
    ("IT/FR AA 123 AA", &["LLDDDLL"]),
    ("IT storica", &["LLDDDDDD", "LLDDDDD"]),
    ("FR AAA 123 AA", &["DDDLLLDD", "DDDLLDD"]),
    ("ES 1234 ABC", &["DDDDLLL"]),
    ("PT", &["DDLLDD", "LLDDLL", "DDDDLL", "LLDDDD"]),
    (
        "NL",
        &["LLDDDD", "DDDDLL", "DDLLDD", "LLDDLL", "DDLLLD", "DLLLDD", "LLDDDL", "LDDDLL", "LLLDDL", "LDDLLL", "DLLDDD"],
    ),
];

fn mask(s: &str) -> Option<String> {
    s.chars().map(|c| if c.is_ascii_digit() { Some('D') } else if c.is_ascii_alphabetic() { Some('L') } else { None }).collect()
}

/// Name of the national format that `text` (upper-case letters and digits only) matches, if any.
pub fn plate_format(text: &str) -> Option<&'static str> {
    let m = mask(text)?;
    for (name, masks) in FORMATS {
        if masks.contains(&m.as_str()) {
            return Some(name);
        }
    }
    // German: 2–5 letters (district + letters) then 1–4 digits, optionally an E/H suffix
    let core = if m.len() >= 4 && m.ends_with("DL") && matches!(text.chars().last(), Some('E' | 'H')) { &m[..m.len() - 1] } else { &m[..] };
    let letters = core.chars().take_while(|&c| c == 'L').count();
    let digits = core.len() - letters;
    if (2..=5).contains(&letters) && (1..=4).contains(&digits) && core[letters..].chars().all(|c| c == 'D') {
        return Some("DE");
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
    for (name, masks) in FORMATS {
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

struct Track {
    kind: Kind,
    items: Vec<Reading>,
}

impl Track {
    fn last(&self) -> &Reading {
        self.items.last().unwrap()
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
            let (c1, c2) = (l.rect.center(), r.rect.center());
            let dist = ((c1.0 - c2.0).powi(2) + (c1.1 - c2.1).powi(2)).sqrt();
            let near = dist < 2.0 * l.rect.w().max(r.rect.w());
            let score = if iou > 0.3 {
                iou + sim
            } else if sim >= 0.6 && near {
                sim
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
            let (mut text, mut conf, _) = vote(&refs, plate);
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
            let best = t.items.iter().max_by(|a, b| (similarity(&a.text, &text) * a.conf).partial_cmp(&(similarity(&b.text, &text) * b.conf)).unwrap()).unwrap();
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
            if plate && format.is_some() {
                conf = conf.min(1.0);
            }
            Hit {
                kind: t.kind,
                text,
                conf,
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
        Reading { frame, rect: Rect { x0: x, y0: 100.0, x1: x + 120.0, y1: 140.0 }, text: text.into(), conf, kind }
    }

    #[test]
    fn plate_formats() {
        assert_eq!(plate_format("AB123CD"), Some("IT/FR AA 123 AA"));
        assert_eq!(plate_format("MI203427"), Some("IT storica"));
        assert_eq!(plate_format("1234BCD"), Some("ES 1234 ABC"));
        assert_eq!(plate_format("BX5072"), Some("PT"));
        assert_eq!(plate_format("LBFP4400"), Some("DE"));
        assert_eq!(plate_format("MAB1234E"), Some("DE"));
        assert_eq!(plate_format("HELLO"), None);
    }

    #[test]
    fn plate_correction() {
        assert_eq!(correct_plate("A8123CD").unwrap().0, "AB123CD");
        assert_eq!(correct_plate("AB1Z3CD").unwrap().0, "AB123CD");
        assert_eq!(correct_plate("AB12OCD").unwrap().0, "AB120CD");
        assert!(correct_plate("XYZXYZXYZ").is_none());
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
    fn gap_splits_tracks() {
        let hits = aggregate(vec![rd(1, 100.0, "AB123CD", 0.9, Kind::Plate), rd(50, 100.0, "AB123CD", 0.9, Kind::Plate)], 5);
        assert_eq!(hits.len(), 2);
    }
}
