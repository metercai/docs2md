//! P0: parse the lm.npz tables.
//!
//! An `.npz` is a zip container whose entries are `.npy` arrays (v1.0 header,
//! little-endian). The Indra LM stores, per history length n in 0..order:
//! `k{n}` (u64 sorted keys), `v{n}` (f32 log-probs), `hk{n}` (u64 sorted
//! history keys), `hb{n}` (f32 backoff weights), plus scalars `order`,
//! `vocab_size` and optional `bits`.

use std::collections::HashMap;

/// One parsed npy array (the element types this file family uses).
#[derive(Debug, Clone)]
pub enum NpyArray {
    ScalarI64(i64),
    Keys(Vec<u64>),
    F32(Vec<f32>),
}

impl NpyArray {
    pub fn len(&self) -> usize {
        match self {
            NpyArray::ScalarI64(_) => 0,
            NpyArray::Keys(k) => k.len(),
            NpyArray::F32(f) => f.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Parse a zip container into name -> payload (deflate or stored).
///
/// Walks the central directory (robust against the data-descriptor flag,
/// which is set by numpy's writer and leaves 0xFFFFFFFF in local headers).
pub fn parse_zip(bytes: &[u8]) -> Result<HashMap<String, Vec<u8>>, String> {
    let n = bytes.len();
    if n < 22 {
        return Err("zip: too short".into());
    }
    // Find the End Of Central Directory record (scan back: comment <= 65535).
    let scan_from = n.saturating_sub(22 + 65535);
    let mut eocd = None;
    for i in (scan_from..=n - 22).rev() {
        if bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06] {
            eocd = Some(i);
            break;
        }
    }
    let eocd = eocd.ok_or("zip: end-of-central-directory not found")?;
    let count = u16::from_le_bytes(bytes[eocd + 10..eocd + 12].try_into().unwrap()) as usize;
    let cd_off = u32::from_le_bytes(bytes[eocd + 16..eocd + 20].try_into().unwrap()) as usize;
    let mut out = HashMap::new();
    let mut off = cd_off;
    for _ in 0..count {
        if off + 46 > n || bytes[off..off + 4] != [0x50, 0x4b, 0x01, 0x02] {
            return Err("zip: bad central directory header".into());
        }
        let method = u16::from_le_bytes(bytes[off + 10..off + 12].try_into().unwrap());
        let csize = u32::from_le_bytes(bytes[off + 20..off + 24].try_into().unwrap()) as usize;
        let usize = u32::from_le_bytes(bytes[off + 24..off + 28].try_into().unwrap()) as usize;
        let nname = u16::from_le_bytes(bytes[off + 28..off + 30].try_into().unwrap()) as usize;
        let nextra = u16::from_le_bytes(bytes[off + 30..off + 32].try_into().unwrap()) as usize;
        let ncomment = u16::from_le_bytes(bytes[off + 32..off + 34].try_into().unwrap()) as usize;
        let local_off = u32::from_le_bytes(bytes[off + 42..off + 46].try_into().unwrap()) as usize;
        let name = String::from_utf8_lossy(&bytes[off + 46..off + 46 + nname]).into_owned();
        // Local header for the data start (nname/nextra may differ from CD).
        if local_off + 30 > n || bytes[local_off..local_off + 4] != [0x50, 0x4b, 0x03, 0x04] {
            return Err(format!("zip: bad local header for {name}"));
        }
        let lnname =
            u16::from_le_bytes(bytes[local_off + 26..local_off + 28].try_into().unwrap()) as usize;
        let lnextra =
            u16::from_le_bytes(bytes[local_off + 28..local_off + 30].try_into().unwrap()) as usize;
        let data_start = local_off + 30 + lnname + lnextra;
        let payload = &bytes[data_start..data_start + csize];
        let data = match method {
            0 => payload.to_vec(), // stored
            8 => {
                // zip method=8 is RAW deflate (no zlib header). decompress_vec
                // appends into the vec's spare capacity, so reserve the exact
                // uncompressed size from the central directory.
                let mut d = flate2::Decompress::new_with_window_bits(false, 15);
                let mut buf = Vec::with_capacity(usize.max(1));
                d.decompress_vec(payload, &mut buf, flate2::FlushDecompress::Finish)
                    .map_err(|e| format!("inflate {name}: {e}"))?;
                if buf.len() != usize {
                    return Err(format!(
                        "inflate {name}: got {} bytes, expected {usize}",
                        buf.len()
                    ));
                }
                buf
            }
            m => return Err(format!("unsupported zip method {m} for {name}")),
        };
        out.insert(name, data);
        off += 46 + nname + nextra + ncomment;
    }
    Ok(out)
}

/// Parse one npy payload (v1.0 header) into the family's element types.
pub fn parse_npy(data: &[u8]) -> Result<NpyArray, String> {
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err("not a v1.0 npy array".into());
    }
    let ver = &data[6..8];
    if ver != [1, 0] {
        return Err(format!("unsupported npy version {ver:?}"));
    }
    let hl = u16::from_le_bytes(data[8..10].try_into().unwrap()) as usize;
    let hdr_end = 10 + hl;
    if hdr_end > data.len() {
        return Err("truncated npy header".into());
    }
    let hdr =
        std::str::from_utf8(&data[10..hdr_end]).map_err(|e| format!("bad npy header: {e}"))?;
    // Header is a Python dict repr: {'descr': '<f4', 'fortran_order': False, 'shape': (552,), }
    let (descr, shape) = (get_str(hdr, "descr"), get_str(hdr, "shape"));
    let descr = descr.unwrap_or_default();
    let shape = shape.unwrap_or_default();
    let payload = &data[hdr_end..];
    let scalar = shape.trim().is_empty() || shape.trim() == "()";
    match (descr.as_str(), scalar) {
        ("<i8", true) => {
            if payload.len() != 8 {
                return Err(format!("i8 scalar: {len} bytes", len = payload.len()));
            }
            Ok(NpyArray::ScalarI64(i64::from_le_bytes(
                payload.try_into().unwrap(),
            )))
        }
        ("<u8", _) => {
            let mut v = Vec::new();
            let (chunks, rem) = payload.as_chunks::<8>();
            if !rem.is_empty() {
                return Err("u8 array: length not a multiple of 8".into());
            }
            v.extend(chunks.iter().map(|c| u64::from_le_bytes(*c)));
            Ok(NpyArray::Keys(v))
        }
        ("<f4", _) => {
            let mut v = Vec::new();
            let (chunks, rem) = payload.as_chunks::<4>();
            if !rem.is_empty() {
                return Err("f32 array: length not a multiple of 4".into());
            }
            v.extend(chunks.iter().map(|c| f32::from_le_bytes(*c)));
            Ok(NpyArray::F32(v))
        }
        (d, s) => Err(format!("unsupported npy descr {d:?} shape {s:?}")),
    }
}

/// Extract a quoted value from an npy dict header, or the parenthesized shape.
fn get_str(hdr: &str, key: &str) -> Option<String> {
    let k = format!("'{key}':");
    let i = hdr.find(&k)?;
    let rest = &hdr[i + k.len()..];
    let rest = rest.trim_start();
    if let Some(q) = rest.strip_prefix('"') {
        let end = q.find('"')?;
        return Some(q[..end].to_string());
    }
    if rest.starts_with('(') {
        let end = rest.find(')')?;
        return Some(rest[..=end].to_string());
    }
    if let Some(q) = rest.strip_prefix('\'') {
        let end = q.find('\'')?;
        return Some(q[..end].to_string());
    }
    None
}

/// Parse a whole lm.npz into its arrays.
pub fn parse_npz(bytes: &[u8]) -> Result<HashMap<String, NpyArray>, String> {
    let entries = parse_zip(bytes)?;
    let mut out = HashMap::new();
    for (name, data) in entries {
        if name.ends_with(".npy") {
            let key = name.trim_end_matches(".npy").to_string();
            match parse_npy(&data) {
                Ok(a) => {
                    out.insert(key, a);
                }
                Err(e) => {
                    return Err(format!(
                        "parse {name}: {e} (len={}, first={:02x?})",
                        data.len(),
                        &data[..data.len().min(8)]
                    ))
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(descr: &str, shape: &str, payload: Vec<u8>) -> Vec<u8> {
        let hdr = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        let hl = hdr.len();
        let mut v = Vec::new();
        v.extend_from_slice(b"\x93NUMPY");
        v.extend_from_slice(&[1, 0]);
        v.extend_from_slice(&(hl as u16).to_le_bytes());
        v.extend_from_slice(hdr.as_bytes());
        v.extend_from_slice(&payload);
        v
    }

    #[test]
    fn npy_roundtrip() {
        let p = npy("<i8", "()", 5i64.to_le_bytes().to_vec());
        assert!(matches!(parse_npy(&p).unwrap(), NpyArray::ScalarI64(5)));
        let p = npy(
            "<u8",
            "(2,)",
            [1u8, 0, 0, 0, 0, 0, 0, 0, 0x02, 0, 0, 0, 0, 0, 0, 0].to_vec(),
        );
        match parse_npy(&p).unwrap() {
            NpyArray::Keys(k) => assert_eq!(k, vec![1, 2]),
            _ => panic!(),
        }
        let f = f32::from_bits(0x3f80_0000); // 1.0
        let p = npy("<f4", "(1,)", f.to_le_bytes().to_vec());
        match parse_npy(&p).unwrap() {
            NpyArray::F32(f) => assert_eq!(f, vec![1.0]),
            _ => panic!(),
        }
    }

    #[test]
    fn zip_roundtrip() {
        // Build a stored-entry zip by hand: local header + central directory
        // + EOCD (the parser walks the central directory).
        let name = b"a.npy";
        let data = vec![1u8, 2, 3, 4];
        let mut z = Vec::new();
        // local header (flags 0 -> sizes in the header)
        z.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        z.extend_from_slice(&20u16.to_le_bytes()); // version
        z.extend_from_slice(&0u16.to_le_bytes()); // flags
        z.extend_from_slice(&0u16.to_le_bytes()); // stored
        z.extend_from_slice(&0u16.to_le_bytes()); // time
        z.extend_from_slice(&0u16.to_le_bytes()); // date
        z.extend_from_slice(&0u32.to_le_bytes()); // crc
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes()); // extra
        z.extend_from_slice(name);
        z.extend_from_slice(&data);
        // central directory header
        let cd_off = z.len();
        z.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        z.extend_from_slice(&20u16.to_le_bytes()); // version made
        z.extend_from_slice(&20u16.to_le_bytes()); // version needed
        z.extend_from_slice(&0u16.to_le_bytes()); // flags
        z.extend_from_slice(&0u16.to_le_bytes()); // stored
        z.extend_from_slice(&0u16.to_le_bytes()); // time
        z.extend_from_slice(&0u16.to_le_bytes()); // date
        z.extend_from_slice(&0u32.to_le_bytes()); // crc
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes()); // extra
        z.extend_from_slice(&0u16.to_le_bytes()); // comment
        z.extend_from_slice(&0u16.to_le_bytes()); // disk
        z.extend_from_slice(&0u16.to_le_bytes()); // attrs
        z.extend_from_slice(&0u32.to_le_bytes()); // perms
        z.extend_from_slice(&0u32.to_le_bytes()); // local offset
        z.extend_from_slice(name);
        // EOCD
        z.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes()); // disk
        z.extend_from_slice(&0u16.to_le_bytes()); // cd disk
        z.extend_from_slice(&1u16.to_le_bytes()); // entries here
        z.extend_from_slice(&1u16.to_le_bytes()); // entries total
        z.extend_from_slice(&((z.len() - cd_off) as u32).to_le_bytes());
        z.extend_from_slice(&(cd_off as u32).to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes()); // comment
        let m = parse_zip(&z).unwrap();
        assert_eq!(m.get("a.npy").unwrap(), &data[..]);
    }

    #[test]
    fn zip_data_descriptor_flag() {
        // numpy's writer sets the data-descriptor bit: local header sizes are
        // 0xFFFFFFFF and the real sizes live only in the central directory.
        let name = b"x.npy";
        let data = vec![7u8, 8, 9];
        let mut z = Vec::new();
        // local header with the data-descriptor flag set
        z.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        z.extend_from_slice(&20u16.to_le_bytes());
        z.extend_from_slice(&0x08u16.to_le_bytes()); // data descriptor flag
        z.extend_from_slice(&0u16.to_le_bytes()); // stored
        z.extend_from_slice(&0u16.to_le_bytes()); // time
        z.extend_from_slice(&0u16.to_le_bytes()); // date
        z.extend_from_slice(&0u32.to_le_bytes()); // crc (descriptor)
        z.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // csize unknown
        z.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // usize unknown
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes()); // extra
        z.extend_from_slice(name);
        z.extend_from_slice(&data);
        // central directory header with the real sizes
        let cd_off = z.len();
        z.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        z.extend_from_slice(&20u16.to_le_bytes());
        z.extend_from_slice(&20u16.to_le_bytes());
        z.extend_from_slice(&0x08u16.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes()); // stored
        z.extend_from_slice(&0u16.to_le_bytes()); // time
        z.extend_from_slice(&0u16.to_le_bytes()); // date
        z.extend_from_slice(&0u32.to_le_bytes()); // crc
        z.extend_from_slice(&(data.len() as u32).to_le_bytes()); // real csize
        z.extend_from_slice(&(data.len() as u32).to_le_bytes()); // real usize
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes()); // extra
        z.extend_from_slice(&0u16.to_le_bytes()); // comment
        z.extend_from_slice(&0u16.to_le_bytes()); // disk
        z.extend_from_slice(&0u16.to_le_bytes()); // attrs
        z.extend_from_slice(&0u32.to_le_bytes()); // perms
        z.extend_from_slice(&0u32.to_le_bytes()); // local offset
        z.extend_from_slice(name);
        // EOCD
        z.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.extend_from_slice(&1u16.to_le_bytes());
        z.extend_from_slice(&1u16.to_le_bytes());
        z.extend_from_slice(&((z.len() - cd_off) as u32).to_le_bytes());
        z.extend_from_slice(&(cd_off as u32).to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        let m = parse_zip(&z).unwrap();
        assert_eq!(m.get("x.npy").unwrap(), &data[..]);
    }
}
