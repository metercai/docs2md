//! Markdown assembly for OCR (L1) pages and the provenance/quality notes.
//!
//! PP-OCRv6 gives no font or heading semantics, so the honest L1 output is
//! plain reading-order text (no claimed markdown structure). Reading order
//! is recovered by a *transitive* comparator: bucket each line's top into a
//! fixed-height band, then sort by x within a band. A plain "close-in-y
//! then x" comparator is not a total order and panics in `sort_by` when the
//! y-gaps chain transitively.

use crate::l1::OcrLine;

/// Convert OCR lines to a markdown string in reading order (top→bottom,
/// left→right), one line per OCR row.
pub fn ocr_lines_to_markdown(lines: &[OcrLine]) -> String {
    // (top, left, text)
    let mut rows: Vec<(f32, f32, &str)> = lines
        .iter()
        .map(|l| (l.top, l.left, l.text.as_str()))
        .collect();
    // 10 px reading-order band (calibrated on the fixtures).
    let band = |y: f32| (y / 10.0).floor() as i64;
    rows.sort_by(|a, b| {
        band(a.0)
            .cmp(&band(b.0))
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut s = String::new();
    for (_, _, t) in rows {
        s.push_str(t);
        s.push('\n');
    }
    s
}

/// The HTML comment appended to a page that needed a higher tier (L2) but
/// the L2 service was unavailable, so the L1 output is a best-effort stand-in.
pub const NEEDS_HIGHER_NOTE: &str =
    "<!-- needs a higher-tier OCR model: L2 service unavailable -->\n";

/// The explicit low-confidence marker emitted as a page's markdown when
/// the whole cascade (L1 serial chain, L2 PP-OCRv6, L3 VLM) produced no
/// usable output: an explicit, non-empty note so the page is never
/// silently empty (plan G, end of the cascade).
pub const LOW_CONF_MARKER: &str = "<!-- low confidence: the OCR chain escalated this page to L2 \
         (PP-OCRv6) and no usable output was produced -->\n";

/// The note attached to a chain-escalated page whose L2 (PP-OCRv6) read
/// nothing, so the (possibly partial) L1 output stands as the page's
/// best-effort markdown.
pub const KEEP_L1_NOTE: &str =
    "<!-- needs a higher-tier OCR model: the serial chain escalated this \
         page to L2 (PP-OCRv6) but it read nothing — keeping the L1 \
         best-effort output -->\n";

/// The note attached when the L2 (PP-OCRv6) result was weak or invalid
/// (plan §2.2) and the L3 VLM leg was unavailable or failed: the local
/// best-effort PP-OCRv6 text stands as the page's markdown.
pub const L3_FALLBACK_NOTE: &str =
    "<!-- weak PP-OCRv6 output: the L3 VLM was unavailable or failed — \
     keeping the local best-effort text -->\n";

#[cfg(test)]
mod tests {
    use super::*;

    fn line(top: f32, left: f32, text: &str) -> OcrLine {
        OcrLine {
            text: text.into(),
            conf: 1.0,
            top,
            left,
            width: 10.0,
            height: 10.0,
        }
    }

    #[test]
    fn sorts_top_then_left() {
        // Out of order: bottom-left, top-right, top-left.
        let lines = vec![
            line(120.0, 10.0, "c"),
            line(10.0, 200.0, "b"),
            line(10.0, 10.0, "a"),
        ];
        let md = ocr_lines_to_markdown(&lines);
        assert_eq!(md, "a\nb\nc\n");
    }

    #[test]
    fn transitive_bucket_does_not_panic_on_chained_y_gaps() {
        // y-gaps that would chain under a non-transitive "close then x" rule:
        // tops 0/6/12/18 sit in bands {0,0,1,1}. Within band 0 the line with
        // the smaller left (L1, left=0) must precede L0 (left=50); within
        // band 1, L2 (left=90) vs L3 (left=0) => L3 first. The comparator
        // must be a total order (it is: band then left) so sort_by cannot
        // panic on the transitive chain.
        let lines = vec![
            line(0.0, 50.0, "L0"),
            line(6.0, 0.0, "L1"),
            line(12.0, 90.0, "L2"),
            line(18.0, 0.0, "L3"),
        ];
        let md = ocr_lines_to_markdown(&lines);
        assert_eq!(md, "L1\nL0\nL3\nL2\n");
    }

    #[test]
    fn empty_input_empty_output() {
        assert_eq!(ocr_lines_to_markdown(&[]), "");
    }
}
