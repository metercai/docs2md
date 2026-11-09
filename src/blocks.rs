//! Structured layout blocks: the single source of truth for the block label
//! vocabulary, the Chandra→OPB class mapping, and bbox normalization.
//!
//! A page's final `blocks` are emitted in `PageRecord` (and carried on
//! `OcrPage` for the L1 engines). They are a NEW output surface: they do not
//! affect markdown bytes or tier routing. The label set and the class mapping
//! are constants here so no call site rewrites them (the layout judge reads
//! exactly these labels).

use crate::l1::Block;

/// The 11 block labels the layout judge accepts, in the same names/order as
/// `omni_parse_bench.views.LABELS`. Code-layer fixed — do not scatter strings.
pub const TEXT: &str = "text";
pub const HEADING: &str = "heading";
pub const PAGE_FURNITURE: &str = "page_furniture";
pub const LIST: &str = "list";
pub const CAPTION: &str = "caption";
pub const FOOTNOTE: &str = "footnote";
pub const EQUATION: &str = "equation";
pub const CODE: &str = "code";
pub const TABLE: &str = "table";
pub const FORM: &str = "form";
pub const FIGURE: &str = "figure";

/// All 11 labels (order = views.LABELS). Exposed for completeness / tests.
pub const LABELS: [&str; 11] = [
    TEXT,
    HEADING,
    PAGE_FURNITURE,
    LIST,
    CAPTION,
    FOOTNOTE,
    EQUATION,
    CODE,
    TABLE,
    FORM,
    FIGURE,
];

/// Chandra block-class index → OPB label. The index is the position in
/// `l1::indra::order::BLOCK_CLASSES` (the order model's cls head order):
/// 0 Text, 1 Section-Header, 2 Page-Header, 3 Page-Footer, 4 List-Group,
/// 5 Table, 6 Caption, 7 Footnote, 8 Table-Of-Contents, 9 Form.
///
/// E1 emits every L1/L0 block as `TEXT`; E2 maps structural classes through
/// this table (Section-Header → heading, Table → table, …). Table-Of-Contents
/// maps to `text` (no OPB class for a TOC; conservative).
pub const CHANDRA_TO_OPB: [&str; 10] = [
    TEXT,
    HEADING,
    PAGE_FURNITURE,
    PAGE_FURNITURE,
    LIST,
    TABLE,
    CAPTION,
    FOOTNOTE,
    TEXT,
    FORM,
];

/// Map a Chandra class index to its OPB label. Out-of-range indexes fall back
/// to `TEXT` (never panic on a model's odd output).
pub fn chandra_to_opb(class: usize) -> &'static str {
    CHANDRA_TO_OPB.get(class).copied().unwrap_or(TEXT)
}

/// Normalize a pixel bbox `[x0, y0, x1, y1]` (y-down, top-left origin) to
/// `[0, 1]` relative to the rendered page `(page_w, page_h)` in the same
/// pixel space, clamped to `[0, 1]`. This is the coordinate space the layout
/// judge consumes.
pub fn normalize_bbox(x0: f64, y0: f64, x1: f64, y1: f64, page_w: f64, page_h: f64) -> [f64; 4] {
    if page_w <= 0.0 || page_h <= 0.0 {
        return [0.0; 4];
    }
    [
        (x0 / page_w).clamp(0.0, 1.0),
        (y0 / page_h).clamp(0.0, 1.0),
        (x1 / page_w).clamp(0.0, 1.0),
        (y1 / page_h).clamp(0.0, 1.0),
    ]
}

/// Map a PDF user-space box (y-UP, points) to the rendered page's
/// normalized `[0,1]` y-DOWN box, for the page's `/Rotate`. The render box
/// (CropBox ∩ MediaBox, the box `render::render_pdf_bytes` renders) is
/// `(box_x0, box_y0, box_w, box_h)` in the same user space. Mirrors
/// pdf_oxide's `page_base_transform` (rendering/mod.rs:188) inverted — the
/// origin translation uses the render box, and the rotation cases are its
/// four branches solved for the user-space point.
pub fn pdf_box_to_norm(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    box_x0: f64,
    box_y0: f64,
    box_w: f64,
    box_h: f64,
    rotation: i32,
) -> [f64; 4] {
    if box_w <= 0.0 || box_h <= 0.0 {
        return [0.0; 4];
    }
    let n = |v: f64| v.clamp(0.0, 1.0);
    // Normalized user-space (y-up) of the box's two edges, relative to the
    // render box: ux 0=left 1=right, uy 0=bottom 1=top.
    let ux0 = (x0 - box_x0) / box_w;
    let ux1 = (x1 - box_x0) / box_w;
    let uy0 = (y0 - box_y0) / box_h;
    let uy1 = (y1 - box_y0) / box_h;
    // Display frame is y-down (top=0, x-right). Each branch maps the box's
    // min/max user edges to the display min/max — the mirrored axes (180°: x,
    // 270°: both) swap which edge lands left/top so the result is never a
    // degenerate box.
    match rotation.rem_euclid(360) {
        // 0°: x as-is, y flipped (user y-up → display y-down).
        0 => [n(ux0), n(1.0 - uy1), n(ux1), n(1.0 - uy0)],
        // 90° CW: user top→display right, user left→display top.
        90 => [n(uy0), n(ux0), n(uy1), n(ux1)],
        // 180°: x mirrored (left→right), y as-is (bottom→top of display).
        180 => [n(1.0 - ux1), n(uy0), n(1.0 - ux0), n(uy1)],
        // 270° CW: x mirrored by uy, y mirrored by ux.
        _ => [n(1.0 - uy1), n(1.0 - ux1), n(1.0 - uy0), n(1.0 - ux0)],
    }
}

/// Cluster PDF text spans into reading-order line blocks (the L0 source of
/// structured blocks). `spans` are `(x0, y0, x1, y1, text)` in PDF user
/// space (y-UP, points); `box_*` is the page render box in the same space;
/// `rotation` the page's `/Rotate`. Spans sharing a vertical line (overlap
/// ≥ 50 % of the smaller height) merge into one block regardless of column —
/// a two-column row stays one block; blocks come out top to bottom. The
/// block label is always `TEXT` (E1); E2 re-labels structural rows from the
/// layout regions (the optional `--blocks` relabel, decision D).
pub fn spans_to_text_blocks(
    spans: &[(f64, f64, f64, f64, String)],
    box_x0: f64,
    box_y0: f64,
    box_w: f64,
    box_h: f64,
    rotation: i32,
) -> Vec<Block> {
    if box_w <= 0.0 || box_h <= 0.0 {
        return Vec::new();
    }
    // Degenerate spans carry no geometry to cluster.
    let mut valid: Vec<usize> = (0..spans.len())
        .filter(|&i| spans[i].2 > spans[i].0 && spans[i].3 > spans[i].1)
        .collect();
    if valid.is_empty() {
        return Vec::new();
    }
    // Top first in the y-up space (larger y = higher on the page).
    valid.sort_by(|&a, &b| {
        let ay = (spans[a].1 + spans[a].3) / 2.0;
        let by = (spans[b].1 + spans[b].3) / 2.0;
        by.total_cmp(&ay).then_with(|| {
            spans[a]
                .0
                .partial_cmp(&spans[b].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    // Greedy rows: a span joins the first row whose vertical overlap with it
    // is ≥ 50 % of the smaller height (same printed line, any column).
    #[derive(Default)]
    struct Row {
        y0: f64,
        y1: f64,
        x0: f64,
        x1: f64,
        idx: Vec<usize>,
    }
    let mut rows: Vec<Row> = Vec::new();
    for &i in &valid {
        let (sx0, sy0, sx1, sy1) = (spans[i].0, spans[i].1, spans[i].2, spans[i].3);
        let sh = sy1 - sy0;
        let mut hit = None;
        for (ri, r) in rows.iter_mut().enumerate() {
            let overlap = r.y1.min(sy1) - r.y0.max(sy0);
            let rh = r.y1 - r.y0;
            if overlap >= 0.5 * sh.min(rh) {
                r.y0 = r.y0.min(sy0);
                r.y1 = r.y1.max(sy1);
                r.x0 = r.x0.min(sx0);
                r.x1 = r.x1.max(sx1);
                r.idx.push(i);
                hit = Some(ri);
                break;
            }
        }
        if hit.is_none() {
            rows.push(Row {
                y0: sy0,
                y1: sy1,
                x0: sx0,
                x1: sx1,
                idx: vec![i],
            });
        }
    }
    rows.into_iter().map(|r| {
        let mut order: Vec<usize> = r.idx;
        order.sort_by(|&a, &b| {
            spans[a]
                .0
                .partial_cmp(&spans[b].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let text = order
            .iter()
            .map(|&i| spans[i].4.as_str())
            .filter(|t| !t.trim().is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        Block {
            label: TEXT,
            bbox: pdf_box_to_norm(
                r.x0, r.y0, r.x1, r.y1, box_x0, box_y0, box_w, box_h, rotation,
            ),
            text,
        }
    })
    .collect()
}

impl Block {
    /// Build a normalized block in the judge's `[0,1]` y-down space.
    pub fn new_normalized(
        label: &'static str,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        page_w: f64,
        page_h: f64,
        text: String,
    ) -> Self {
        let bbox = normalize_bbox(x0, y0, x1, y1, page_w, page_h);
        Block { label, bbox, text }
    }
}

/// E2 (L0): re-label L0 line blocks from the layout regions (decision D's
/// optional `--blocks` relabel). `regions`
/// are `(x0, y0, x1, y1, label)` in the same normalized `[0,1]` y-down space
/// as the block bboxes. A block takes the label of the region its CENTER
/// point falls in (a body line hugging a region edge keeps `text`); when the
/// center lands in several regions the smallest (most specific) wins; with no
/// regions a block keeps its current label (E1: all `text`).
pub fn relabel_l0_blocks(
    blocks: Vec<Block>,
    regions: &[(f64, f64, f64, f64, &'static str)],
) -> Vec<Block> {
    if regions.is_empty() {
        return blocks;
    }
    blocks
        .into_iter()
        .map(|b| {
            let cx = (b.bbox[0] + b.bbox[2]) / 2.0;
            let cy = (b.bbox[1] + b.bbox[3]) / 2.0;
            let mut best: Option<(f64, &str)> = None;
            for &(rx0, ry0, rx1, ry1, label) in regions {
                if cx >= rx0 && cx <= rx1 && cy >= ry0 && cy <= ry1 {
                    let area = (rx1 - rx0) * (ry1 - ry0);
                    match best {
                        None => best = Some((area, label)),
                        Some((a, _)) if area < a => best = Some((area, label)),
                        _ => {}
                    }
                }
            }
            match best {
                Some((_, label)) => Block {
                    label,
                    bbox: b.bbox,
                    text: b.text,
                },
                None => b,
            }
        })
        .collect()
}

/// D (optional `--blocks` relabel): map the layout detector's regions to
/// the `relabel` input of [`relabel_l0_blocks`]. Only `Title` → `heading`
/// and `Table` → `table` are structural block labels; every other class
/// (text, figure, caption, …) stays `text`. A region is kept only when
/// confident (conf > `RELABEL_MIN_CONF`) and compact (< `RELABEL_MAX_AREA_FRAC`
/// of the page) — the guard that stops a page-wide table/figure blob from
/// relabeling every line. Whole-page table pages never reach this point in
/// the first place (the free `has_table` signal reroutes them to L1, whose
/// order model emits the table structure). Bboxes come out in the rendered
/// page's `[0,1]` y-down space, clamped.
pub const RELABEL_MIN_CONF: f32 = 0.4;
pub const RELABEL_MAX_AREA_FRAC: f64 = 0.5;

pub fn layout_regions_to_relabel(
    regions: &[rusto::layout::LayoutRegion],
    page_w: f64,
    page_h: f64,
) -> Vec<(f64, f64, f64, f64, &'static str)> {
    if page_w <= 0.0 || page_h <= 0.0 {
        return Vec::new();
    }
    let page_area = page_w * page_h;
    let mut out = Vec::new();
    for r in regions {
        let label = match r.layout_type {
            rusto::layout::LayoutType::Title => Some(HEADING),
            rusto::layout::LayoutType::Table => Some(TABLE),
            _ => None,
        };
        let Some(label) = label else { continue };
        if !(r.confidence > RELABEL_MIN_CONF) {
            continue;
        }
        let area_frac = ((r.bbox.x_max - r.bbox.x_min) as f64
            * (r.bbox.y_max - r.bbox.y_min) as f64)
            / page_area;
        if area_frac >= RELABEL_MAX_AREA_FRAC {
            continue;
        }
        out.push((
            ((r.bbox.x_min as f64) / page_w).clamp(0.0, 1.0),
            ((r.bbox.y_min as f64) / page_h).clamp(0.0, 1.0),
            ((r.bbox.x_max as f64) / page_w).clamp(0.0, 1.0),
            ((r.bbox.y_max as f64) / page_h).clamp(0.0, 1.0),
            label,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_match_judge_vocabulary() {
        assert_eq!(LABELS.len(), 11);
        assert_eq!(LABELS[0], "text");
        assert_eq!(LABELS[8], "table");
        assert_eq!(LABELS[10], "figure");
    }

    #[test]
    fn chandra_mapping_full_table() {
        assert_eq!(CHANDRA_TO_OPB.len(), 10);
        // The structural classes E2 depends on.
        assert_eq!(chandra_to_opb(0), TEXT); // Text
        assert_eq!(chandra_to_opb(1), HEADING); // Section-Header
        assert_eq!(chandra_to_opb(2), PAGE_FURNITURE); // Page-Header
        assert_eq!(chandra_to_opb(3), PAGE_FURNITURE); // Page-Footer
        assert_eq!(chandra_to_opb(4), LIST); // List-Group
        assert_eq!(chandra_to_opb(5), TABLE); // Table
        assert_eq!(chandra_to_opb(6), CAPTION); // Caption
        assert_eq!(chandra_to_opb(7), FOOTNOTE); // Footnote
        assert_eq!(chandra_to_opb(8), TEXT); // Table-Of-Contents (conservative)
        assert_eq!(chandra_to_opb(9), FORM); // Form
    }

    #[test]
    fn chandra_mapping_is_bounded() {
        // Out-of-range never panics; falls back to text.
        assert_eq!(chandra_to_opb(99), TEXT);
    }

    #[cfg(feature = "indra")]
    #[test]
    fn chandra_mapping_tracks_block_classes_order() {
        // The mapping index must stay aligned with the cls-head order.
        use crate::l1::indra::order::BLOCK_CLASSES;
        assert_eq!(CHANDRA_TO_OPB.len(), BLOCK_CLASSES.len());
        assert_eq!(BLOCK_CLASSES[1], "Section-Header");
        assert_eq!(BLOCK_CLASSES[5], "Table");
    }

    #[test]
    fn normalize_bbox_identity_and_clamp() {
        // Exact fractions.
        assert_eq!(
            normalize_bbox(0.0, 0.0, 100.0, 50.0, 100.0, 50.0),
            [0.0, 0.0, 1.0, 1.0]
        );
        // Partial.
        assert_eq!(
            normalize_bbox(25.0, 10.0, 75.0, 40.0, 100.0, 50.0),
            [0.25, 0.2, 0.75, 0.8]
        );
        // Clamped out-of-range.
        let b = normalize_bbox(-10.0, -5.0, 120.0, 60.0, 100.0, 50.0);
        assert_eq!(b, [0.0, 0.0, 1.0, 1.0]);
        // Degenerate page → zeros.
        assert_eq!(normalize_bbox(0.0, 0.0, 1.0, 1.0, 0.0, 0.0), [0.0; 4]);
    }

    #[test]
    fn pdf_box_to_norm_flip_and_offset() {
        // Render box at origin 0,0 size 100x200 (points), rotation 0. A span
        // box (10, 50, 30, 60): its top (y-up) is 60, bottom 50. In y-down
        // normalized: y_top = (200-60)/200 = 0.7, y_bot = (200-50)/200 = 0.75.
        let b = pdf_box_to_norm(10.0, 50.0, 30.0, 60.0, 0.0, 0.0, 100.0, 200.0, 0);
        assert!((b[0] - 0.1).abs() < 1e-9);
        assert!((b[2] - 0.3).abs() < 1e-9);
        assert!((b[1] - 0.7).abs() < 1e-9, "y_top got {}", b[1]);
        assert!((b[3] - 0.75).abs() < 1e-9, "y_bot got {}", b[3]);
        // Offset render box (crop): the same span in a box starting at (20,10).
        // Span (30,60)-(50,70) user y-up; box top py=10+200=210. y-down top =
        // (210-70)/200 = 0.7, bottom = (210-60)/200 = 0.75.
        let b2 = pdf_box_to_norm(30.0, 60.0, 50.0, 70.0, 20.0, 10.0, 100.0, 200.0, 0);
        assert!((b2[0] - 0.1).abs() < 1e-9);
        assert!((b2[1] - 0.7).abs() < 1e-9);
        // Clamped out of the render box.
        let b3 = pdf_box_to_norm(-50.0, -50.0, 250.0, 250.0, 0.0, 0.0, 100.0, 200.0, 0);
        assert_eq!(b3, [0.0, 0.0, 1.0, 1.0]);
        // 180°: the page is turned; the same point maps to the opposite corner.
        let b4 = pdf_box_to_norm(10.0, 50.0, 30.0, 60.0, 0.0, 0.0, 100.0, 200.0, 180);
        assert!((b4[0] - 0.7).abs() < 1e-9, "180 x0 got {}", b4[0]);
        assert!((b4[1] - 0.25).abs() < 1e-9, "180 y0 got {}", b4[1]);
        assert!((b4[2] - 0.9).abs() < 1e-9, "180 x1 got {}", b4[2]);
        assert!((b4[3] - 0.3).abs() < 1e-9, "180 y1 got {}", b4[3]);
        // Degenerate box → zeros.
        assert_eq!(
            pdf_box_to_norm(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0),
            [0.0; 4]
        );
    }

    #[test]
    fn spans_cluster_by_line_and_read_top_down() {
        // Page box 0,0,100,200, rotation 0. Two lines: a two-column row at
        // the top (y 150-160) and a single span below (y 100-110).
        let spans = vec![
            (10.0, 100.0, 50.0, 110.0, "second".to_string()), // lower line
            (60.0, 150.0, 90.0, 160.0, "right".to_string()),  // top row, col 2
            (10.0, 150.0, 40.0, 160.0, "left".to_string()),   // top row, col 1
        ];
        let blocks = spans_to_text_blocks(&spans, 0.0, 0.0, 100.0, 200.0, 0);
        assert_eq!(blocks.len(), 2, "rows: {:?}", blocks);
        // Top line first; its two columns merged into one block.
        assert_eq!(blocks[0].text, "left right");
        assert_eq!(blocks[0].label, TEXT);
        // The row's bbox spans both columns: x 10..90. In y-down it sits at
        // the TOP of the page (user y 150..160 of a 200-pt box → 0.2..0.25).
        assert!((blocks[0].bbox[0] - 0.1).abs() < 1e-9);
        assert!((blocks[0].bbox[2] - 0.9).abs() < 1e-9);
        assert!((blocks[0].bbox[1] - 0.2).abs() < 1e-9);
        assert!((blocks[0].bbox[3] - 0.25).abs() < 1e-9);
        // The lower line.
        assert_eq!(blocks[1].text, "second");
        // Empty span set → no blocks.
        assert!(spans_to_text_blocks(&[], 0.0, 0.0, 100.0, 200.0, 0).is_empty());
        // Degenerate page box → no blocks.
        assert!(spans_to_text_blocks(&spans, 0.0, 0.0, 0.0, 0.0, 0).is_empty());
    }

    #[test]
    fn relabel_l0_blocks_by_region_center() {
        // Three line blocks in the [0,1] y-down space: a title row at the
        // top, two body rows below.
        let mk = |y: f64, t: &str| Block {
            label: TEXT,
            bbox: [0.1, y, 0.9, y + 0.02],
            text: t.to_string(),
        };
        let blocks = vec![mk(0.05, "title"), mk(0.30, "body1"), mk(0.60, "body2")];
        // A title region covering the top row and a table region covering the
        // second row (a wider, coarser table region too, to test
        // smallest-wins).
        let regions = [
            (0.0, 0.0, 1.0, 0.15, HEADING),
            (0.05, 0.2, 0.95, 0.5, TABLE),
            (0.0, 0.25, 1.0, 0.9, TEXT), // large coarse region, same center hit
        ];
        let out = relabel_l0_blocks(blocks, &regions);
        assert_eq!(out[0].label, HEADING, "title row inside the heading region");
        // Center of body1 (0.5, 0.31) falls in BOTH the TABLE region and the
        // coarse TEXT region → the smaller (table) wins.
        assert_eq!(out[1].label, TABLE);
        // body2 (center 0.71) is inside only the coarse TEXT region → stays
        // text.
        assert_eq!(out[2].label, TEXT);
        // Empty regions → every block keeps its label.
        let same = relabel_l0_blocks(
            vec![mk(0.5, "x")],
            &[],
        );
        assert_eq!(same[0].label, TEXT);
        // Text/bbox are untouched by the relabel.
        assert_eq!(out[0].bbox, [0.1, 0.05, 0.9, 0.07]);
        assert_eq!(out[0].text, "title");
    }

    #[test]
    fn layout_regions_map_title_and_table_only() {
        use rusto::layout::{BBox, LayoutRegion, LayoutType};
        fn reg(t: LayoutType, conf: f32, x0: i32, y0: i32, x1: i32, y1: i32) -> LayoutRegion {
            LayoutRegion {
                bbox: BBox {
                    x_min: x0,
                    y_min: y0,
                    x_max: x1,
                    y_max: y1,
                },
                layout_type: t,
                confidence: conf,
            }
        }
        // 100x100 page: a title bar, a compact table, a text band, a figure,
        // a low-confidence table, and a page-wide table blob.
        let regions = vec![
            reg(LayoutType::Title, 0.9, 10, 5, 90, 20),
            reg(LayoutType::Table, 0.8, 5, 30, 95, 50),
            reg(LayoutType::Text, 0.9, 0, 60, 100, 80),
            reg(LayoutType::Figure, 0.9, 0, 80, 100, 99),
            reg(LayoutType::Table, 0.3, 0, 0, 90, 90),
            reg(LayoutType::Table, 0.9, 0, 0, 99, 99),
        ];
        let out = layout_regions_to_relabel(&regions, 100.0, 100.0);
        assert_eq!(out.len(), 2, "only the confident compact title+table: {out:?}");
        assert_eq!(out[0], (0.1, 0.05, 0.9, 0.2, HEADING));
        assert_eq!(out[1], (0.05, 0.3, 0.95, 0.5, TABLE));
        // Boundary: conf exactly RELABEL_MIN_CONF does not pass (`>`).
        let edge = vec![reg(LayoutType::Title, RELABEL_MIN_CONF, 0, 0, 50, 50)];
        assert!(layout_regions_to_relabel(&edge, 100.0, 100.0).is_empty());
        // Degenerate page → empty.
        assert!(layout_regions_to_relabel(&regions, 0.0, 100.0).is_empty());
    }
}
