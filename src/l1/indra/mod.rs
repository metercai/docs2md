//! The Indra L1 engine (port of indra-ocr 1.1.2): 8 ONNX models + 3 n-gram LM
//! tables embedded under assets/models/indra/. P0 delivers the skeleton:
//! model loading, the rten session wrapper, npz/n-gram LM parsing, and the
//! vocab/meta table. The pipeline (det -> rec -> order -> layout ->
//! markdown) lands in P1-P6.
//!
//! rten 0.27 imports ONNX INT64 tensors as Int32 (onnx_loader.rs:350), so
//! the rec model's widths/ids/topi I/O is handled as i32 throughout.

pub mod det;
pub mod layer_fusion;
pub mod lm;
pub mod markdown;
pub mod npz;
pub mod order;
pub mod preproc;
pub mod rec;
pub mod rtl;
pub mod vocab;

use std::collections::HashMap;

use rten::Model;
use rten_tensor::prelude::*;

pub(crate) mod assets {
    pub const META: &[u8] = include_bytes!("../../../assets/models/indra/meta.json");
    pub const LM: &[u8] = include_bytes!("../../../assets/models/indra/lm.npz");
    pub const LM_CJK: &[u8] = include_bytes!("../../../assets/models/indra/lm_cjk.npz");
    pub const LM_WORLD: &[u8] = include_bytes!("../../../assets/models/indra/lm_world.npz");
    pub const DET: &[u8] = include_bytes!("../../../assets/models/indra/det.onnx");
    pub const REC: &[u8] = include_bytes!("../../../assets/models/indra/rec.onnx");
    pub const REC_CJK: &[u8] = include_bytes!("../../../assets/models/indra/rec_cjk.onnx");
    pub const REC_RTL: &[u8] = include_bytes!("../../../assets/models/indra/rec_rtl.onnx");
    pub const REC_WORLD: &[u8] = include_bytes!("../../../assets/models/indra/rec_world.onnx");
    pub const ORDER: &[u8] = include_bytes!("../../../assets/models/indra/order.onnx");
    pub const ORIENT: &[u8] = include_bytes!("../../../assets/models/indra/orient.onnx");
    pub const SCRIPT: &[u8] = include_bytes!("../../../assets/models/indra/script.onnx");
}

/// One output tensor in the element type rten 0.27 actually produces
/// (ONNX float32 -> Float, ONNX int64/int32 -> Int).
#[derive(Debug)]
pub enum Tensor {
    Float { shape: Vec<usize>, data: Vec<f32> },
    Int { shape: Vec<usize>, data: Vec<i32> },
}

impl Tensor {
    pub fn as_float(&self) -> Option<&Vec<f32>> {
        match self {
            Tensor::Float { data, .. } => Some(data),
            _ => None,
        }
    }
    pub fn as_int(&self) -> Option<&Vec<i32>> {
        match self {
            Tensor::Int { data, .. } => Some(data),
            _ => None,
        }
    }
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::Float { shape, .. } | Tensor::Int { shape, .. } => shape,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Tensor::Float { data, .. } => data.len(),
            Tensor::Int { data, .. } => data.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn value_to_tensor(v: &rten::Value) -> Result<Tensor, String> {
    match v {
        rten::Value::FloatTensor(t) => Ok(Tensor::Float {
            shape: t.shape().to_vec(),
            data: t.iter().copied().collect(),
        }),
        rten::Value::Int32Tensor(t) => Ok(Tensor::Int {
            shape: t.shape().to_vec(),
            data: t.iter().copied().collect(),
        }),
        rten::Value::Int8Tensor(t) => Ok(Tensor::Int {
            shape: t.shape().to_vec(),
            data: t.iter().map(|x| *x as i32).collect(),
        }),
        rten::Value::UInt8Tensor(t) => Ok(Tensor::Int {
            shape: t.shape().to_vec(),
            data: t.iter().map(|x| *x as i32).collect(),
        }),
        other => Err(format!("unsupported output value: {:?}", other.dtype())),
    }
}

/// A loaded rten model plus its input/output node ids (name-ordered).
pub struct Session {
    pub model: Model,
    pub inputs: Vec<(String, rten::NodeId)>,
    pub outputs: Vec<(String, rten::NodeId)>,
}

impl Session {
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        let model = Model::load(bytes.to_vec()).map_err(|e| format!("rten load: {e}"))?;
        let inputs: Vec<(String, rten::NodeId)> = model
            .input_ids()
            .iter()
            .map(|id| {
                let name = model
                    .node_info(*id)
                    .and_then(|i| i.name().map(|s| s.to_string()))
                    .unwrap_or_else(|| "in0".into());
                (name, *id)
            })
            .collect();
        let outputs: Vec<(String, rten::NodeId)> = model
            .output_ids()
            .iter()
            .map(|id| {
                let name = model
                    .node_info(*id)
                    .and_then(|i| i.name().map(|s| s.to_string()))
                    .unwrap_or_else(|| "out0".into());
                (name, *id)
            })
            .collect();
        Ok(Self {
            model,
            inputs,
            outputs,
        })
    }

    fn input_id(&self, name: &str) -> Result<rten::NodeId, String> {
        self.inputs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, id)| *id)
            .ok_or_else(|| {
                format!(
                    "input '{name}' not found (have {:?})",
                    self.inputs
                        .iter()
                        .map(|(n, _)| n.as_str())
                        .collect::<Vec<_>>()
                )
            })
    }

    /// Run with one float input.
    pub fn run_f(&self, input: &str, shape: &[usize], data: &[f32]) -> Result<Vec<Tensor>, String> {
        let id = self.input_id(input)?;
        let t = rten_tensor::Tensor::from_data(shape, data.to_vec());
        let outs = self
            .model
            .run(
                vec![(id, t.view().into())],
                &self.outputs.iter().map(|(_, i)| *i).collect::<Vec<_>>(),
                None,
            )
            .map_err(|e| format!("rten run: {e}"))?;
        self.collect(outs)
    }

    /// Run with a float + an int input (the rec models: x + widths).
    pub fn run_fi(
        &self,
        float_in: &str,
        float_shape: &[usize],
        float_data: &[f32],
        int_in: &str,
        int_shape: &[usize],
        int_data: &[i32],
    ) -> Result<Vec<Tensor>, String> {
        let fid = self.input_id(float_in)?;
        let iid = self.input_id(int_in)?;
        let ft = rten_tensor::Tensor::from_data(float_shape, float_data.to_vec());
        let it = rten_tensor::Tensor::from_data(int_shape, int_data.to_vec());
        let outs = self
            .model
            .run(
                vec![(fid, ft.view().into()), (iid, it.view().into())],
                &self.outputs.iter().map(|(_, i)| *i).collect::<Vec<_>>(),
                None,
            )
            .map_err(|e| format!("rten run: {e}"))?;
        self.collect(outs)
    }

    fn collect(&self, outs: Vec<rten::Value>) -> Result<Vec<Tensor>, String> {
        let mut r = Vec::with_capacity(outs.len());
        for o in &outs {
            r.push(value_to_tensor(o)?);
        }
        Ok(r)
    }
}

/// The Indra engine. P0: models loaded and queryable; the ocr_page pipeline
/// lands in P1-P6.
pub struct IndraEngine {
    pub meta: vocab::Meta,
    pub lms: HashMap<String, lm::NgramLm>,
    pub det: Session,
    pub rec: HashMap<String, Session>,
    pub order: Session,
    pub orient: Session,
    pub script: Session,
}

/// `assemble` output: (texts, confs, tidy'd boxes, reading order, markdown,
/// the same-block matrix used to group lines into blocks — `[]` when the
/// order model ran no grid head — the per-line Chandra block-class logits
/// `cls` of shape `[n_lines, 10]`, `[]` when the order model emitted none).
type Assembled = (
    Vec<String>,
    Vec<f64>,
    Vec<[f64; 4]>,
    Vec<usize>,
    String,
    Vec<f32>,
    Vec<f32>,
);

impl IndraEngine {
    /// Load every model from the embedded assets.
    pub fn init() -> Result<Self, String> {
        let meta = vocab::Meta::parse(assets::META)?;
        let lms = [
            ("base", assets::LM),
            ("cjk", assets::LM_CJK),
            ("world", assets::LM_WORLD),
        ]
        .into_iter()
        .map(|(name, b)| -> Result<_, String> {
            let arr = npz::parse_npz(b)?;
            Ok((name.to_string(), lm::NgramLm::from_arrays(&arr)?))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;
        Ok(Self {
            meta,
            lms,
            det: Session::load(assets::DET)?,
            rec: [
                ("base", assets::REC),
                ("cjk", assets::REC_CJK),
                ("rtl", assets::REC_RTL),
                ("world", assets::REC_WORLD),
            ]
            .into_iter()
            .map(|(name, b)| -> Result<_, String> { Ok((name.to_string(), Session::load(b)?)) })
            .collect::<Result<HashMap<_, _>, _>>()?,
            order: Session::load(assets::ORDER)?,
            orient: Session::load(assets::ORIENT)?,
            script: Session::load(assets::SCRIPT)?,
        })
    }

    /// OCR one page image (product `Pipeline.__call__`, `turn=True`):
    /// preview -> skew -> orientation -> turn -> detect -> recognize ->
    /// assemble (gutter/split -> tidy -> reading order -> blocks -> markdown
    /// with the table post-filter applied).
    pub fn ocr_page(
        &mut self,
        gray: &preproc::Gray,
        rc: &crate::config::RouteConfig,
    ) -> Result<super::OcrPage, String> {
        self.ocr_page_with_layer(gray, rc, None)
    }

    /// `ocr_page` with the page's usable text layer (plan 0.3.2 T4): covered
    /// lines are read from the exact PDF characters instead of the OCR guess;
    /// uncovered lines keep the OCR result. `None` runs the plain pipeline
    /// (byte-identical to `ocr_page`).
    pub fn ocr_page_with_layer(
        &mut self,
        gray: &preproc::Gray,
        rc: &crate::config::RouteConfig,
        layer: Option<&super::LayerPage>,
    ) -> Result<super::OcrPage, String> {
        let small = preproc::preview(gray, preproc::PREVIEW_SIDE);
        let a = preproc::estimate_skew(&small, 800);
        // The orientation model reads the page straightened by its row angle
        // (a tilted page could read as turned).
        let k = if a.abs() >= preproc::DESKEW_MIN {
            self.orientation(&preproc::rotate_page(&small, a))?
        } else {
            self.orientation(&small)?
        };
        self.read(gray, k, a, &small, rc, layer)
    }

    /// orient.onnx quarter-turn choice (product `orientation`): argmax over the
    /// class probabilities, 0 when no turn is more likely than ORIENT_MIN.
    fn orientation(&self, gray: &preproc::Gray) -> Result<i32, String> {
        let ink = preproc::orient_input(gray, preproc::ORIENT_SIDE);
        let outs = self
            .orient
            .run_f(
                "x",
                &[1, 1, preproc::ORIENT_SIDE, preproc::ORIENT_SIDE],
                &ink.data,
            )
            .map_err(|e| format!("orient run: {e}"))?;
        let p = outs[0]
            .as_float()
            .cloned()
            .ok_or_else(|| "orient output is not float".to_string())?;
        Ok(preproc::orientation_from_probs(&p))
    }

    /// Product `_read`: turn the page, detect, recognize, assemble, then map
    /// the line boxes back onto the page as it is (deskew and vertical turn
    /// undone). The L1→L2 serial metric chain (plan §2.1) is evaluated as
    /// the pipeline advances — after det, after the script pre-pass, after
    /// rec, after assemble — and the first hit stops the pipeline (saving
    /// the later stages) and tags the (possibly partial) output
    /// `escalate_l2` for the orchestrator.
    fn read(
        &mut self,
        gray: &preproc::Gray,
        k: i32,
        a: f64,
        small: &preproc::Gray,
        rc: &crate::config::RouteConfig,
        layer: Option<&super::LayerPage>,
    ) -> Result<super::OcrPage, String> {
        // The display (render) size, captured before `turn` shadows `gray`
        // with the worked frame — the layer's point map needs it.
        let render_wh = (gray.w, gray.h);
        let (gray, _k, skew) = preproc::turn(gray, k, a, small);
        let (boxes, pictures) =
            det::detect(&self.det, &gray, self.meta.det_long_side as usize, true);
        let vertical = k == preproc::VERTICAL;

        // [det] stage: a near-image page (fewer line boxes than
        // det_min_lines) escalates before any recognition — the whole rec
        // pass is saved.
        if let Some(tag) = crate::route::chain_det(boxes.len(), rc) {
            return Ok(super::OcrPage {
                lines: Vec::new(),
                md: Some(String::new()),
                blocks: Vec::new(),
                escalate_l2: Some(tag),
            });
        }

        // Script pre-pass (crops + per-line script routing), computed ONCE:
        // the chain reads its metrics before any recognition, and
        // recognize_groups reuses the same crops (no second pass).
        let rs = rec::route_script(self, &gray, &boxes, vertical);
        let loaded: Vec<String> = self.rec.keys().cloned().collect();
        let loaded: Vec<&str> = loaded.iter().map(|s| s.as_str()).collect();
        if let Some(tag) = crate::route::chain_script(&rs.metrics(&loaded), rc) {
            return Ok(super::OcrPage {
                lines: Vec::new(),
                md: Some(String::new()),
                blocks: Vec::new(),
                escalate_l2: Some(tag),
            });
        }

        let mut lines = rec::recognize_groups(self, &rs, &boxes);
        if lines.is_empty() {
            // Boxes were detected but nothing could be read: the conf
            // criterion is undefined (mean conf of zero lines), treat the
            // page as the chars criterion firing.
            return Ok(super::OcrPage {
                lines: Vec::new(),
                md: Some(String::new()),
                blocks: Vec::new(),
                escalate_l2: Some(crate::route::TAG_CHARS),
            });
        }

        // T4 (plan 0.3.2): fuse the page's text layer onto the detected
        // lines — a covered line is read from the exact PDF characters
        // (conf 1.0, no OCR) instead of the recognizer's guess; an
        // uncovered line keeps its OCR reading. No-op when the page has no
        // usable layer (layer_fusion off, a pure scan, or a rotated page).
        if let Some(layer) = layer {
            let (fused, _lost) = layer_fusion::fuse_lines(layer, &boxes, k, skew, &gray, render_wh);
            for (i, f) in fused.into_iter().enumerate() {
                if let Some(f) = f {
                    let toks: Vec<markdown::Tok> = f
                        .tokens
                        .iter()
                        .map(|(c, x)| markdown::Tok::Ch(*c, *x))
                        .collect();
                    lines[i].text = markdown::decode_tokens(&toks, &self.meta.vocab["base"], false);
                    lines[i].conf = 1.0;
                    lines[i].toks = toks;
                }
            }
        }

        // [rec] stage: mean line conf below rec_hard_conf escalates before
        // the order model and markdown assembly (the former G3 conf
        // criterion, now without the char-count precondition). The partial
        // L1 output is kept as the low-confidence fallback.
        let mean_conf: f32 = lines.iter().map(|l| l.conf).sum::<f32>() / lines.len() as f32;
        if let Some(tag) = crate::route::output_gate(usize::MAX, mean_conf, rc) {
            let page = self.partial_page(&gray, &boxes, &lines, vertical, skew);
            return Ok(page.with_escalate(Some(tag)));
        }

        let (texts, confs, boxes, order, md, same, cls) =
            self.assemble(&gray, &boxes, &lines, vertical, pictures)?;

        // [assemble] stage: the output-level char gate (former G3
        // min_chars) on the assembled markdown.
        if let Some(tag) = crate::route::output_gate(md.chars().count(), mean_conf, rc) {
            let page = self.full_page(
                &gray, &boxes, &texts, &confs, &order, &md, &same, &cls, vertical, skew,
            );
            return Ok(page.with_escalate(Some(tag)));
        }

        Ok(self.full_page(
            &gray, &boxes, &texts, &confs, &order, &md, &same, &cls, vertical, skew,
        ))
    }

    /// The chain's early-exit page (rec stage): the recognized lines mapped
    /// back onto the page as it is, WITHOUT the order model, blocks, or
    /// markdown (they would cost as much as the stages the chain just
    /// saved).
    fn partial_page(
        &self,
        gray: &preproc::Gray,
        boxes: &[[f64; 4]],
        lines: &[rec::Line],
        vertical: bool,
        skew: f64,
    ) -> super::OcrPage {
        let n = boxes.len();
        let wh = (gray.w, gray.h);
        let olines: Vec<super::OcrLine> = (0..n)
            .map(|i| {
                let mut b = boxes[i];
                if skew != 0.0 {
                    b = preproc::unrotate_box(b, skew, wh);
                }
                let b = if vertical {
                    let wpx = gray.h as f64;
                    [wpx - b[3], b[0], wpx - b[1], b[2]]
                } else {
                    b
                };
                super::OcrLine {
                    text: lines[i].text.clone(),
                    conf: ((lines[i].conf as f32) * 10000.0).round() / 10000.0,
                    top: b[1] as f32,
                    left: b[0] as f32,
                    width: (b[2] - b[0]) as f32,
                    height: (b[3] - b[1]) as f32,
                }
            })
            .collect();
        super::OcrPage {
            lines: olines,
            md: Some(String::new()),
            blocks: Vec::new(),
            escalate_l2: None,
        }
    }

    /// A fully assembled page (reading order, blocks, markdown) with the
    /// line boxes mapped back onto the page as it is (deskew and vertical
    /// turn undone).
    fn full_page(
        &self,
        gray: &preproc::Gray,
        boxes: &[[f64; 4]],
        texts: &[String],
        confs: &[f64],
        order: &[usize],
        md: &str,
        same: &[f32],
        cls: &[f32],
        vertical: bool,
        skew: f64,
    ) -> super::OcrPage {
        // Lines in reading order, boxes rounded to 0.1 (product Line), then
        // mapped back onto the page as it is.
        let mut line_boxes: Vec<[f64; 4]> = order
            .iter()
            .map(|&i| {
                [
                    (boxes[i][0] * 10.0).round() / 10.0,
                    (boxes[i][1] * 10.0).round() / 10.0,
                    (boxes[i][2] * 10.0).round() / 10.0,
                    (boxes[i][3] * 10.0).round() / 10.0,
                ]
            })
            .collect();
        let wh = (gray.w, gray.h);
        if skew != 0.0 {
            // Boxes back on the page as turned, before the deskew.
            for b in line_boxes.iter_mut() {
                *b = preproc::unrotate_box(*b, skew, wh);
            }
        }
        if vertical {
            // Back on the page as it is: (x, y) -> (W - y, x).
            let wpx = gray.h as f64;
            for b in line_boxes.iter_mut() {
                *b = [wpx - b[3], b[0], wpx - b[1], b[2]];
            }
        }
        let olines: Vec<super::OcrLine> = order
            .iter()
            .enumerate()
            .map(|(pos, &i)| {
                let b = line_boxes[pos];
                super::OcrLine {
                    text: texts[i].clone(),
                    conf: ((confs[i] as f32) * 10000.0).round() / 10000.0,
                    top: b[1] as f32,
                    left: b[0] as f32,
                    width: (b[2] - b[0]) as f32,
                    height: (b[3] - b[1]) as f32,
                }
            })
            .collect();
        // Structured blocks (E2): one per same-block group, in reading order,
        // in the same page space as the lines (deskew/turn already undone).
        // The label is the group's argmax Chandra class (cls summed over the
        // group's lines, as `page_blocks` does) mapped through CHANDRA_TO_OPB;
        // `text` when the order model emitted no class head.
        let pos_of: Vec<usize> = {
            let mut v = vec![0usize; order.len()];
            for (p, &i) in order.iter().enumerate() {
                v[i] = p;
            }
            v
        };
        let groups: Vec<Vec<usize>> = if same.is_empty() {
            order.iter().map(|&i| vec![i]).collect()
        } else {
            markdown::split_blocks(&order, &same)
        };
        let has_cls = !cls.is_empty();
        let pw = gray.w as f64;
        let ph = gray.h as f64;
        let blocks: Vec<super::Block> = groups
            .iter()
            .map(|g| {
                let (mut x0, mut y0, mut x1, mut y1) = (
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                );
                for &i in g {
                    let b = line_boxes[pos_of[i]];
                    x0 = x0.min(b[0]);
                    y0 = y0.min(b[1]);
                    x1 = x1.max(b[2]);
                    y1 = y1.max(b[3]);
                }
                let text = g
                    .iter()
                    .map(|&i| texts[i].as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let label = if has_cls {
                    crate::blocks::chandra_to_opb(markdown::block_class(g, &cls))
                } else {
                    crate::blocks::TEXT
                };
                super::Block {
                    label,
                    bbox: crate::blocks::normalize_bbox(x0, y0, x1, y1, pw, ph),
                    text,
                }
            })
            .collect();
        super::OcrPage {
            lines: olines,
            md: Some(md.to_string()),
            blocks,
            escalate_l2: None,
        }
    }

    /// Product `assemble`: lines -> split at wide gaps (SPLIT_GAP) -> tidy ->
    /// reading order -> blocks -> markdown.  The table post-filter (new Rust
    /// logic, plan §1.5) demotes full-page low-confidence tables to plain
    /// joined text before the markdown is built.
    fn assemble(
        &self,
        gray: &preproc::Gray,
        boxes_in: &[[f64; 4]],
        lines: &[rec::Line],
        vertical: bool,
        pictures: Vec<[f64; 4]>,
    ) -> Result<Assembled, String> {
        // A right-to-left page: reading order and markdown work on the page
        // mirrored left to right, so the first column is the right one.
        let mut letters = 0usize;
        let mut n_rtl = 0usize;
        for ln in lines {
            for c in ln.text.chars() {
                if order::is_py_alpha(c) {
                    letters += 1;
                    if rtl::is_rtl(c) {
                        n_rtl += 1;
                    }
                }
            }
        }
        let page_rtl = n_rtl > letters - n_rtl;
        let width = gray.w as f64;
        let geo = |b: [f64; 4]| -> [f64; 4] {
            if page_rtl {
                [width - b[2], b[1], width - b[0], b[3]]
            } else {
                b
            }
        };
        let geo_vec = |bs: &[[f64; 4]]| -> Vec<[f64; 4]> { bs.iter().map(|b| geo(*b)).collect() };

        // Text-layer labels split a line at wide gaps; do the same (SPLIT_GAP).
        let toks: Vec<Vec<markdown::Tok>> = lines.iter().map(|l| l.toks.clone()).collect();
        let space_id = self.space_id();
        let vocab = &self.meta.vocab["base"];
        let cuts = markdown::gutter_cuts(boxes_in, &toks, space_id);
        let mut nb: Vec<[f64; 4]> = Vec::new();
        let mut nrec: Vec<(String, f64, Vec<markdown::Tok>)> = Vec::new();
        for (i, b) in boxes_in.iter().enumerate() {
            for (bx, tx, tk) in markdown::split_line(
                *b,
                &lines[i].toks,
                markdown::SPLIT_GAP,
                cuts.get(i).map(|v| v.as_slice()),
                space_id,
                vocab,
                page_rtl,
            ) {
                nb.push(bx);
                nrec.push((tx, lines[i].conf as f64, tk));
            }
        }
        let mut boxes = nb;
        markdown::tidy_lines(&mut boxes, &mut nrec, page_rtl, space_id);
        let texts: Vec<String> = nrec.iter().map(|r| r.0.clone()).collect();
        let confs: Vec<f64> = nrec.iter().map(|r| r.1).collect();
        let rec_toks: Vec<Vec<markdown::Tok>> = nrec.iter().map(|r| r.2.clone()).collect();

        let ord = self.reading_order(&geo_vec(&boxes), &texts, (gray.w as f64, gray.h as f64))?;
        let (mut order, mut cls, mut same) = match (ord.cls, ord.same) {
            (Some(c), Some(s)) => (ord.order, c, s),
            _ => (ord.order, Vec::new(), Vec::new()),
        };
        if vertical && !same.is_empty() {
            // The order model saw no vertical pages: turned, their columns of
            // short lines look like one table read row by row across tiers.
            // Blocks and order come from XY-cut instead, without grid classes.
            let f32b: Vec<[f32; 4]> = boxes
                .iter()
                .map(|b| [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32])
                .collect();
            let blocks = order::xycut_blocks(&f32b, 1.0);
            order = blocks.iter().flatten().copied().collect();
            let n = boxes.len();
            same = vec![-1.0f32; n * n];
            for b in &blocks {
                for &i in b {
                    for &j in b {
                        same[j * n + i] = 1.0;
                    }
                }
            }
            for &c in &order::GRID_CLASSES {
                for i in 0..n {
                    cls[i * order::N_BLOCK_CLASSES + c] = -1e4;
                }
            }
        }

        // The table heads learned one line per cell, but the detector can join
        // cells: with table_heads on the product re-splits table lines and
        // re-runs the order model.  This build runs the product default
        // (table_heads=False, no tab head), so the branch never fires.

        if !same.is_empty() {
            let gboxes = geo_vec(&boxes);
            // Inside tables, labels split cells at smaller gaps (CELL_GAP) —
            // every line, as the product computes cells for all of them.
            let cells: markdown::Cells = (0..boxes.len())
                .map(|i| {
                    markdown::split_line(
                        boxes[i],
                        &rec_toks[i],
                        markdown::CELL_GAP,
                        None,
                        space_id,
                        vocab,
                        page_rtl,
                    )
                    .into_iter()
                    .map(|(bx, tx, _)| (geo(bx).to_vec(), tx))
                    .collect()
                })
                .collect();
            let block_groups: Vec<Vec<usize>> = markdown::split_blocks(&order, &same);
            let has_caption = block_groups
                .iter()
                .any(|b| markdown::block_class(b, &cls) == markdown::CAPTION);
            let keep = markdown::table_postfilter(
                &block_groups,
                &gboxes,
                &cls,
                &confs.iter().map(|&c| c as f32).collect::<Vec<_>>(),
                (gray.w * gray.h) as f64,
                has_caption,
            );
            let pics: Option<Vec<[f64; 4]>> = if vertical || pictures.is_empty() {
                None
            } else {
                Some(geo_vec(&pictures))
            };
            let md = markdown::page_markdown(
                &gboxes,
                &texts,
                &order,
                &cls,
                &same,
                false,
                Some(&cells),
                None,
                pics.as_deref(),
                Some(&keep),
            );
            Ok((texts, confs, boxes, order, md, same, cls))
        } else {
            let md = texts.join("\n");
            Ok((texts, confs, boxes, order, md, same, cls))
        }
    }

    /// The base vocab's space token class id (product `vocab.index.get(" ")`).
    fn space_id(&self) -> i32 {
        self.meta.vocab["base"]
            .chars
            .iter()
            .position(|c| c == " ")
            .map(|p| (p + 1) as i32)
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> IndraEngine {
        IndraEngine::init().expect("indra init (all 8 ONNX + 3 LM + meta)")
    }

    #[test]
    fn all_models_load_with_expected_io() {
        let e = engine();
        assert_eq!(
            e.det
                .inputs
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["x"]
        );
        assert_eq!(
            e.det
                .outputs
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["out"]
        );
        for g in ["base", "cjk", "rtl", "world"] {
            let s = &e.rec[g];
            assert_eq!(
                s.inputs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
                vec!["x", "widths"]
            );
            let onames: Vec<&str> = s.outputs.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(
                onames,
                vec!["ids", "conf", "lengths", "blank", "topv", "topi"]
            );
        }
        assert_eq!(
            e.order
                .inputs
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["feats", "mask"]
        );
        assert_eq!(
            e.orient
                .inputs
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["x"]
        );
        assert_eq!(
            e.script
                .inputs
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["x", "widths"]
        );
    }

    // Smoke inferences with fixed inputs; gate = outputs come back with the
    // declared shapes and sane element ranges.
    #[test]
    fn det_runs() {
        let e = engine();
        // 1x1x8x8 (the det model is fully convolutional on a 1024-scaled page;
        // any H/W works at smoke level). Output is [1, 3, h, w] of raw LOGITS:
        // channel 0 is sigmoided in decode (P2), channels 1/2 are top/bottom
        // box offsets. Smoke gate: all channels finite.
        let data = vec![0.5f32; 64];
        let out = e.det.run_f("x", &[1, 1, 8, 8], &data).expect("det run");
        assert_eq!(&out[0].shape()[..2], &[1, 3]);
        let t = out[0].as_float().unwrap();
        assert!(t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rec_runs() {
        let e = engine();
        // base rec expects [B,1,96,W]; widths in [0,W].
        let w = 64usize;
        let data = vec![0.0f32; 96 * w];
        let out = e.rec["base"]
            .run_fi("x", &[1, 1, 96, w], &data, "widths", &[1], &[w as i32])
            .expect("rec run");
        let names: Vec<&str> = vec!["ids", "conf", "lengths", "blank", "topv", "topi"];
        for (t, n) in out.iter().zip(names) {
            assert_eq!(t.shape()[0], 1, "{n}");
        }
        // zero input -> no non-blank tokens expected; at minimum ids are valid
        // classes and lengths stays within [0, W].
        let ids = out[0].as_int().expect("ids int");
        assert!(ids.iter().all(|&i| (0..=552).contains(&i)));
        let lengths = out[2].as_int().expect("lengths int");
        assert!((0..=w as i32).contains(&lengths[0]));
    }

    #[test]
    fn script_runs() {
        let e = engine();
        // script shares the rec input contract: [B,1,32,W] + widths.
        let w = 32usize;
        let data = vec![0.0f32; 32 * w];
        let out = e
            .script
            .run_fi("x", &[1, 1, 32, w], &data, "widths", &[1], &[w as i32])
            .expect("script run");
        assert_eq!(out[0].shape(), &vec![1, 4]);
        // softmax over 4 script groups.
        let p = out[0].as_float().unwrap();
        assert!(p.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn orient_runs() {
        let e = engine();
        let n = 768 * 768;
        let data = vec![0.0f32; n];
        let out = e
            .orient
            .run_f("x", &[1, 1, 768, 768], &data)
            .expect("orient run");
        assert_eq!(out[0].shape(), &vec![1, 5]);
    }

    #[test]
    fn order_runs() {
        let e = engine();
        let n = 2usize;
        let feats = vec![0.1f32; n * 16];
        // ONNX mask is bool; rten 0.27 imports BOOL as i32 (0/1).
        let mask = vec![1i32; n];
        let fid = e.order.input_id("feats").unwrap();
        let mid = e.order.input_id("mask").unwrap();
        let ft = rten_tensor::Tensor::from_data(&[1, n, 16], feats);
        let mt = rten_tensor::Tensor::from_data(&[1, n], mask);
        let oids: Vec<rten::NodeId> = e.order.outputs.iter().map(|(_, i)| *i).collect();
        let outs = e
            .order
            .model
            .run(
                vec![(fid, ft.view().into()), (mid, mt.view().into())],
                &oids,
                None,
            )
            .expect("order run");
        let t0 = value_to_tensor(&outs[0]).unwrap();
        assert_eq!(t0.shape()[0], 1);
    }
}
