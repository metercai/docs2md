//! The Indra L1 engine (port of indra-ocr 1.1.2): 8 ONNX models + 3 n-gram LM
//! tables embedded under assets/models/indra/. P0 delivers the skeleton:
//! model loading, the rten session wrapper, npz/n-gram LM parsing, and the
//! vocab/meta table. The pipeline (det -> rec -> order -> layout ->
//! markdown) lands in P1-P6.
//!
//! rten 0.27 imports ONNX INT64 tensors as Int32 (onnx_loader.rs:350), so
//! the rec model's widths/ids/topi I/O is handled as i32 throughout.

pub mod lm;
pub mod npz;
pub mod vocab;

use std::collections::HashMap;

use rten::Model;
use rten_tensor::prelude::*;

pub(crate) mod assets {
    pub const META: &[u8] = include_bytes!("../../../assets/models/indra/meta.json");
    pub const LM: &[u8] = include_bytes!("../../../assets/models/indra/lm.npz");
    pub const LM_CJK: &[u8] = include_bytes!("../../../assets/models/indra/lm_cjk.npz");
    pub const LM_WORLD: &[u8] = include_bytes!(
        "../../../assets/models/indra/lm_world.npz"
    );
    pub const DET: &[u8] = include_bytes!("../../../assets/models/indra/det.onnx");
    pub const REC: &[u8] = include_bytes!("../../../assets/models/indra/rec.onnx");
    pub const REC_CJK: &[u8] = include_bytes!(
        "../../../assets/models/indra/rec_cjk.onnx"
    );
    pub const REC_RTL: &[u8] = include_bytes!("../../../assets/models/indra/rec_rtl.onnx");
    pub const REC_WORLD: &[u8] = include_bytes!(
        "../../../assets/models/indra/rec_world.onnx"
    );
    pub const ORDER: &[u8] = include_bytes!("../../../assets/models/indra/order.onnx");
    pub const ORIENT: &[u8] = include_bytes!("../../../assets/models/indra/orient.onnx");
    pub const SCRIPT: &[u8] = include_bytes!(
        "../../../assets/models/indra/script.onnx"
    );
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
        Ok(Self { model, inputs, outputs })
    }

    fn input_id(&self, name: &str) -> Result<rten::NodeId, String> {
        self.inputs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, id)| *id)
            .ok_or_else(|| format!("input '{name}' not found (have {:?})", self.inputs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()))
    }

    /// Run with one float input.
    pub fn run_f(
        &self,
        input: &str,
        shape: &[usize],
        data: &[f32],
    ) -> Result<Vec<Tensor>, String> {
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
            e.det.inputs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            vec!["x"]
        );
        assert_eq!(
            e.det.outputs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            vec!["out"]
        );
        for g in ["base", "cjk", "rtl", "world"] {
            let s = &e.rec[g];
            assert_eq!(
                s.inputs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
                vec!["x", "widths"]
            );
            let onames: Vec<&str> =
                s.outputs.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(
                onames,
                vec!["ids", "conf", "lengths", "blank", "topv", "topi"]
            );
        }
        assert_eq!(
            e.order.inputs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            vec!["feats", "mask"]
        );
        assert_eq!(
            e.orient.inputs.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
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
        let out = e
            .rec["base"]
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
        let out = e.orient.run_f("x", &[1, 1, 768, 768], &data).expect("orient run");
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
            .run(vec![(fid, ft.view().into()), (mid, mt.view().into())], &oids, None)
            .expect("order run");
        let t0 = value_to_tensor(&outs[0]).unwrap();
        assert_eq!(t0.shape()[0], 1);
    }
}
