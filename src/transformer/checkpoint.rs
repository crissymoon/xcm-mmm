//! Mid-run state, so an interrupted training can be resumed exactly.
//!
//! A checkpoint holds everything a run needs to continue rather than restart:
//! the current weights, the best weights seen, the two Adam moments, the step
//! counter and the epoch cursor. It is stored at full precision as JSON, which
//! makes a resumed run bit for bit identical to the uninterrupted one instead of
//! nearly identical, and leaves the file readable when something looks wrong.
//!
//! This is not the model file. [`crate::transformer::store`] writes a compact
//! `f32` model for inference and keeps no optimizer state; a checkpoint is
//! transient, roughly four times the size, and is deleted once training ends.
//!
//! The tokenizer travels in the checkpoint because retraining it is the part of
//! a re-run that costs the most on a large corpus, and a resumed run must encode
//! its rows with the vocabulary the weights were trained against.

use std::io;

use serde::{Deserialize, Serialize};

use crate::transformer::model::{Config, Params};
use crate::transformer::tensor::Real;
use crate::transformer::tokenizer::TokenizerFile;

/// The format tag, bumped when the layout changes.
pub const FORMAT: &str = "xcm-mmm-transformer-checkpoint-v1";

/// Full precision floats, written as sixteen hex digits of their bit pattern.
///
/// A resumed run must continue from the exact weights the previous leg stopped
/// at, and decimal JSON is not good enough for that: `serde_json` parses some
/// `f64` values one unit in the last place away from what it wrote, which is
/// enough for a few hundred Adam steps to drift visibly. The bit pattern cannot
/// drift. It costs about the same as the decimal form and is read back by this
/// module alone.
mod exact {
    use serde::de::Error as _;
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serializer};

    use super::Real;

    pub fn serialize<S>(values: &[Real], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(values.len()))?;
        for value in values {
            seq.serialize_element(&format!("{:016x}", value.to_bits()))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Real>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let words: Vec<String> = Vec::deserialize(deserializer)?;
        words
            .iter()
            .map(|word| {
                u64::from_str_radix(word, 16)
                    .map(Real::from_bits)
                    .map_err(|_| D::Error::custom(format!("not a float bit pattern: {word:?}")))
            })
            .collect()
    }
}

/// One named tensor at full precision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    #[serde(with = "exact")]
    pub data: Vec<Real>,
}

/// The hyperparameters of the run that wrote the checkpoint.
///
/// They are recorded so a resume can say which settings produced the weights,
/// and so a shape mismatch is caught with a sentence rather than a panic.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunSettings {
    pub epochs: usize,
    pub batch: usize,
    pub lr: Real,
    pub weight_decay: Real,
    pub warmup: Real,
    pub clip: Real,
    pub dropout: Real,
    pub seed: u64,
    pub patience: usize,
    pub min_count: u64,
    /// Sampling weight per task, in `Task::ALL` order.
    pub task_weights: [Real; 6],
}

/// The whole checkpoint file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointFile {
    pub format: String,
    pub config: Config,
    pub tokenizer: TokenizerFile,
    /// Epochs finished, which is also the index of the next epoch to run.
    pub epoch: usize,
    /// Optimizer steps taken across every finished epoch.
    pub step: usize,
    /// Epochs since the best validation score improved.
    pub stale: usize,
    /// The epoch that produced `best`, one based. Zero means no epoch yet.
    pub best_epoch: usize,
    /// Training loss of the last finished epoch.
    pub last_loss: Real,
    pub run: RunSettings,
    pub adam_step: usize,
    pub weights: Vec<Block>,
    pub best: Vec<Block>,
    pub first_moment: Vec<Block>,
    pub second_moment: Vec<Block>,
}

/// Every parameter block, in the order [`Params::blocks_mut`] visits them.
pub fn blocks(params: &mut Params) -> Vec<Block> {
    params
        .blocks_mut()
        .into_iter()
        .map(|(name, rows, cols, data)| Block {
            name,
            rows,
            cols,
            data: data.to_vec(),
        })
        .collect()
}

/// Copy stored blocks back into parameters, checking every name and shape.
pub fn fill(params: &mut Params, stored: &[Block]) -> Result<(), String> {
    let mut target = params.blocks_mut();
    if target.len() != stored.len() {
        return Err(format!(
            "checkpoint has {} tensors but the model has {}",
            stored.len(),
            target.len()
        ));
    }
    for (index, (name, rows, cols, data)) in target.iter_mut().enumerate() {
        let block = &stored[index];
        if block.name != *name
            || block.rows != *rows
            || block.cols != *cols
            || block.data.len() != data.len()
        {
            return Err(format!(
                "tensor {index} does not match: checkpoint has {} {}x{}, model expects {} {}x{}",
                block.name, block.rows, block.cols, name, rows, cols
            ));
        }
        data.copy_from_slice(&block.data);
    }
    Ok(())
}

/// Pair flat optimizer moments with the shapes of the parameters they belong to.
///
/// Adam keeps its moments as one flat vector per block, without names. Writing
/// them with the parameter names and shapes means a resume against a different
/// model is refused instead of quietly pairing the wrong vectors.
pub fn moment_blocks(params: &mut Params, moments: &[Vec<Real>]) -> Result<Vec<Block>, String> {
    let target = params.blocks_mut();
    if target.len() != moments.len() {
        return Err(format!(
            "{} moment vectors for {} tensors",
            moments.len(),
            target.len()
        ));
    }
    let mut out = Vec::with_capacity(target.len());
    for ((name, rows, cols, data), values) in target.iter().zip(moments) {
        if values.len() != data.len() {
            return Err(format!(
                "moment for {name} has {} values, the tensor has {}",
                values.len(),
                data.len()
            ));
        }
        out.push(Block {
            name: name.to_string(),
            rows: *rows,
            cols: *cols,
            data: values.clone(),
        });
    }
    Ok(out)
}

/// The inverse of [`moment_blocks`]: one flat vector per parameter block.
pub fn moments_from(params: &mut Params, stored: &[Block]) -> Result<Vec<Vec<Real>>, String> {
    let mut target = params.blocks_mut();
    if target.len() != stored.len() {
        return Err(format!(
            "checkpoint has {} moment tensors but the model has {}",
            stored.len(),
            target.len()
        ));
    }
    let mut out = Vec::with_capacity(stored.len());
    for ((name, _, _, data), block) in target.iter_mut().zip(stored) {
        if block.name != *name || block.data.len() != data.len() {
            return Err(format!(
                "moment {name} does not match the checkpoint layout"
            ));
        }
        out.push(block.data.clone());
    }
    Ok(out)
}

/// Write a checkpoint.
pub fn save(path: &str, file: &CheckpointFile) -> io::Result<u64> {
    let body = serde_json::to_string(file)
        .map_err(|e| io::Error::other(format!("checkpoint encode: {e}")))?;
    std::fs::write(path, &body)?;
    Ok(body.len() as u64)
}

/// Read a checkpoint, refusing anything written by a different layout.
pub fn load(path: &str) -> Result<CheckpointFile, String> {
    let body = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let file: CheckpointFile =
        serde_json::from_str(&body).map_err(|e| format!("parse {path}: {e}"))?;
    if file.format != FORMAT {
        return Err(format!(
            "unsupported checkpoint format {:?}, expected {:?}",
            file.format, FORMAT
        ));
    }
    file.config.validate()?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::model::Model;

    fn tiny() -> Config {
        Config {
            vocab: 260,
            max_len: 8,
            dim: 8,
            heads: 2,
            ff: 12,
            layers: 1,
            tasks: 2,
            dropout: 0.0,
            norm_eps: 1e-5,
        }
    }

    #[test]
    fn weights_survive_a_round_trip_exactly() {
        let mut model = Model::new(tiny(), 9);
        let before = blocks(&mut model.params);
        let mut restored = Model::new(tiny(), 0);
        fill(&mut restored.params, &before).unwrap();
        let after = blocks(&mut restored.params);
        for (left, right) in before.iter().zip(&after) {
            assert_eq!(left.name, right.name);
            assert_eq!(left.data, right.data);
        }
        assert_eq!(
            model.score(&[4, 20, 21], 0),
            restored.score(&[4, 20, 21], 0)
        );
    }

    #[test]
    fn the_file_keeps_every_bit() {
        // The first value is one that decimal JSON parses one ulp away.
        let values = vec![
            -0.010746284872993877,
            0.0,
            -0.0,
            Real::MIN_POSITIVE,
            1.0 / 3.0,
            f64::MAX,
        ];
        let stored = Block {
            name: "text".to_string(),
            rows: 1,
            cols: values.len(),
            data: values.clone(),
        };
        let body = serde_json::to_string(&stored).unwrap();
        let back: Block = serde_json::from_str(&body).unwrap();
        let before: Vec<u64> = values.iter().map(|v| v.to_bits()).collect();
        let after: Vec<u64> = back.data.iter().map(|v| v.to_bits()).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn a_shape_mismatch_is_refused() {
        let mut model = Model::new(tiny(), 9);
        let mut stored = blocks(&mut model.params);
        stored[0].data.pop();
        let mut other = Model::new(tiny(), 0);
        assert!(fill(&mut other.params, &stored).is_err());
    }

    #[test]
    fn moments_keep_their_block_shapes() {
        let mut model = Model::new(tiny(), 4);
        let moments: Vec<Vec<Real>> = model
            .params
            .blocks_mut()
            .iter()
            .map(|(_, rows, cols, _)| vec![0.25; rows * cols])
            .collect();
        let stored = moment_blocks(&mut model.params, &moments).unwrap();
        let back = moments_from(&mut model.params, &stored).unwrap();
        assert_eq!(moments, back);
    }

    #[test]
    fn a_file_round_trip_restores_the_cursor() {
        let mut model = Model::new(tiny(), 3);
        let zeros: Vec<Vec<Real>> = model
            .params
            .blocks_mut()
            .iter()
            .map(|(_, rows, cols, _)| vec![0.0; rows * cols])
            .collect();
        let file = CheckpointFile {
            format: FORMAT.to_string(),
            config: tiny(),
            tokenizer: crate::transformer::tokenizer::Tokenizer::bytes_only().to_file(),
            epoch: 2,
            step: 17,
            stale: 1,
            best_epoch: 1,
            last_loss: 0.5,
            run: RunSettings {
                epochs: 3,
                batch: 4,
                lr: 0.01,
                weight_decay: 0.0,
                warmup: 0.1,
                clip: 1.0,
                dropout: 0.0,
                seed: 3,
                patience: 2,
                min_count: 2,
                task_weights: [1.0; 6],
            },
            adam_step: 17,
            weights: blocks(&mut model.params),
            best: blocks(&mut model.params),
            first_moment: moment_blocks(&mut model.params, &zeros).unwrap(),
            second_moment: moment_blocks(&mut model.params, &zeros).unwrap(),
        };
        let path =
            std::env::temp_dir().join(format!("xcm-mmm-checkpoint-{}.json", std::process::id()));
        let path = path.to_string_lossy().to_string();
        save(&path, &file).unwrap();
        let back = load(&path).unwrap();
        assert_eq!(back.epoch, 2);
        assert_eq!(back.step, 17);
        assert_eq!(back.run.batch, 4);
        assert_eq!(back.weights.len(), file.weights.len());
        std::fs::remove_file(&path).unwrap();
    }
}
