//! One model file: tokenizer, config, weights and the run metadata.
//!
//! The weights are written as `f32` in a named block per tensor, in the same
//! order [`crate::transformer::model::Params::blocks_mut`] visits them. Loading
//! checks the name and the shape of every block rather than trusting the order,
//! so a file written by a different build fails with a sentence instead of
//! silently loading transposed weights. The metadata is a free form JSON value:
//! the trainer puts the report there and nothing in the loader reads it.

use std::io;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::transformer::model::{Config, Model};
use crate::transformer::tokenizer::{Tokenizer, TokenizerFile};

/// The format tag, bumped when the layout changes.
pub const FORMAT: &str = "xcm-mmm-transformer-v1";

/// One named tensor, stored at reduced precision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

/// The whole file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Saved {
    pub format: String,
    pub config: Config,
    pub tokenizer: TokenizerFile,
    pub blocks: Vec<Block>,
    #[serde(default)]
    pub meta: Value,
}

/// Write a model and its tokenizer to one JSON file.
pub fn save(model: &Model, tokenizer: &Tokenizer, path: &str, meta: Value) -> io::Result<u64> {
    let mut params = model.params.clone();
    let blocks = params
        .blocks_mut()
        .into_iter()
        .map(|(name, rows, cols, data)| Block {
            name,
            rows,
            cols,
            data: data.iter().map(|value| *value as f32).collect(),
        })
        .collect();
    let saved = Saved {
        format: FORMAT.to_string(),
        config: model.config.clone(),
        tokenizer: tokenizer.to_file(),
        blocks,
        meta,
    };
    let body = serde_json::to_string(&saved)
        .map_err(|e| io::Error::other(format!("model encode: {e}")))?;
    std::fs::write(path, &body)?;
    Ok(body.len() as u64)
}

/// Read a model, its tokenizer and its metadata.
pub fn load(path: &str) -> Result<(Model, Tokenizer, Value), String> {
    let body = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let saved: Saved = serde_json::from_str(&body).map_err(|e| format!("parse {path}: {e}"))?;
    if saved.format != FORMAT {
        return Err(format!(
            "unsupported model format {:?}, expected {:?}",
            saved.format, FORMAT
        ));
    }
    saved.config.validate()?;
    let tokenizer = Tokenizer::from_file(&saved.tokenizer)?;
    if tokenizer.vocab_size() != saved.config.vocab {
        return Err(format!(
            "tokenizer has {} tokens but the config expects {}",
            tokenizer.vocab_size(),
            saved.config.vocab
        ));
    }
    let mut model = Model::new(saved.config.clone(), 0);
    let mut blocks = model.params.blocks_mut();
    if blocks.len() != saved.blocks.len() {
        return Err(format!(
            "model has {} tensors but the file has {}",
            blocks.len(),
            saved.blocks.len()
        ));
    }
    for (index, block) in blocks.iter_mut().enumerate() {
        let stored = &saved.blocks[index];
        if stored.name != block.0
            || stored.rows != block.1
            || stored.cols != block.2
            || stored.data.len() != block.3.len()
        {
            return Err(format!(
                "tensor {} does not match: file has {} {}x{}, model expects {} {}x{}",
                index, stored.name, stored.rows, stored.cols, block.0, block.1, block.2
            ));
        }
        for (target, value) in block.3.iter_mut().zip(&stored.data) {
            *target = *value as f64;
        }
    }
    Ok((model, tokenizer, saved.meta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_restores_the_same_scores() {
        let cfg = Config {
            vocab: 260,
            max_len: 8,
            dim: 8,
            heads: 2,
            ff: 12,
            layers: 1,
            tasks: 2,
            dropout: 0.0,
            norm_eps: 1e-5,
        };
        let model = Model::new(cfg, 5);
        let tokenizer = Tokenizer::bytes_only();
        let path = std::env::temp_dir().join(format!("xcm-mmm-model-{}.json", std::process::id()));
        let path = path.to_string_lossy().to_string();
        save(
            &model,
            &tokenizer,
            &path,
            serde_json::json!({"note": "test"}),
        )
        .unwrap();
        let (back, back_tokens, meta) = load(&path).unwrap();
        let tokens = vec![4u32, 20, 21];
        let before = model.score(&tokens, 0);
        let after = back.score(&tokens, 0);
        assert!((before - after).abs() < 1e-4);
        assert_eq!(back_tokens.vocab_size(), tokenizer.vocab_size());
        assert_eq!(meta["note"], "test");
        std::fs::remove_file(&path).unwrap();
    }
}
