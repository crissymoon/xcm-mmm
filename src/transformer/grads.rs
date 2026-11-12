//! The gradient buffers, one per trainable tensor.
//!
//! A buffer has the same layout as [`Params`], so a backward pass can write a
//! block at a time and the optimizer can walk the two together. Each example of
//! a batch writes its own buffer and the buffers are added in batch order, which
//! is what makes a parallel step reproduce the serial one.

use crate::transformer::model::{LayerParams, Params};
use crate::transformer::tensor::{Matrix, Real};

/// Gradients of every trainable tensor, same layout as [`Params`].
#[derive(Clone, Debug)]
pub struct Grads {
    pub tok: Matrix,
    pub pos: Matrix,
    pub layers: Vec<LayerParams>,
    pub lnf_g: Matrix,
    pub lnf_b: Matrix,
    pub head_w: Matrix,
    pub head_b: Matrix,
}

impl Grads {
    /// A zero gradient buffer shaped like the parameters.
    pub fn zeros(params: &Params) -> Self {
        let zero_like = |m: &Matrix| Matrix::zeros(m.rows, m.cols);
        let layer = |l: &LayerParams| LayerParams {
            ln1_g: zero_like(&l.ln1_g),
            ln1_b: zero_like(&l.ln1_b),
            wq: zero_like(&l.wq),
            wk: zero_like(&l.wk),
            wv: zero_like(&l.wv),
            wo: zero_like(&l.wo),
            bq: zero_like(&l.bq),
            bk: zero_like(&l.bk),
            bv: zero_like(&l.bv),
            bo: zero_like(&l.bo),
            ln2_g: zero_like(&l.ln2_g),
            ln2_b: zero_like(&l.ln2_b),
            w1: zero_like(&l.w1),
            b1: zero_like(&l.b1),
            w2: zero_like(&l.w2),
            b2: zero_like(&l.b2),
        };
        Self {
            tok: zero_like(&params.tok),
            pos: zero_like(&params.pos),
            layers: params.layers.iter().map(layer).collect(),
            lnf_g: zero_like(&params.lnf_g),
            lnf_b: zero_like(&params.lnf_b),
            head_w: zero_like(&params.head_w),
            head_b: zero_like(&params.head_b),
        }
    }

    /// Every gradient block, in the same order as [`Params::blocks_mut`].
    pub fn blocks_mut(&mut self) -> Vec<(String, usize, usize, &mut [Real])> {
        let Grads {
            tok,
            pos,
            layers,
            lnf_g,
            lnf_b,
            head_w,
            head_b,
        } = self;
        let mut out: Vec<(String, usize, usize, &mut [Real])> = Vec::new();
        out.push((
            "tok".to_string(),
            tok.rows,
            tok.cols,
            tok.data.as_mut_slice(),
        ));
        out.push((
            "pos".to_string(),
            pos.rows,
            pos.cols,
            pos.data.as_mut_slice(),
        ));
        for (index, layer) in layers.iter_mut().enumerate() {
            let name = |suffix: &str| format!("layers.{index}.{suffix}");
            out.push((
                name("ln1_g"),
                1,
                layer.ln1_g.cols,
                layer.ln1_g.data.as_mut_slice(),
            ));
            out.push((
                name("ln1_b"),
                1,
                layer.ln1_b.cols,
                layer.ln1_b.data.as_mut_slice(),
            ));
            out.push((
                name("wq"),
                layer.wq.rows,
                layer.wq.cols,
                layer.wq.data.as_mut_slice(),
            ));
            out.push((
                name("wk"),
                layer.wk.rows,
                layer.wk.cols,
                layer.wk.data.as_mut_slice(),
            ));
            out.push((
                name("wv"),
                layer.wv.rows,
                layer.wv.cols,
                layer.wv.data.as_mut_slice(),
            ));
            out.push((
                name("wo"),
                layer.wo.rows,
                layer.wo.cols,
                layer.wo.data.as_mut_slice(),
            ));
            out.push((name("bq"), 1, layer.bq.cols, layer.bq.data.as_mut_slice()));
            out.push((name("bk"), 1, layer.bk.cols, layer.bk.data.as_mut_slice()));
            out.push((name("bv"), 1, layer.bv.cols, layer.bv.data.as_mut_slice()));
            out.push((name("bo"), 1, layer.bo.cols, layer.bo.data.as_mut_slice()));
            out.push((
                name("ln2_g"),
                1,
                layer.ln2_g.cols,
                layer.ln2_g.data.as_mut_slice(),
            ));
            out.push((
                name("ln2_b"),
                1,
                layer.ln2_b.cols,
                layer.ln2_b.data.as_mut_slice(),
            ));
            out.push((
                name("w1"),
                layer.w1.rows,
                layer.w1.cols,
                layer.w1.data.as_mut_slice(),
            ));
            out.push((name("b1"), 1, layer.b1.cols, layer.b1.data.as_mut_slice()));
            out.push((
                name("w2"),
                layer.w2.rows,
                layer.w2.cols,
                layer.w2.data.as_mut_slice(),
            ));
            out.push((name("b2"), 1, layer.b2.cols, layer.b2.data.as_mut_slice()));
        }
        out.push((
            "lnf_g".to_string(),
            1,
            lnf_g.cols,
            lnf_g.data.as_mut_slice(),
        ));
        out.push((
            "lnf_b".to_string(),
            1,
            lnf_b.cols,
            lnf_b.data.as_mut_slice(),
        ));
        out.push((
            "head_w".to_string(),
            head_w.rows,
            head_w.cols,
            head_w.data.as_mut_slice(),
        ));
        out.push((
            "head_b".to_string(),
            head_b.rows,
            head_b.cols,
            head_b.data.as_mut_slice(),
        ));
        out
    }

    /// Visit every gradient in the order of [`Params::blocks_mut`].
    pub fn visit_blocks<F>(&mut self, mut f: F)
    where
        F: FnMut(&str, usize, usize, &mut [Real]),
    {
        for (name, rows, cols, data) in self.blocks_mut() {
            f(&name, rows, cols, data);
        }
    }

    /// Add another gradient buffer into this one, block by block.
    ///
    /// The source is taken mutably so the two block lists can be walked in
    /// lockstep without copying a whole model-sized buffer per accumulation.
    /// Both buffers come from [`Grads::zeros`] on the same parameters, so the
    /// lengths always agree; a mismatch means the caller mixed two models.
    pub fn accumulate(&mut self, source: &mut Grads) {
        let mut target_blocks = self.blocks_mut();
        let mut source_blocks = source.blocks_mut();
        assert_eq!(
            target_blocks.len(),
            source_blocks.len(),
            "gradient buffers from different models"
        );
        for ((name, rows, cols, into), (theirs, their_rows, their_cols, from)) in
            target_blocks.iter_mut().zip(source_blocks.iter_mut())
        {
            assert!(
                *name == *theirs && *rows == *their_rows && *cols == *their_cols,
                "gradient block shape mismatch"
            );
            for (slot, value) in into.iter_mut().zip(from.iter()) {
                *slot += *value;
            }
        }
    }

    /// Multiply every gradient by a scalar.
    pub fn scale(&mut self, factor: Real) {
        self.visit_blocks(|_, _, _, data| {
            for value in data.iter_mut() {
                *value *= factor;
            }
        });
    }

    /// The Euclidean norm over every block.
    pub fn global_norm(&self) -> Real {
        let mut total = 0.0;
        let mut copy = self.clone();
        copy.visit_blocks(|_, _, _, data| {
            for value in data.iter() {
                total += value * value;
            }
        });
        total.sqrt()
    }

    /// Scale the buffer down so its norm is at most `limit`.
    pub fn clip(&mut self, limit: Real) -> Real {
        let norm = self.global_norm();
        if norm > limit && norm > 0.0 {
            self.scale(limit / norm);
        }
        norm
    }
}
