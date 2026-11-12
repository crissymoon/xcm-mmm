//! A pre norm transformer encoder with a hand written backward pass.
//!
//! One encoder is shared by every task. The task changes only the final linear
//! head, so the trunk sees ranking, preference, contrast, label, correction and
//! retrieval rows in the same space, and a corpus that mixes them trains one
//! model rather than six. The pooled representation is the mean over the token
//! positions after the last block, normalised once more; that is the vector a
//! downstream ranker uses when it does not want a task head at all.
//!
//! Every block is `x = x + attn(ln(x))` then `x = x + ffn(ln(x))`, the ordering
//! that trains without a warmup trick. Gradients are written by hand because
//! the alternative is an autodiff dependency the rest of the crate does not
//! want; the test module checks a random sample of every parameter block
//! against a central finite difference of the loss, which is the check that
//! catches a transposed matrix or a missing residual.
//!
//! There is one sequence at a time. The shapes are small, the corpus is text,
//! and a batch of sequences would need padding and a mask in every kernel; the
//! trainer accumulates gradients over a batch instead, which is simpler and
//! gives the same update.

#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};

use crate::rng::Rng;
pub use crate::transformer::grads::Grads;
use crate::transformer::tensor::{
    self, add_bias, col_mean, col_sum, matmul, matmul_abt, matmul_atb, Matrix, NormCache, Real,
};

/// The shape of the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    /// Vocabulary size, including the control tokens.
    pub vocab: usize,
    /// Longest token sequence the model accepts.
    pub max_len: usize,
    /// Width of the residual stream.
    pub dim: usize,
    /// Attention heads. `dim` must divide evenly.
    pub heads: usize,
    /// Width of the feed forward block.
    pub ff: usize,
    /// Number of blocks.
    pub layers: usize,
    /// Number of task heads.
    pub tasks: usize,
    /// Dropout rate on the two residual branches. Zero disables it.
    pub dropout: Real,
    /// Layer normalisation epsilon.
    pub norm_eps: Real,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab: 2048,
            max_len: 64,
            dim: 64,
            heads: 4,
            ff: 128,
            layers: 2,
            tasks: 6,
            dropout: 0.1,
            norm_eps: 1e-5,
        }
    }
}

impl Config {
    /// Check the invariants the kernels rely on.
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab < 260 {
            return Err(format!("vocab {} is below the byte floor", self.vocab));
        }
        if self.dim == 0 || self.heads == 0 || self.dim % self.heads != 0 {
            return Err(format!(
                "dim {} must be a positive multiple of heads {}",
                self.dim, self.heads
            ));
        }
        if self.ff == 0 || self.layers == 0 || self.max_len == 0 || self.tasks == 0 {
            return Err("ff, layers, max_len and tasks must be positive".to_string());
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(format!("dropout {} is outside [0, 1)", self.dropout));
        }
        if self.norm_eps <= 0.0 {
            return Err("norm_eps must be positive".to_string());
        }
        Ok(())
    }

    /// Number of attention heads. Kept as a method so the kernels read evenly.
    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }
}

fn random_matrix(rows: usize, cols: usize, scale: Real, rng: &mut Rng) -> Matrix {
    let data = (0..rows * cols).map(|_| rng.normal(0.0, scale)).collect();
    Matrix::new(rows, cols, data)
}

fn ones(rows: usize, cols: usize) -> Matrix {
    Matrix::new(rows, cols, vec![1.0; rows * cols])
}

fn zeros(rows: usize, cols: usize) -> Matrix {
    Matrix::zeros(rows, cols)
}

/// One transformer block's parameters.
#[derive(Clone, Debug)]
pub struct LayerParams {
    pub ln1_g: Matrix,
    pub ln1_b: Matrix,
    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    pub wo: Matrix,
    pub bq: Matrix,
    pub bk: Matrix,
    pub bv: Matrix,
    pub bo: Matrix,
    pub ln2_g: Matrix,
    pub ln2_b: Matrix,
    pub w1: Matrix,
    pub b1: Matrix,
    pub w2: Matrix,
    pub b2: Matrix,
}

impl LayerParams {
    fn init(cfg: &Config, rng: &mut Rng, scale: Real) -> Self {
        let d = cfg.dim;
        let ff = cfg.ff;
        Self {
            ln1_g: ones(1, d),
            ln1_b: zeros(1, d),
            wq: random_matrix(d, d, scale, rng),
            wk: random_matrix(d, d, scale, rng),
            wv: random_matrix(d, d, scale, rng),
            wo: random_matrix(d, d, scale, rng),
            bq: zeros(1, d),
            bk: zeros(1, d),
            bv: zeros(1, d),
            bo: zeros(1, d),
            ln2_g: ones(1, d),
            ln2_b: zeros(1, d),
            w1: random_matrix(d, ff, scale, rng),
            b1: zeros(1, ff),
            w2: random_matrix(ff, d, scale, rng),
            b2: zeros(1, d),
        }
    }
}

/// Every trainable tensor.
#[derive(Clone, Debug)]
pub struct Params {
    pub tok: Matrix,
    pub pos: Matrix,
    pub layers: Vec<LayerParams>,
    pub lnf_g: Matrix,
    pub lnf_b: Matrix,
    /// One row per task.
    pub head_w: Matrix,
    pub head_b: Matrix,
}

impl Params {
    /// Initialise from a seed.
    pub fn init(cfg: &Config, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let scale = 0.02;
        Self {
            tok: random_matrix(cfg.vocab, cfg.dim, scale, &mut rng),
            pos: random_matrix(cfg.max_len, cfg.dim, scale, &mut rng),
            layers: (0..cfg.layers)
                .map(|_| LayerParams::init(cfg, &mut rng, scale))
                .collect(),
            lnf_g: ones(1, cfg.dim),
            lnf_b: zeros(1, cfg.dim),
            head_w: random_matrix(cfg.tasks, cfg.dim, scale, &mut rng),
            head_b: zeros(1, cfg.tasks),
        }
    }

    /// Every tensor, with its name and shape, borrowed mutably.
    ///
    /// The order is stable and the gradients use the same one, so a trainer can
    /// zip the two lists instead of matching names on every step.
    pub fn blocks_mut(&mut self) -> Vec<(String, usize, usize, &mut [Real])> {
        let Params {
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

    /// Visit every tensor in the order of [`Params::blocks_mut`].
    pub fn visit_blocks<F>(&mut self, mut f: F)
    where
        F: FnMut(&str, usize, usize, &mut [Real]),
    {
        for (name, rows, cols, data) in self.blocks_mut() {
            f(&name, rows, cols, data);
        }
    }

    /// Total number of scalars.
    pub fn count(&self) -> usize {
        let mut total = 0;
        let mut copy = self.clone();
        copy.visit_blocks(|_, rows, cols, _| total += rows * cols);
        total
    }
}

struct LayerCache {
    x_in: Matrix,
    y1: Matrix,
    norm1: Vec<NormCache>,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    /// Attention weights, `heads x len x len`, row major.
    attn: Vec<Real>,
    o: Matrix,
    z_mask: Matrix,
    x_mid: Matrix,
    y2: Matrix,
    norm2: Vec<NormCache>,
    u: Matrix,
    g: Matrix,
    f_mask: Matrix,
}

/// Everything the backward pass needs from one forward call.
pub struct Cache {
    tokens: Vec<u32>,
    layers: Vec<LayerCache>,
    pooled: Vec<Real>,
    final_norm: NormCache,
    hnorm: Vec<Real>,
    task: usize,
}

/// The encoder and its parameters.
#[derive(Clone, Debug)]
pub struct Model {
    pub config: Config,
    pub params: Params,
}

fn dropout_mask(rows: usize, cols: usize, rate: Real, rng: &mut Rng) -> Matrix {
    if rate <= 0.0 {
        return ones(rows, cols);
    }
    let keep = 1.0 - rate;
    let scale = 1.0 / keep;
    let data = (0..rows * cols)
        .map(|_| if rng.next_f64() < rate { 0.0 } else { scale })
        .collect();
    Matrix::new(rows, cols, data)
}

impl Model {
    /// Build a model from a config and a seed.
    pub fn new(config: Config, seed: u64) -> Self {
        let params = Params::init(&config, seed);
        Self { config, params }
    }

    /// Score one encoded sequence for one task.
    pub fn score(&self, tokens: &[u32], task: usize) -> Real {
        let mut rng = Rng::new(0);
        self.forward(tokens, task, false, &mut rng).0
    }

    /// Forward pass with dropout drawn from a per-example seed.
    ///
    /// A batch can be spread over any number of threads because the mask of an
    /// example depends only on the seed the caller hands in, never on how many
    /// examples ran before it. That is what makes a parallel gradient buffer
    /// reproduce the serial one exactly.
    pub fn forward_seeded(&self, tokens: &[u32], task: usize, seed: u64) -> (Real, Cache) {
        let mut rng = Rng::new(seed);
        self.forward(tokens, task, true, &mut rng)
    }

    /// Forward pass, returning the score and the cache for the backward pass.
    pub fn forward(
        &self,
        tokens: &[u32],
        task: usize,
        train: bool,
        rng: &mut Rng,
    ) -> (Real, Cache) {
        let cfg = &self.config;
        assert!(!tokens.is_empty(), "a sequence needs at least one token");
        let task = task.min(cfg.tasks - 1);
        let len = tokens.len().min(cfg.max_len);
        let dim = cfg.dim;
        let heads = cfg.heads;
        let dh = cfg.head_dim();
        let scale = 1.0 / (dh as Real).sqrt();

        let mut x = Matrix::zeros(len, dim);
        for i in 0..len {
            let id = tokens[i].min(cfg.vocab as u32 - 1) as usize;
            for j in 0..dim {
                x.set(i, j, self.params.tok.get(id, j) + self.params.pos.get(i, j));
            }
        }

        let mut layers = Vec::with_capacity(cfg.layers);
        for layer_index in 0..cfg.layers {
            let p = &self.params.layers[layer_index];
            let x_in = x.clone();

            let mut y1 = Matrix::zeros(len, dim);
            let mut norm1 = Vec::with_capacity(len);
            for i in 0..len {
                let cache = tensor::layer_norm(
                    x_in.row(i),
                    p.ln1_g.row(0),
                    p.ln1_b.row(0),
                    cfg.norm_eps,
                    y1.row_mut(i),
                );
                norm1.push(cache);
            }

            let mut q = matmul(&y1, &p.wq);
            let mut k = matmul(&y1, &p.wk);
            let mut v = matmul(&y1, &p.wv);
            add_bias(&mut q, p.bq.row(0));
            add_bias(&mut k, p.bk.row(0));
            add_bias(&mut v, p.bv.row(0));

            let mut attn = vec![0.0; heads * len * len];
            let mut o = Matrix::zeros(len, dim);
            for h in 0..heads {
                let base = h * dh;
                for i in 0..len {
                    let mut row = vec![0.0; len];
                    for j in 0..len {
                        let mut dot = 0.0;
                        for t in 0..dh {
                            dot += q.get(i, base + t) * k.get(j, base + t);
                        }
                        row[j] = dot * scale;
                    }
                    tensor::softmax_in_place(&mut row);
                    let offset = h * len * len + i * len;
                    for j in 0..len {
                        attn[offset + j] = row[j];
                        if row[j] == 0.0 {
                            continue;
                        }
                        for t in 0..dh {
                            let current = o.get(i, base + t);
                            o.set(i, base + t, current + row[j] * v.get(j, base + t));
                        }
                    }
                }
            }

            let mut z = matmul(&o, &p.wo);
            add_bias(&mut z, p.bo.row(0));
            let z_mask = if train {
                dropout_mask(len, dim, cfg.dropout, rng)
            } else {
                ones(len, dim)
            };
            let mut x_mid = x_in.clone();
            for i in 0..len {
                for j in 0..dim {
                    let value = z.get(i, j) * z_mask.get(i, j);
                    x_mid.set(i, j, x_in.get(i, j) + value);
                }
            }

            let mut y2 = Matrix::zeros(len, dim);
            let mut norm2 = Vec::with_capacity(len);
            for i in 0..len {
                let cache = tensor::layer_norm(
                    x_mid.row(i),
                    p.ln2_g.row(0),
                    p.ln2_b.row(0),
                    cfg.norm_eps,
                    y2.row_mut(i),
                );
                norm2.push(cache);
            }

            let mut u = matmul(&y2, &p.w1);
            add_bias(&mut u, p.b1.row(0));
            let mut g = Matrix::zeros(len, cfg.ff);
            for (out, input) in g.data.iter_mut().zip(&u.data) {
                *out = tensor::gelu(*input);
            }
            let mut f = matmul(&g, &p.w2);
            add_bias(&mut f, p.b2.row(0));
            let f_mask = if train {
                dropout_mask(len, dim, cfg.dropout, rng)
            } else {
                ones(len, dim)
            };

            x = x_mid.clone();
            for i in 0..len {
                for j in 0..dim {
                    let value = f.get(i, j) * f_mask.get(i, j);
                    x.set(i, j, x_mid.get(i, j) + value);
                }
            }

            layers.push(LayerCache {
                x_in,
                y1,
                norm1,
                q,
                k,
                v,
                attn,
                o,
                z_mask,
                x_mid,
                y2,
                norm2,
                u,
                g,
                f_mask,
            });
        }

        let pooled = col_mean(&x);
        let mut hnorm = vec![0.0; dim];
        let final_norm = tensor::layer_norm(
            &pooled,
            self.params.lnf_g.row(0),
            self.params.lnf_b.row(0),
            cfg.norm_eps,
            &mut hnorm,
        );
        let mut score = self.params.head_b.get(0, task);
        for j in 0..dim {
            score += self.params.head_w.get(task, j) * hnorm[j];
        }

        (
            score,
            Cache {
                tokens: tokens[..len].to_vec(),
                layers,
                pooled,
                final_norm,
                hnorm,
                task,
            },
        )
    }

    /// Backward pass. `dscore` is the derivative of the loss to the score.
    pub fn backward(&self, cache: &Cache, dscore: Real, grads: &mut Grads) {
        let cfg = &self.config;
        let dim = cfg.dim;
        let task = cache.task;

        for j in 0..dim {
            grads
                .head_w
                .set(task, j, grads.head_w.get(task, j) + cache.hnorm[j] * dscore);
        }
        grads
            .head_b
            .set(0, task, grads.head_b.get(0, task) + dscore);

        let mut dhnorm = vec![0.0; dim];
        for j in 0..dim {
            dhnorm[j] = self.params.head_w.get(task, j) * dscore;
        }
        let mut dpooled = vec![0.0; dim];
        tensor::layer_norm_backward(
            &dhnorm,
            &cache.pooled,
            &cache.final_norm,
            self.params.lnf_g.row(0),
            &mut dpooled,
            grads.lnf_g.row_mut(0),
            grads.lnf_b.row_mut(0),
        );

        let len = cache.tokens.len();
        let mut dx = Matrix::zeros(len, dim);
        let share = 1.0 / len as Real;
        for i in 0..len {
            for j in 0..dim {
                dx.set(i, j, dpooled[j] * share);
            }
        }

        for layer_index in (0..cfg.layers).rev() {
            dx = self.backward_layer(layer_index, &cache.layers[layer_index], &dx, grads);
        }

        for (i, token) in cache.tokens.iter().enumerate() {
            let id = *token as usize;
            for j in 0..dim {
                let value = dx.get(i, j);
                grads.tok.set(id, j, grads.tok.get(id, j) + value);
                grads.pos.set(i, j, grads.pos.get(i, j) + value);
            }
        }
    }

    fn backward_layer(
        &self,
        index: usize,
        cache: &LayerCache,
        dout: &Matrix,
        grads: &mut Grads,
    ) -> Matrix {
        let cfg = &self.config;
        let p = &self.params.layers[index];
        let g = &mut grads.layers[index];
        let len = cache.x_in.rows;
        let dim = cfg.dim;
        let ff = cfg.ff;
        let heads = cfg.heads;
        let dh = cfg.head_dim();
        let scale = 1.0 / (dh as Real).sqrt();

        // Feed forward branch.
        let mut df = dout.clone();
        for (value, mask) in df.data.iter_mut().zip(&cache.f_mask.data) {
            *value *= mask;
        }
        let dw2 = matmul_atb(&cache.g, &df);
        accumulate(&mut g.w2, &dw2);
        let db2 = col_sum(&df);
        accumulate_row(g.b2.row_mut(0), &db2);
        let dg = matmul_abt(&df, &p.w2);
        let mut du = Matrix::zeros(len, ff);
        for i in 0..du.data.len() {
            du.data[i] = dg.data[i] * tensor::gelu_grad(cache.u.data[i]);
        }
        let dw1 = matmul_atb(&cache.y2, &du);
        accumulate(&mut g.w1, &dw1);
        let db1 = col_sum(&du);
        accumulate_row(g.b1.row_mut(0), &db1);
        let dy2 = matmul_abt(&du, &p.w1);
        let mut dx_ln2 = Matrix::zeros(len, dim);
        for i in 0..len {
            tensor::layer_norm_backward(
                dy2.row(i),
                cache.x_mid.row(i),
                &cache.norm2[i],
                p.ln2_g.row(0),
                dx_ln2.row_mut(i),
                g.ln2_g.row_mut(0),
                g.ln2_b.row_mut(0),
            );
        }
        let mut dx_mid = dout.clone();
        accumulate(&mut dx_mid, &dx_ln2);

        // Attention branch.
        let mut dz = dx_mid.clone();
        for (value, mask) in dz.data.iter_mut().zip(&cache.z_mask.data) {
            *value *= mask;
        }
        let dwo = matmul_atb(&cache.o, &dz);
        accumulate(&mut g.wo, &dwo);
        let dbo = col_sum(&dz);
        accumulate_row(g.bo.row_mut(0), &dbo);
        let d_o = matmul_abt(&dz, &p.wo);

        let mut dq = Matrix::zeros(len, dim);
        let mut dk = Matrix::zeros(len, dim);
        let mut dv = Matrix::zeros(len, dim);
        let mut da = vec![0.0; len];
        let mut ds = vec![0.0; len];
        for h in 0..heads {
            let base = h * dh;
            for i in 0..len {
                let offset = h * len * len + i * len;
                let mut weighted = 0.0;
                for j in 0..len {
                    let mut dot = 0.0;
                    for t in 0..dh {
                        dot += d_o.get(i, base + t) * cache.v.get(j, base + t);
                    }
                    da[j] = dot;
                    weighted += dot * cache.attn[offset + j];
                }
                for j in 0..len {
                    let weight = cache.attn[offset + j];
                    ds[j] = weight * (da[j] - weighted);
                }
                for j in 0..len {
                    let gradient = ds[j] * scale;
                    if gradient == 0.0 {
                        continue;
                    }
                    for t in 0..dh {
                        let qv = cache.q.get(i, base + t);
                        let kv = cache.k.get(j, base + t);
                        dq.set(i, base + t, dq.get(i, base + t) + gradient * kv);
                        dk.set(j, base + t, dk.get(j, base + t) + gradient * qv);
                    }
                }
                for j in 0..len {
                    let weight = cache.attn[offset + j];
                    if weight == 0.0 {
                        continue;
                    }
                    for t in 0..dh {
                        let value = d_o.get(i, base + t);
                        dv.set(j, base + t, dv.get(j, base + t) + weight * value);
                    }
                }
            }
        }

        let dwq = matmul_atb(&cache.y1, &dq);
        accumulate(&mut g.wq, &dwq);
        accumulate_row(g.bq.row_mut(0), &col_sum(&dq));
        let dwk = matmul_atb(&cache.y1, &dk);
        accumulate(&mut g.wk, &dwk);
        accumulate_row(g.bk.row_mut(0), &col_sum(&dk));
        let dwv = matmul_atb(&cache.y1, &dv);
        accumulate(&mut g.wv, &dwv);
        accumulate_row(g.bv.row_mut(0), &col_sum(&dv));

        let mut dy1 = matmul_abt(&dq, &p.wq);
        accumulate(&mut dy1, &matmul_abt(&dk, &p.wk));
        accumulate(&mut dy1, &matmul_abt(&dv, &p.wv));

        let mut dx_ln1 = Matrix::zeros(len, dim);
        for i in 0..len {
            tensor::layer_norm_backward(
                dy1.row(i),
                cache.x_in.row(i),
                &cache.norm1[i],
                p.ln1_g.row(0),
                dx_ln1.row_mut(i),
                g.ln1_g.row_mut(0),
                g.ln1_b.row_mut(0),
            );
        }
        let mut dx_in = dx_mid.clone();
        accumulate(&mut dx_in, &dx_ln1);
        dx_in
    }
}

fn accumulate(target: &mut Matrix, source: &Matrix) {
    debug_assert_eq!(target.rows, source.rows);
    debug_assert_eq!(target.cols, source.cols);
    for (a, b) in target.data.iter_mut().zip(&source.data) {
        *a += b;
    }
}

fn accumulate_row(target: &mut [Real], source: &[Real]) {
    for (a, b) in target.iter_mut().zip(source) {
        *a += b;
    }
}

#[cfg(test)]
mod tests;
