//! The model checks: shapes, determinism and the finite difference test
//! over a random sample of every parameter block.

use super::*;
use approx::assert_relative_eq;

fn tiny() -> Config {
    Config {
        vocab: 280,
        max_len: 10,
        dim: 8,
        heads: 2,
        ff: 12,
        layers: 1,
        tasks: 2,
        dropout: 0.0,
        norm_eps: 1e-5,
    }
}

fn block_lengths(params: &Params) -> Vec<usize> {
    let mut lengths = Vec::new();
    let mut copy = params.clone();
    copy.visit_blocks(|_, rows, cols, _| lengths.push(rows * cols));
    lengths
}

fn set_at(params: &mut Params, block: usize, offset: usize, value: Real) {
    let mut index = 0;
    params.visit_blocks(|_, _, _, data| {
        if index == block {
            data[offset] = value;
        }
        index += 1;
    });
}

fn grad_at(grads: &mut Grads, block: usize, offset: usize) -> Real {
    let mut value = 0.0;
    let mut index = 0;
    grads.visit_blocks(|_, _, _, data| {
        if index == block {
            value = data[offset];
        }
        index += 1;
    });
    value
}

#[test]
fn forward_is_finite_and_deterministic() {
    let model = Model::new(tiny(), 4);
    let tokens = vec![5u32, 200, 17, 40];
    let a = model.score(&tokens, 0);
    let b = model.score(&tokens, 0);
    assert!(a.is_finite());
    assert_eq!(a, b);
}

#[test]
fn params_and_grads_share_their_layout() {
    let cfg = tiny();
    let params = Params::init(&cfg, 1);
    let grads = Grads::zeros(&params);
    let mut params_copy = params.clone();
    let mut grads_copy = grads.clone();
    let mut a = Vec::new();
    let mut b = Vec::new();
    params_copy.visit_blocks(|name, rows, cols, _| a.push((name.to_string(), rows, cols)));
    grads_copy.visit_blocks(|name, rows, cols, _| b.push((name.to_string(), rows, cols)));
    assert_eq!(a, b);
    assert!(params.count() > 0);
}

#[test]
fn gradient_matches_finite_difference_on_sampled_parameters() {
    let cfg = tiny();
    let model = Model::new(cfg, 7);
    let tokens: Vec<u32> = vec![5, 200, 17, 40, 61];
    let task = 1;
    let mut rng = Rng::new(3);
    let (score, cache) = model.forward(&tokens, task, false, &mut rng);
    let mut grads = Grads::zeros(&model.params);
    // Loss is 0.5 * score^2, so the score gradient is the score itself.
    model.backward(&cache, score, &mut grads);

    let lengths = block_lengths(&model.params);
    let total: usize = lengths.iter().sum();
    let mut samples = Vec::new();
    for _ in 0..40 {
        samples.push((rng.below(total), 0usize));
    }
    let eps = 1e-5;
    for (flat, _) in samples {
        let mut block = 0;
        let mut offset = flat;
        while offset >= lengths[block] {
            offset -= lengths[block];
            block += 1;
        }
        let mut plus = model.clone();
        let mut minus = model.clone();
        let mut rng_a = Rng::new(9);
        let mut rng_b = Rng::new(9);
        let original = {
            let mut value = 0.0;
            let mut index = 0;
            plus.params.visit_blocks(|_, _, _, data| {
                if index == block {
                    value = data[offset];
                }
                index += 1;
            });
            value
        };
        set_at(&mut plus.params, block, offset, original + eps);
        set_at(&mut minus.params, block, offset, original - eps);
        let upper = plus.forward(&tokens, task, false, &mut rng_a).0;
        let lower = minus.forward(&tokens, task, false, &mut rng_b).0;
        let numeric = (0.5 * upper * upper - 0.5 * lower * lower) / (2.0 * eps);
        let analytic = grad_at(&mut grads, block, offset);
        assert_relative_eq!(analytic, numeric, epsilon = 1e-7, max_relative = 1e-4);
    }
}

#[test]
fn dropout_is_off_at_evaluation() {
    let mut cfg = tiny();
    cfg.dropout = 0.5;
    let model = Model::new(cfg, 2);
    let tokens = vec![11u32, 12, 13];
    let a = model.score(&tokens, 0);
    let b = model.score(&tokens, 0);
    assert_eq!(a, b);
}

fn flat(grads: &mut Grads) -> Vec<Real> {
    let mut out = Vec::new();
    grads.visit_blocks(|_, _, _, data| out.extend_from_slice(data));
    out
}

#[test]
fn accumulate_adds_every_block() {
    let model = Model::new(tiny(), 3);
    let mut left = Grads::zeros(&model.params);
    let mut right = Grads::zeros(&model.params);
    left.visit_blocks(|_, _, _, data| {
        for (index, value) in data.iter_mut().enumerate() {
            *value = index as Real + 1.0;
        }
    });
    right.visit_blocks(|_, _, _, data| {
        for value in data.iter_mut() {
            *value = 0.5;
        }
    });
    let mut expected = left.clone();
    expected.visit_blocks(|_, _, _, data| {
        for value in data.iter_mut() {
            *value += 0.5;
        }
    });
    left.accumulate(&mut right);
    assert_eq!(flat(&mut left), flat(&mut expected));
}

#[test]
fn seeded_forward_repeats_and_varies() {
    let mut config = tiny();
    config.dropout = 0.5;
    let model = Model::new(config, 7);
    let tokens: Vec<u32> = (1..8).collect();
    let (first, _) = model.forward_seeded(&tokens, 0, 99);
    let (again, _) = model.forward_seeded(&tokens, 0, 99);
    assert_eq!(first, again, "one seed must draw one mask");
    let (other, _) = model.forward_seeded(&tokens, 0, 100);
    assert_ne!(first, other, "two seeds must draw two masks");
}
