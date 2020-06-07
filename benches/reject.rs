#![feature(test)]

extern crate test;

use rand::prelude::*;
use rand::distributions::uniform::{UniformInt, UniformSampler};
use test::{Bencher, black_box};

// In src/distributions/uniform.rs, we say:
// Implementation of [`sample_single`] is optional, and is only useful when
// the implementation can be faster than `Self::new(low, high).sample(rng)`.

// `UniformSampler::sample_single` compromises on the rejection range to be
// faster. This benchmark demonstrates both the speed gain of doing this, and
// the worst case behavior.  It allows improvements to the algorithm to be
// demonstrated.

/// Sample random values from a pre-existing distribution.  This uses the
/// half open `new` to be equivalent to the behavior of `sample_single`.
macro_rules! sample_dist {
    ($fnn:ident, $type:ident, $low:expr, $high:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = thread_rng();
            let low = black_box($low);
            let high = black_box($high);
            b.iter(|| {
                for _ in 0..10 {
                    let dist = UniformInt::<$type>::new(low, high);
                    black_box(dist.sample(&mut rng));
                }
            });
        }
    };
}

/// Use `sample_single` to create a one-off random value
macro_rules! sample_single {
    ($fnn:ident, $type:ident, $low:expr, $high:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = thread_rng();
            let low = black_box($low);
            let high = black_box($high);
            b.iter(|| {
                for _ in 0..10 {
                    black_box(UniformInt::<$type>::sample_single(low, high, &mut rng));
                }
            });
        }
    };
}


macro_rules! sample_single_cp {
    ($fnn:ident, $type:ident) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = thread_rng();
            b.iter(|| {
                for _ in 0..10 {
                    black_box(UniformInt::<$type>::sample_single(1, 7, &mut rng));
                }
            });
        }
    };
}


// For n = 1<<32, the number of values in a u32, benchmark:
// (n-1): only the max value is rejected: expect this to be fast
// n/2+1: almost half of the values are rejected, and we can do no better
// n/2: half the values are rejected, but could be doubled to have no rejection
// n/2-1: only a few values are rejected: expect this to be fast
// 6: 25% rejected, but could have few rejections
// 4: half the values are rejected, but could have no rejection

const HALF_32_BIT_UNSIGNED: u32 = 1 << 31;

sample_dist!(sample_u32_max_dist, u32, 0, u32::max_value());
sample_dist!(sample_u32_halfp1_dist, u32, 0, HALF_32_BIT_UNSIGNED + 1);
sample_dist!(sample_u32_half_dist, u32, 0, HALF_32_BIT_UNSIGNED);
sample_dist!(sample_u32_halfm1_dist, u32, 0, HALF_32_BIT_UNSIGNED - 1);
sample_dist!(sample_u32_6_dist, u32, 0, 6u32);
sample_dist!(sample_u32_4_dist, u32, 0, 4u32);

sample_single!(sample_u32_max_single, u32, 0, u32::max_value());
sample_single!(sample_u32_halfp1_single, u32, 0, HALF_32_BIT_UNSIGNED + 1);
sample_single!(sample_u32_half_single, u32, 0, HALF_32_BIT_UNSIGNED);
sample_single!(sample_u32_halfm1_single, u32, 0, HALF_32_BIT_UNSIGNED - 1);
sample_single!(sample_u32_6_single, u32, 0, 6u32);
sample_single_cp!(sample_u32_6_single_constant, u32);
sample_single!(sample_u32_4_single, u32, 0, 4u32);

const HALF_16_BIT_UNSIGNED: u16 = 1 << 15;

sample_dist!(sample_u16_max_dist, u16, 0, u16::max_value());
sample_dist!(sample_u16_halfp1_dist, u16, 0, HALF_16_BIT_UNSIGNED + 1);
sample_dist!(sample_u16_half_dist, u16, 0, HALF_16_BIT_UNSIGNED);
sample_dist!(sample_u16_halfm1_dist, u16, 0, HALF_16_BIT_UNSIGNED - 1);
sample_dist!(sample_u16_6_dist, u16, 0, 6u16);
sample_dist!(sample_u16_4_dist, u16, 0, 4u16);

sample_single!(sample_u16_max_single, u16, 0, u16::max_value());
sample_single!(sample_u16_halfp1_single, u16, 0, HALF_16_BIT_UNSIGNED + 1);
sample_single!(sample_u16_half_single, u16, 0, HALF_16_BIT_UNSIGNED);
sample_single!(sample_u16_halfm1_single, u16, 0, HALF_16_BIT_UNSIGNED - 1);
sample_single!(sample_u16_6_single, u16, 0, 6u16);
sample_single_cp!(sample_u16_6_single_constant, u16);
sample_single!(sample_u16_4_single, u16, 0, 4u16);

const HALF_8_BIT_UNSIGNED: u8 = 1 << 7;

sample_dist!(sample_u8_max_dist, u8, 0, u8::max_value());
sample_dist!(sample_u8_halfp1_dist, u8, 0, HALF_8_BIT_UNSIGNED + 1);
sample_dist!(sample_u8_half_dist, u8, 0, HALF_8_BIT_UNSIGNED);
sample_dist!(sample_u8_halfm1_dist, u8, 0, HALF_8_BIT_UNSIGNED - 1);
sample_dist!(sample_u8_6_dist, u8, 0, 6u8);
sample_dist!(sample_u8_4_dist, u8, 0, 4u8);

sample_single!(sample_u8_max_single, u8, 0, u8::max_value());
sample_single!(sample_u8_halfp1_single, u8, 0, HALF_8_BIT_UNSIGNED + 1);
sample_single!(sample_u8_half_single, u8, 0, HALF_8_BIT_UNSIGNED);
sample_single!(sample_u8_halfm1_single, u8, 0, HALF_8_BIT_UNSIGNED - 1);
sample_single!(sample_u8_6_single, u8, 0, 6u8);
sample_single_cp!(sample_u8_6_single_constant, u8);
sample_single!(sample_u8_4_single, u8, 0, 4u8);
