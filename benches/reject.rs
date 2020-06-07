#![feature(test)]

extern crate test;

use rand::prelude::*;
use rand::distributions::uniform::{UniformInt, UniformSampler};
use test::Bencher;

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
    ($fnn:ident, $low:expr, $high:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = thread_rng();
            let dist = <UniformInt<u32> as UniformSampler>::new($low, $high);
            b.iter(|| {
                test::black_box(dist.sample(&mut rng));
                test::black_box(dist.sample(&mut rng));
            });
        }
    };
}

/// Use `sample_single` to create a one-off random value
macro_rules! sample_single {
    ($fnn:ident, $low:expr, $high:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = thread_rng();
            b.iter(|| {
                test::black_box(<UniformInt<u32> as UniformSampler>::sample_single($low, $high, &mut rng));
                test::black_box(<UniformInt<u32> as UniformSampler>::sample_single($low, $high, &mut rng));
            });
        }
    };
}


// For n = 1<<32, the number of values in a u32, benchmark:
// (n-1): only the max value is rejected: expect this to be fast
// n/2+1: almost half of the values are rejected, and we can do no better
// n/2: half the values are rejected, but could be doubled to have no rejection
// n/2-1: only a few values are rejected: expect this to be fast
// 4: almost half that values are rejected, but could have no rejection
// 3: 25% rejected, but we could reject one value

const HALF_32_BIT_UNSIGNED: u32 = 1 << 31;

sample_dist!(sample_u32_max_dist, 0, u32::max_value());
sample_dist!(sample_u32_halfp1_dist, 0, HALF_32_BIT_UNSIGNED + 1);
sample_dist!(sample_u32_half_dist, 0, HALF_32_BIT_UNSIGNED);
sample_dist!(sample_u32_halfm1_dist, 0, HALF_32_BIT_UNSIGNED - 1);
sample_dist!(sample_u32_4_dist, 0, 4_u32);
sample_dist!(sample_u32_3_dist, 0, 3u32);

sample_single!(sample_u32_max_single, 0, u32::max_value());
sample_single!(sample_u32_halfp1_single, 0, HALF_32_BIT_UNSIGNED + 1);
sample_single!(sample_u32_half_single, 0, HALF_32_BIT_UNSIGNED);
sample_single!(sample_u32_halfm1_single, 0, HALF_32_BIT_UNSIGNED - 1);
sample_single!(sample_u32_4_single, 0, 4u32);
sample_single!(sample_u32_3_single, 0, 3u32);
