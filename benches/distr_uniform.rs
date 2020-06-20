#![feature(test)]

extern crate test;

use std::mem::size_of;
use rand::prelude::*;
use rand::distributions::uniform::{Uniform, UniformInt, UniformSampler};
use rand_pcg::Pcg64Mcg;
use test::{Bencher, black_box};

// In src/distributions/uniform.rs, we say:
// Implementation of [`uniform_single`] is optional, and is only useful when
// the implementation can be faster than `Self::new(low, high).sample(rng)`.

// `UniformSampler::uniform_single` compromises on the rejection range to be
// faster. This benchmark demonstrates both the speed gain of doing this, and
// the worst case behavior.  It allows improvements to the algorithm to be
// demonstrated.

/// Sample random values from a pre-existing distribution.  This uses the
/// half open `new` to be equivalent to the behavior of `uniform_single`.
macro_rules! uniform_sample {
    ($fnn:ident, $type:ident, $low:expr, $high:expr, $count:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = thread_rng();
            let low = black_box($low);
            let high = black_box($high);
            b.iter(|| {
                for _ in 0..10 {
                    let dist = UniformInt::<$type>::new(low, high);
                    for _ in 0..$count {
                        black_box(dist.sample(&mut rng));
                    }
                }
            });
        }
    };
}

macro_rules! uniform_inclusive {
    ($fnn:ident, $type:ident, $low:expr, $high:expr, $count:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = thread_rng();
            let low = black_box($low);
            let high = black_box($high);
            b.iter(|| {
                for _ in 0..10 {
                    let dist = UniformInt::<$type>::new_inclusive(low, high);
                    for _ in 0..$count {
                        black_box(dist.sample(&mut rng));
                    }
                }
            });
        }
    };
}

/// Use `uniform_single` to create a one-off random value
macro_rules! uniform_single {
    ($fnn:ident, $type:ident, $low:expr, $high:expr, $count:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = thread_rng();
            let low = black_box($low);
            let high = black_box($high);
            b.iter(|| {
                for _ in 0..(10 * $count) {
                    black_box(UniformInt::<$type>::sample_single(low, high, &mut rng));
                }
            });
        }
    };
}


// There are two classes of integer types in `uniform.rs`: those that get
// widened to a larger type; and those that use the same type.  Use u16 and u32
// to represent the behavior of these two classes.

// With the use of u32 as the wide size, the worst-case u16 range (32769)
// will only loop 32769 / 4294967296 samples. Because of this, all values
// have roughly the same timings so we just check the values that might
// change with a different implementation.
const HALF_16_BIT_UNSIGNED: u16 = 1 << 15;

uniform_inclusive!(uniform_u16x10_all_new_inclusive, u16, 0, u16::max_value(), 10);
uniform_sample!(uniform_u16x10_allm1_new, u16, 0, u16::max_value(), 10);
uniform_sample!(uniform_u16x10_halfp1_new, u16, 0, HALF_16_BIT_UNSIGNED + 1, 10);
uniform_sample!(uniform_u16x10_6_new, u16, 0, 6u16, 10);
uniform_sample!(uniform_u16x10_4_new, u16, 0, 4u16, 10);

uniform_single!(uniform_u16x1_allm1_single, u16, 0, u16::max_value(), 1);
uniform_single!(uniform_u16x1_halfp1_single, u16, 0, HALF_16_BIT_UNSIGNED + 1, 1);
uniform_single!(uniform_u16x1_6_single, u16, 0, 6u16, 1);
uniform_single!(uniform_u16x1_4_single, u16, 0, 4u16, 1);

// For n = 1<<32, the number of values in a u32, benchmark:
// (n-1): only the max value is rejected: expect this to be fast
// n/2+1: almost half of the values are rejected, and we can do no better
// n/2: half the values are rejected, but could be doubled to have no rejection
// n/2-1: only a few values are rejected: expect this to be fast
// 6: 25% rejected, but could have few rejections

const HALF_32_BIT_UNSIGNED: u32 = 1 << 31;

uniform_sample!(uniform_u32x1_allm1_new, u32, 0, u32::max_value(), 1);
uniform_sample!(uniform_u32x1_halfp1_new, u32, 0, HALF_32_BIT_UNSIGNED + 1, 1);
uniform_sample!(uniform_u32x1_half_new, u32, 0, HALF_32_BIT_UNSIGNED, 1);
uniform_sample!(uniform_u32x1_halfm1_new, u32, 0, HALF_32_BIT_UNSIGNED - 1, 1);
uniform_sample!(uniform_u32x1_6_new, u32, 0, 6u32, 1);

uniform_single!(uniform_u32x1_allm1_single, u32, 0, u32::max_value(), 1);
uniform_single!(uniform_u32x1_halfp1_single, u32, 0, HALF_32_BIT_UNSIGNED + 1, 1);
uniform_single!(uniform_u32x1_half_single, u32, 0, HALF_32_BIT_UNSIGNED, 1);
uniform_single!(uniform_u32x1_halfm1_single, u32, 0, HALF_32_BIT_UNSIGNED - 1, 1);
uniform_single!(uniform_u32x1_6_single, u32, 0, 6u32, 1);

uniform_inclusive!(uniform_u32x10_all_new_inclusive, u32, 0, u32::max_value(), 10);
uniform_sample!(uniform_u32x10_allm1_new, u32, 0, u32::max_value(), 10);
uniform_sample!(uniform_u32x10_halfp1_new, u32, 0, HALF_32_BIT_UNSIGNED + 1, 10);
uniform_sample!(uniform_u32x10_half_new, u32, 0, HALF_32_BIT_UNSIGNED, 10);
uniform_sample!(uniform_u32x10_halfm1_new, u32, 0, HALF_32_BIT_UNSIGNED - 1, 10);
uniform_sample!(uniform_u32x10_6_new, u32, 0, 6u32, 10);

uniform_single!(uniform_u32x10_allm1_single, u32, 0, u32::max_value(), 10);
uniform_single!(uniform_u32x10_halfp1_single, u32, 0, HALF_32_BIT_UNSIGNED + 1, 10);
uniform_single!(uniform_u32x10_half_single, u32, 0, HALF_32_BIT_UNSIGNED, 10);
uniform_single!(uniform_u32x10_halfm1_single, u32, 0, HALF_32_BIT_UNSIGNED - 1, 10);
uniform_single!(uniform_u32x10_6_single, u32, 0, 6u32, 10);

const HALF_64_BIT_UNSIGNED: u64 = 1 << 63;

uniform_sample!(uniform_u64x1_allm1_new, u64, 0, u64::max_value(), 1);
uniform_sample!(uniform_u64x1_halfp1_new, u64, 0, HALF_64_BIT_UNSIGNED + 1, 1);
uniform_sample!(uniform_u64x1_half_new, u64, 0, HALF_64_BIT_UNSIGNED, 1);
uniform_sample!(uniform_u64x1_halfm1_new, u64, 0, HALF_64_BIT_UNSIGNED - 1, 1);
uniform_sample!(uniform_u64x1_6_new, u64, 0, 6u64, 1);

uniform_single!(uniform_u64x1_allm1_single, u64, 0, u64::max_value(), 1);
uniform_single!(uniform_u64x1_halfp1_single, u64, 0, HALF_64_BIT_UNSIGNED + 1, 1);
uniform_single!(uniform_u64x1_half_single, u64, 0, HALF_64_BIT_UNSIGNED, 1);
uniform_single!(uniform_u64x1_halfm1_single, u64, 0, HALF_64_BIT_UNSIGNED - 1, 1);
uniform_single!(uniform_u64x1_6_single, u64, 0, 6u64, 1);

uniform_inclusive!(uniform_u64x10_all_new_inclusive, u64, 0, u64::max_value(), 10);
uniform_sample!(uniform_u64x10_allm1_new, u64, 0, u64::max_value(), 10);
uniform_sample!(uniform_u64x10_halfp1_new, u64, 0, HALF_64_BIT_UNSIGNED + 1, 10);
uniform_sample!(uniform_u64x10_half_new, u64, 0, HALF_64_BIT_UNSIGNED, 10);
uniform_sample!(uniform_u64x10_halfm1_new, u64, 0, HALF_64_BIT_UNSIGNED - 1, 10);
uniform_sample!(uniform_u64x10_6_new, u64, 0, 6u64, 10);

uniform_single!(uniform_u64x10_allm1_single, u64, 0, u64::max_value(), 10);
uniform_single!(uniform_u64x10_halfp1_single, u64, 0, HALF_64_BIT_UNSIGNED + 1, 10);
uniform_single!(uniform_u64x10_half_single, u64, 0, HALF_64_BIT_UNSIGNED, 10);
uniform_single!(uniform_u64x10_halfm1_single, u64, 0, HALF_64_BIT_UNSIGNED - 1, 10);
uniform_single!(uniform_u64x10_6_single, u64, 0, 6u64, 10);

const HALF_128_BIT_UNSIGNED: u128 = 1 << 127;

uniform_sample!(uniform_u128x1_allm1_new, u128, 0, u128::max_value(), 1);
uniform_sample!(uniform_u128x1_halfp1_new, u128, 0, HALF_128_BIT_UNSIGNED + 1, 1);
uniform_sample!(uniform_u128x1_half_new, u128, 0, HALF_128_BIT_UNSIGNED, 1);
uniform_sample!(uniform_u128x1_halfm1_new, u128, 0, HALF_128_BIT_UNSIGNED - 1, 1);
uniform_sample!(uniform_u128x1_6_new, u128, 0, 6u128, 1);

uniform_single!(uniform_u128x1_allm1_single, u128, 0, u128::max_value(), 1);
uniform_single!(uniform_u128x1_halfp1_single, u128, 0, HALF_128_BIT_UNSIGNED + 1, 1);
uniform_single!(uniform_u128x1_half_single, u128, 0, HALF_128_BIT_UNSIGNED, 1);
uniform_single!(uniform_u128x1_halfm1_single, u128, 0, HALF_128_BIT_UNSIGNED - 1, 1);
uniform_single!(uniform_u128x1_6_single, u128, 0, 6u128, 1);

uniform_inclusive!(uniform_u128x10_all_new_inclusive, u128, 0, u128::max_value(), 10);
uniform_sample!(uniform_u128x10_allm1_new, u128, 0, u128::max_value(), 10);
uniform_sample!(uniform_u128x10_halfp1_new, u128, 0, HALF_128_BIT_UNSIGNED + 1, 10);
uniform_sample!(uniform_u128x10_half_new, u128, 0, HALF_128_BIT_UNSIGNED, 10);
uniform_sample!(uniform_u128x10_halfm1_new, u128, 0, HALF_128_BIT_UNSIGNED - 1, 10);
uniform_sample!(uniform_u128x10_6_new, u128, 0, 6u128, 10);

uniform_single!(uniform_u128x10_allm1_single, u128, 0, u128::max_value(), 10);
uniform_single!(uniform_u128x10_halfp1_single, u128, 0, HALF_128_BIT_UNSIGNED + 1, 10);
uniform_single!(uniform_u128x10_half_single, u128, 0, HALF_128_BIT_UNSIGNED, 10);
uniform_single!(uniform_u128x10_halfm1_single, u128, 0, HALF_128_BIT_UNSIGNED - 1, 10);
uniform_single!(uniform_u128x10_6_single, u128, 0, 6u128, 10);

const RAND_BENCH_N: u64 = 1000;

// construct and sample from a range
macro_rules! gen_range_int {
    ($fnn:ident, $ty:ident, $low:expr, $high:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = Pcg64Mcg::from_entropy();

            b.iter(|| {
                let mut high = $high;
                let mut accum: $ty = 0;
                for _ in 0..RAND_BENCH_N {
                    accum = accum.wrapping_add(rng.gen_range($low, high));
                    // Force recalculation of range each time.
                    // The `& MAX` cycles ints through non-negative values.
                    high = high.wrapping_add(1) & std::$ty::MAX;
                }
                accum
            });
            b.bytes = size_of::<$ty>() as u64 * RAND_BENCH_N;
        }
    };
}

// Fisher–Yates shuffle uses single values from incrementing ranges.
// Usually, this is 0..n but we use -1..n here to prevent the wrapping in
// the test from generating a 0-sized range.
gen_range_int!(gen_range_i8_from_1, i8, -1i8, 0);
gen_range_int!(gen_range_i16_from_1, i16, -1i16, 0);
gen_range_int!(gen_range_i32_from_1, i32, -1i32, 0);
gen_range_int!(gen_range_i64_from_1, i64, -1i64, 0);
gen_range_int!(gen_range_i128_from_1, i128, -1i128, 0);

// These ranges are carried over from previous test
gen_range_int!(gen_range_i8, i8, -20i8, 100);
gen_range_int!(gen_range_i16, i16, -500i16, 2000);
gen_range_int!(gen_range_i32, i32, -200_000_000i32, 800_000_000);
gen_range_int!(gen_range_i64, i64, 3i64, 123_456_789_123);
gen_range_int!(gen_range_i128, i128, -12345678901234i128, 123_456_789_123_456_789);

macro_rules! distr_int {
    ($fnn:ident, $ty:ty, $distr:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = Pcg64Mcg::from_entropy();
            let distr = $distr;

            b.iter(|| {
                let mut accum = 0 as $ty;
                for _ in 0..RAND_BENCH_N {
                    let x: $ty = distr.sample(&mut rng);
                    accum = accum.wrapping_add(x);
                }
                accum
            });
            b.bytes = size_of::<$ty>() as u64 * RAND_BENCH_N;
        }
    };
}

distr_int!(distr_uniform_i8, i8, Uniform::new(20i8, 100));
distr_int!(distr_uniform_i16, i16, Uniform::new(-500i16, 2000));
distr_int!(distr_uniform_i32, i32, Uniform::new(-200_000_000i32, 800_000_000));
distr_int!(distr_uniform_i64, i64, Uniform::new(3i64, 123_456_789_123));
distr_int!(distr_uniform_i128, i128, Uniform::new(-123_456_789_123i128, 123_456_789_123_456_789));
distr_int!(distr_uniform_usize16, usize, Uniform::new(0usize, 0xb9d7));
distr_int!(distr_uniform_usize32, usize, Uniform::new(0usize, 0x548c0f43));
#[cfg(target_pointer_width = "64")]
distr_int!(distr_uniform_usize64, usize, Uniform::new(0usize, 0x3a42714f2bf927a8));
distr_int!(distr_uniform_isize, isize, Uniform::new(-1060478432isize, 1858574057));
