use std::ops::{BitXor, BitXorAssign};

use curve25519_dalek::{
    constants::ED25519_BASEPOINT_TABLE,
    edwards::{EdwardsBasepointTable, EdwardsPoint},
    scalar::Scalar,
};
use okvs::bits::Bits;
use rand::{thread_rng, RngCore};

use crate::{
    point_conversion::{point_to_uniform_bytes, uniform_bytes_to_point},
    wide_ciphertexts::WideCiphertext,
};

#[derive(Debug, Clone, Copy)]
pub struct Ciphertext {
    s1: [u8; 32],
    s2: [u8; 32],
}

impl Ciphertext {
    pub fn zero(public_key_table: &EdwardsBasepointTable) -> Self {
        let (s1, s2) = loop {
            let r = Scalar::random(&mut thread_rng());

            let c1 = &r * ED25519_BASEPOINT_TABLE;
            let s1 = point_to_uniform_bytes(c1);
            if s1.is_none() {
                continue;
            }

            let c2 = &r * public_key_table;
            let s2 = point_to_uniform_bytes(c2);
            if let Some(s2_unwrapped) = s2 {
                break (s1.unwrap(), s2_unwrapped);
            }
        };

        Self { s1, s2 }
    }

    pub fn add_and_randomize(
        ciphertext_a: Self,
        ciphertext_b: Self,
        public_key: &EdwardsPoint,
    ) -> WideCiphertext {
        let randomness = Scalar::random(&mut thread_rng());

        let point_a_c1 = uniform_bytes_to_point(&ciphertext_a.s1);
        let point_a_c2 = uniform_bytes_to_point(&ciphertext_a.s2);
        let point_b_c1 = uniform_bytes_to_point(&ciphertext_b.s1);
        let point_b_c2 = uniform_bytes_to_point(&ciphertext_b.s2);

        // Represents the sum of the two ciphertexts multiplied by some random scalar.
        WideCiphertext {
            a: (randomness * point_a_c1).compress().to_bytes(),
            b: (randomness * point_b_c1).compress().to_bytes(),
            c: (&randomness * ED25519_BASEPOINT_TABLE)
                .compress()
                .to_bytes(),
            d: (randomness * (point_a_c2 + point_b_c2 + public_key))
                .compress()
                .to_bytes(),
        }
    }
}

impl Bits for Ciphertext {
    fn random() -> Self {
        let mut s1 = [0; 32];
        thread_rng().fill_bytes(&mut s1);
        let mut s2 = [0; 32];
        thread_rng().fill_bytes(&mut s2);

        Self { s1, s2 }
    }

    const BYTES: usize = 64;

    fn to_bytes(self) -> Vec<u8> {
        [self.s1, self.s2].concat()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut s1 = [0; 32];
        let mut s2 = [0; 32];

        s1.copy_from_slice(&bytes[0..32]);
        s2.copy_from_slice(&bytes[32..64]);

        Self { s1, s2 }
    }
}

impl BitXor for Ciphertext {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            s1: self.s1.zip(rhs.s1).map(|(x, y)| x ^ y),
            s2: self.s2.zip(rhs.s2).map(|(x, y)| x ^ y),
        }
    }
}

impl BitXorAssign for Ciphertext {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}
