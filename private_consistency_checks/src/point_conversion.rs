use curve25519_dalek::{
    edwards::EdwardsPoint,
    field::FieldElement51,
    montgomery::{elligator_decode, elligator_encode},
};
use rand::{thread_rng, Rng};
use subtle::Choice;

pub fn point_to_uniform_bytes(point: EdwardsPoint) -> Option<[u8; 32]> {
    let montgomery = point.to_montgomery();
    let v_negative: u8 = thread_rng().gen_range(0..=1);
    let mut bytes = elligator_decode(&montgomery, Choice::from(v_negative))?.as_bytes();

    // Check the sign of the point
    if uniform_bytes_to_point(&bytes) == point {
        return Some(bytes);
    }

    // If it must be negative, encode it in the highest bit
    bytes[31] |= 1 << 7;
    Some(bytes)
}

pub fn uniform_bytes_to_point(bytes: &[u8; 32]) -> EdwardsPoint {
    let sign = bytes[31] >> 7;
    elligator_encode(&FieldElement51::from_bytes(bytes))
        .to_edwards(sign)
        .expect("The point from the Elligator map is invalid, this should never happen.")
}

#[cfg(test)]
mod tests {
    use curve25519_dalek::{constants::ED25519_BASEPOINT_TABLE, scalar::Scalar};
    use rand::thread_rng;

    use super::{point_to_uniform_bytes, uniform_bytes_to_point};

    #[test]
    fn test_correct_mapping_random_point() {
        let (point, s) = loop {
            let r = Scalar::random(&mut thread_rng());
            let point = &r * ED25519_BASEPOINT_TABLE;
            let s = point_to_uniform_bytes(point.clone());
            if s.is_some() {
                break (point, s.unwrap());
            }
        };

        assert_eq!(uniform_bytes_to_point(&s), point);
    }
}
