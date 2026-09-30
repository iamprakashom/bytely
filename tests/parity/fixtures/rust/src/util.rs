use std::cmp::min;

pub fn clamp(x: u32) -> u32 {
    min(x, crate::LIMIT)
}
