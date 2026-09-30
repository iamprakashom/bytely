pub mod shapes;
mod util;

use crate::shapes::{area, Shape};
use crate::util::*;

pub const LIMIT: u32 = 10;

pub struct Cache {
    hits: u32,
}

impl Cache {
    pub fn new() -> Self {
        Cache { hits: 0 }
    }

    pub fn get(&self) -> u32 {
        clamp(self.hits)
    }
}

pub fn run(shape: &Shape) -> f64 {
    fn scale(x: f64) -> f64 {
        x * 2.0
    }
    let cache = Cache::new();
    scale(area(shape)) + f64::from(cache.get())
}
