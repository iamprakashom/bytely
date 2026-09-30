pub enum Shape {
    Circle(f64),
    Square(f64),
}

pub trait Named {
    fn name(&self) -> String;

    fn describe(&self) -> String {
        format!("a {}", self.name())
    }
}

impl Named for Shape {
    fn name(&self) -> String {
        String::from("shape")
    }
}

pub fn area(shape: &Shape) -> f64 {
    match shape {
        Shape::Circle(r) => 3.14 * r * r,
        Shape::Square(s) => s * s,
    }
}
