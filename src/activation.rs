#[derive(Clone, Copy)]
pub struct Activation {
    pub function: fn(f32) -> f32,
    pub derivative: fn(f32) -> f32,
}

impl Activation {
    // sigmoid function [0, 1]
    pub fn sigmoid() -> Self {
        Self {
            function: |x| 1.0 / (1.0 + (-x).exp()),
            derivative: |x| {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            },
        }
    }
    // 0 if x<0 ; x if x>0
    pub fn relu() -> Self {
        Self {
            function: |x| if x > 0.0 { x } else { 0.0 },
            derivative: |x| if x > 0.0 { 1.0 } else { 0.0 },
        }
    }
    // [-inf, +inf] , useful to avoid dead neurons
    pub fn leaky_relu() -> Self {
        Self {
            function: |x| if x > 0.0 { x } else { 0.01 * x },
            derivative: |x| if x > 0.0 { 1.0 } else { 0.01 },
        }
    }

    // Tanh : [-1, 1] - faster than a sigmoid
    pub fn tanh() -> Self {
        Self {
            function: |x| x.to_owned().tanh(),
            derivative: |x| {
                let t = x.to_owned().tanh();
                1.0 - t * t
            },
        }
    }

    // No changes
    pub fn identity() -> Self {
        Self {
            function: |x| x,
            derivative: |_| 1.0,
        }
    }
}
