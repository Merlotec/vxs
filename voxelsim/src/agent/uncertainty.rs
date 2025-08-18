use nalgebra::{UnitVector3, Vector3};

/// Dense grid of uncertainty.
/// Allows decoupling of uncertainty and block position.
pub struct UncertaintyField<T> {
    field_storage: Vec<[T; 8]>,
    size: Vector3<usize>,
    node_bounds: Vector3<f64>,
    origin: Vector3<f64>,
}

impl<T: num::Num + Copy> UncertaintyField<T> {
    pub fn new(size: Vector3<usize>, node_bounds: Vector3<f64>) -> Self {
        let len = size.x * size.y * size.z;

        Self {
            field_storage: vec![[T::zero(); 8]; len],
            size,
            node_bounds,
        }
    }
}
impl<T> UncertaintyField<T> {
    pub const fn dir(i: usize) -> UnitVector3<f64> {
        let x: f64 = if i < 4 {
            std::f64::consts::SQRT_3
        } else {
            -std::f64::consts::SQRT_3
        };

        let y: f64 = if i % 4 < 2 {
            std::f64::consts::SQRT_3
        } else {
            -std::f64::consts::SQRT_3
        };

        let z: f64 = if i % 2 == 0 {
            std::f64::consts::SQRT_3
        } else {
            -std::f64::consts::SQRT_3
        };

        UnitVector3::new_unchecked(Vector3::new(x, y, z))
    }

    pub const fn dirs() -> [UnitVector3<f64>; 8] {
        [
            Self::dir(0),
            Self::dir(1),
            Self::dir(2),
            Self::dir(3),
            Self::dir(4),
            Self::dir(5),
            Self::dir(6),
            Self::dir(7),
        ]
    }

    pub fn index(&self, coord: Vector3<usize>) -> usize {
        coord.x * (self.size.y * self.size.z) + coord.y * self.size.z + coord.z
    }
    pub fn get(&self, coord: Vector3<usize>) -> Option<&[T; 8]> {
        self.field_storage.get(self.index(coord))
    }
}

impl<T: num::Num + TryInto<f64> + Copy> UncertaintyField<T> {
    pub fn sample_node(&self, coord: Vector3<usize>, in_dir: UnitVector3<f64>) -> Option<f64> {
        let values = self.get(coord)?;
        let dirs = Self::dirs().map(|d| d.dot(&in_dir));

        let to_collect: Vec<(usize, f64)> = dirs
            .into_iter()
            .enumerate()
            .filter(|(_, x)| *x > 0.0)
            .collect();

        let sum: f64 = to_collect.iter().map(|(_, x)| x).sum();

        Some(
            to_collect
                .iter()
                .map(|(i, x)| (x / sum) * values[*i].try_into().unwrap_or(0.0))
                .sum(),
        )
    }

    pub fn sample_field(&self, pos: Vector3<f64>, in_dir: UnitVector3<f64>) -> Option<f64> {
        // Take weighted average of four nearest nodes.
    }
}
