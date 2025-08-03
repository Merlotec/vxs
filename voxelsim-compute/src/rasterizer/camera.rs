use nalgebra::{Matrix4, Vector3};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

// The CameraBinding struct has been removed entirely.

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraMatrix {
    pub view_proj: [[f32; 4]; 4],
    pub pos: [f32; 3],
}

impl Default for CameraMatrix {
    fn default() -> Self {
        let v = nalgebra_glm::look_at_rh(
            &Vector3::<f32>::new(0.0, 80.0, 0.0),
            &Vector3::<f32>::new(40.0, 0.0, 40.0),
            &Vector3::<f32>::new(0.0, 1.0, 0.0),
        );
        let p = nalgebra_glm::perspective(1.4, 0.8, 1.0, 200.0);
        Self::from_view_proj(v, p)
    }
}

impl CameraMatrix {
    pub fn from_view_proj(view: Matrix4<f32>, proj: Matrix4<f32>) -> Self {
        let pos = Self::camera_pos_fast(&view);
        Self {
            view_proj: (proj * view).into(),
            pos: pos.into(),
        }
    }

    fn camera_pos_fast(view: &Matrix4<f32>) -> Vector3<f32> {
        // Grab the 3×3 rotation block
        let rot3 = view.fixed_view::<3, 3>(0, 0).into_owned();
        // Grab the translation column (3×1)
        let trans = view.fixed_view::<3, 1>(0, 3).into_owned();
        // p_cam = - Rᵀ * t
        -(rot3.transpose() * trans)
    }
    // The buffer creation and writing methods have been removed.
}
