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
}

impl Default for CameraMatrix {
    fn default() -> Self {
        let v = nalgebra_glm::look_at_rh(
            &Vector3::<f32>::new(0.0, 80.0, 0.0),
            &Vector3::<f32>::new(40.0, 0.0, 40.0),
            &Vector3::<f32>::new(0.0, 1.0, 0.0),
        );
        let p = nalgebra_glm::perspective(1.4, 0.8, 1.0, 200.0);
        Self::from_view_proj(p * v)
    }
}

impl CameraMatrix {
    pub fn from_view_proj(camera_view_proj: Matrix4<f32>) -> Self {
        Self {
            view_proj: (camera_view_proj).into(),
        }
    }

    // The buffer creation and writing methods have been removed.
}
