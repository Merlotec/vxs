use nalgebra::Vector2;

pub struct TextureSet {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl TextureSet {
    pub fn create_render_target_texture(
        device: &wgpu::Device,
        size: [u32; 2],
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        let size3d = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size: size3d,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // Usage needs to include RENDER_ATTACHMENT to be a render target
            // and TEXTURE_BINDING to be readable by a shader.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,

            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }

    pub fn create_depth_texture(device: &wgpu::Device, size: [u32; 2], label: &str) -> Self {
        let size = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellTexel {
    coord: Coord,
    value: u32,
}

use voxelsim::{Cell, Coord};
use wgpu::util::DeviceExt;

pub async fn extract_cells_from_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Vec<(Coord, Cell)> {
    let data = extract_texture_bytes(device, queue, texture).await.unwrap();
    let cells: &[(Coord, Cell)] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const (Coord, Cell),
            data.len() / std::mem::size_of::<(Coord, Cell)>(),
        )
    };
    cells.to_vec()
}

pub async fn extract_texture_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Result<Vec<u8>, wgpu::Error> {
    let texture_format = texture.format();
    let (block_width, block_height) = texture_format.block_dimensions();
    let bytes_per_block = texture_format.block_copy_size(None).unwrap();

    let buffer_size = wgpu::util::align_to(
        texture.width() / block_width * bytes_per_block,
        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
    ) * (texture.height() / block_height);

    // 1. Create the destination buffer
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Texture Readback Buffer"),
        size: buffer_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // 2. Issue the copy command
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Texture-to-Buffer Encoder"),
    });

    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(wgpu::util::align_to(
                    texture.width() / block_width * bytes_per_block,
                    wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
                )),
                rows_per_image: None,
            },
        },
        texture.size(),
    );

    // 3. Submit and map the buffer
    queue.submit(Some(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |result| {
        if let Err(e) = result {
            eprintln!("Failed to map buffer: {:?}", e);
        }
    });

    // You must poll the device to drive the async map operation.
    // In a real app, this would be part of your event loop.
    device.poll(wgpu::PollType::Wait);

    let data = buffer_slice.get_mapped_range().to_vec();
    output_buffer.unmap();

    Ok(data)
}

pub fn create_depth_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Depth Comparison Sampler"),

        // How to handle coordinates outside the 0.0-1.0 range.
        // ClampToEdge is standard for screen-space textures.
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,

        // Filtering for magnification and minification.
        // Linear provides smooth sampling between texels.
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,

        // Mipmap filtering. Nearest is fine if you're not using mipmaps.
        mipmap_filter: wgpu::FilterMode::Nearest,

        // Level-of-detail clamps (can usually be left as default).
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,

        ..Default::default()
    })
}
