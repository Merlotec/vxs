pub async fn clear_gpu_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Clear Buffer Encoder"),
    });

    encoder.clear_buffer(buffer, 0, None);

    queue.submit(Some(encoder.finish()));

    device.poll(wgpu::wgt::PollType::Wait).unwrap();
}

pub async fn read_gpu_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_buffer: &wgpu::Buffer,
    clear: bool,
) -> Vec<u8> {
    let buffer_size = source_buffer.size();

    // 1. Create a staging buffer to copy the data into
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size,
        // This buffer is for copying data from the GPU and mapping it for reading on the CPU
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 2. Create a command encoder to copy the data
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Read Buffer Encoder"),
    });

    encoder.copy_buffer_to_buffer(
        &source_buffer,
        0, // source offset
        &staging_buffer,
        0, // destination offset
        buffer_size,
    );

    // Clear the buffer if required.
    if clear {
        encoder.clear_buffer(&source_buffer, 0, None);
    }

    // Submit the command to the GPU
    queue.submit(Some(encoder.finish()));

    // 3. Map the staging buffer to read its contents
    let buffer_slice = staging_buffer.slice(..);

    // Asynchronously request to map the buffer.
    // The `*_async` methods are non-blocking.
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    // You must poll the device to ensure the async operations complete.
    // In a real application, this would be part of your event loop.
    device.poll(wgpu::wgt::PollType::Wait).unwrap();

    // Wait for the mapping to complete and check for errors.
    if let Some(Ok(())) = receiver.receive().await {
        // Get the mapped buffer view
        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();

        // Unmap the buffer after you're done with it
        drop(data);
        staging_buffer.unmap();

        result
    } else {
        panic!("Failed to read buffer from GPU!");
    }
}
