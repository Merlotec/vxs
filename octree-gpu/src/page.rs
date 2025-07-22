pub struct PageAllocator<const N: usize>([u8; N / 8]);

pub struct Page<const P: usize> {}
