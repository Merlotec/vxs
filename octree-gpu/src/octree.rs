use crate::page::{Page, PageAllocator};

pub enum Node<T> {
    Empty,
    Parent([NodeIndex; 8]),
    Leaf(T),
}

pub struct NodeStore<T> {
    pub node: Option<Node<T>>,
    pub next_free: NodeIndex,
}

impl<T> NodeStore<T> {
    pub fn is_occupied(&self) -> bool {
        self.node.is_some()
    }

    pub fn store(&mut self, node: Node<T>) -> Option<Node<T>> {
        let old = self.node.take();
        self.node = Some(node);

        old
    }

    pub fn free(&mut self) -> Option<Node<T>> {
        self.node.take()
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct NodeIndex(usize);

impl NodeIndex {
    pub fn value(&self) -> usize {
        self.0
    }
}

pub struct GpuOctree<const N: usize, T> {
    head_free: NodeIndex,
    nodes: [NodeStore<T>; N],
}

impl<const N: usize, T> GpuOctree<N, T> {
    pub fn new() -> Self {
        let nodes = std::array::from_fn(|i| NodeStore {
            node: None,
            next_free: NodeIndex(i + 1),
        });
        Self {
            head_free: NodeIndex(0),
            nodes,
        }
    }

    fn allocate(&mut self, node: Node<T>) -> Result<(), OctreeError> {
        // Get node store by traversing the linked list.
        if self.head_free.value() >= N {
            return Err(OctreeError::OutOfMemory);
        }
        debug_assert!(!self.nodes[self.head_free.value()].is_occupied());

        self.nodes[self.head_free.value()].store(node);
        self.head_free = self.nodes[self.head_free.value()].next_free;

        Ok(())
    }

    pub fn insert(&mut self, coord: [i32; 3], value: T) {}
}

pub enum OctreeError {
    OutOfMemory,
    InconsistentFree,
}
