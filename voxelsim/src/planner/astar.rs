use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::agent::MoveDir;
use crate::{ActionIntent, PlannerError};
use crate::{Coord, VoxelGrid, env::Cell};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct Node(Coord);

#[derive(Copy, Clone, Debug, PartialEq)]
struct Cost {
    g: i32, // cost from start
    f: i32, // g + heuristic
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct QueueEntry {
    f: i32,
    node: Node,
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap behavior with BinaryHeap
        other
            .f
            .cmp(&self.f)
            .then_with(|| self.node.0.x.cmp(&other.node.0.x))
            .then_with(|| self.node.0.y.cmp(&other.node.0.y))
            .then_with(|| self.node.0.z.cmp(&other.node.0.z))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn manhattan(a: Coord, b: Coord) -> i32 {
    (a.x - b.x).abs() + (a.y - b.y).abs() + (a.z - b.z).abs()
}

fn is_obstacle_cell(cell: Cell) -> bool {
    cell.intersects(Cell::FILLED | Cell::GROUND | Cell::DRONE_OCCUPIED | Cell::DRONE_TRAJECTORY)
}

fn is_blocked(world: &VoxelGrid, c: &Coord, padding: i32) -> bool {
    // If padding is 0, only check the cell itself
    if padding <= 0 {
        return world
            .cells()
            .get(c)
            .map(|v| is_obstacle_cell(*v))
            .unwrap_or(false);
    }

    // Check Chebyshev neighbourhood within padding for any obstacle
    for dx in -padding..=padding {
        for dy in -padding..=padding {
            for dz in -padding..=padding {
                let nb = *c + Coord::new(dx, dy, dz);
                if let Some(cell) = world.cells().get(&nb) {
                    if is_obstacle_cell(*cell) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn neighbours6(c: Coord) -> [Coord; 6] {
    [
        c + Coord::new(1, 0, 0),
        c + Coord::new(-1, 0, 0),
        c + Coord::new(0, 1, 0),
        c + Coord::new(0, -1, 0),
        c + Coord::new(0, 0, 1),
        c + Coord::new(0, 0, -1),
    ]
}

fn delta_to_dir(d: Coord) -> Option<MoveDir> {
    match (d.x, d.y, d.z) {
        (1, 0, 0) => Some(MoveDir::Forward),
        (-1, 0, 0) => Some(MoveDir::Back),
        (0, 1, 0) => Some(MoveDir::Right),
        (0, -1, 0) => Some(MoveDir::Left),
        (0, 0, 1) => Some(MoveDir::Down),
        (0, 0, -1) => Some(MoveDir::Up),
        _ => None,
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct AStarActionPlanner {
    /// Padding radius (in blocks) around obstacles to treat as blocked.
    /// Padding applies in all directions (including corners/edges), i.e.
    /// any cell within Chebyshev distance <= padding from an obstacle is blocked.
    padding: i32,
}

impl AStarActionPlanner {
    pub fn new(padding: i32) -> Self {
        Self {
            padding: padding.max(0),
        }
    }

    fn find_nearest_target(start: Coord, world: &VoxelGrid) -> Option<Coord> {
        let mut best: Option<(i32, Coord)> = None;
        for kv in world.cells().iter() {
            if kv.intersects(Cell::TARGET) {
                let c = *kv.key();
                let dist = manhattan(start, c);
                if best.map(|(d, _)| dist < d).unwrap_or(true) {
                    best = Some((dist, c));
                }
            }
        }
        best.map(|(_, c)| c)
    }

    fn reconstruct_path(
        came_from: &HashMap<Coord, Coord>,
        mut current: Coord,
        start: Coord,
    ) -> Vec<MoveDir> {
        let mut steps: Vec<MoveDir> = Vec::new();
        while current != start {
            if let Some(prev) = came_from.get(&current) {
                let delta = current - *prev;
                if let Some(dir) = delta_to_dir(delta) {
                    steps.push(dir);
                }
                current = *prev;
            } else {
                break;
            }
        }
        steps.reverse();
        steps
    }
}

impl super::ActionPlanner for AStarActionPlanner {
    fn plan_action(
        &self,
        world: &VoxelGrid,
        origin: Coord,
        dst: Coord,
        urgency: f64,
        yaw: f64,
    ) -> Result<ActionIntent, PlannerError> {
        if origin == dst {
            return Err(PlannerError::InvalidParams);
        }

        if is_blocked(world, &origin, self.padding) {
            return Err(PlannerError::PathBlocked);
        }

        let mut open = BinaryHeap::<QueueEntry>::new();
        let mut came_from: HashMap<Coord, Coord> = HashMap::new();
        let mut g_score: HashMap<Coord, i32> = HashMap::new();
        let mut closed: HashSet<Coord> = HashSet::new();

        g_score.insert(origin, 0);
        open.push(QueueEntry {
            f: manhattan(origin, dst),
            node: Node(origin),
        });

        // Limit to avoid infinite search in degenerate inputs
        let max_expansions = 200_000;
        let mut expansions = 0;

        while let Some(QueueEntry {
            node: Node(curr), ..
        }) = open.pop()
        {
            if curr == dst {
                let move_sequence = Self::reconstruct_path(&came_from, dst, origin);
                return Ok(ActionIntent::new(urgency, yaw, move_sequence));
            }
            if !closed.insert(curr) {
                continue;
            }
            expansions += 1;
            if expansions > max_expansions {
                break;
            }

            for nb in neighbours6(curr) {
                if is_blocked(world, &nb, self.padding) || closed.contains(&nb) {
                    continue;
                }
                let tentative_g = g_score.get(&curr).copied().unwrap_or(i32::MAX - 1) + 1;
                let g_old = g_score.get(&nb).copied();
                if g_old.is_none() || tentative_g < g_old.unwrap() {
                    came_from.insert(nb, curr);
                    g_score.insert(nb, tentative_g);
                    let f = tentative_g + manhattan(nb, dst);
                    open.push(QueueEntry { f, node: Node(nb) });
                }
            }
        }

        // If search fails, return empty (no path)

        Err(PlannerError::NoViablePath)
    }
}
