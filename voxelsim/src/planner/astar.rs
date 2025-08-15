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

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct StateKey {
    coord: Coord,
    // -1 indicates None (no previous direction)
    last_dir_code: i8,
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct QueueEntry {
    f: i32,
    state: StateKey,
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap behavior with BinaryHeap
        other
            .f
            .cmp(&self.f)
            .then_with(|| self.state.coord.x.cmp(&other.state.coord.x))
            .then_with(|| self.state.coord.y.cmp(&other.state.coord.y))
            .then_with(|| self.state.coord.z.cmp(&other.state.coord.z))
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

#[inline]
fn move_dir_code(dir: Option<MoveDir>) -> i8 {
    match dir {
        Some(d) => d as i8,
        None => -1,
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

        // Direction-aware A*: include last direction in the state to prefer
        // paths with more direction changes (i.e., fewer repeated straight steps).
        let mut open = BinaryHeap::<QueueEntry>::new();
        let mut came_from: HashMap<StateKey, StateKey> = HashMap::new();
        let mut g_score: HashMap<StateKey, i32> = HashMap::new();
        let mut closed: HashSet<StateKey> = HashSet::new();

        // Scale costs to allow a small per-step penalty while keeping integers
        const STEP_COST: i32 = 1000;
        const STRAIGHT_PENALTY: i32 = 1; // applied when continuing in the same direction

        let start = StateKey { coord: origin, last_dir_code: -1 };
        g_score.insert(start, 0);
        open.push(QueueEntry {
            f: manhattan(origin, dst) * STEP_COST,
            state: start,
        });

        // Limit to avoid infinite search in degenerate inputs
        let max_expansions = 200_000;
        let mut expansions = 0;

        while let Some(QueueEntry { state, .. }) = open.pop()
        {
            if state.coord == dst {
                // Reconstruct using direction-aware came_from
                let mut steps: Vec<MoveDir> = Vec::new();
                let mut cur = state;
                while cur.coord != origin {
                    if let Some(prev) = came_from.get(&cur) {
                        let delta = cur.coord - prev.coord;
                        if let Some(dir) = delta_to_dir(delta) {
                            steps.push(dir);
                        }
                        cur = *prev;
                    } else {
                        break;
                    }
                }
                steps.reverse();
                let move_sequence = steps;
                return Ok(ActionIntent::new(urgency, yaw, move_sequence));
            }
            if !closed.insert(state) {
                continue;
            }
            expansions += 1;
            if expansions > max_expansions {
                break;
            }

            for nb in neighbours6(state.coord) {
                let next_dir = delta_to_dir(nb - state.coord);
                if is_blocked(world, &nb, self.padding) {
                    continue;
                }
                // Compute per-step cost with preference for direction change
                let same_dir = match next_dir {
                    Some(d) if state.last_dir_code >= 0 => (d as i8) == state.last_dir_code,
                    _ => false,
                };
                let step_penalty = if same_dir { STRAIGHT_PENALTY } else { 0 };
                let curr_g = g_score.get(&state).copied().unwrap_or(i32::MAX / 4);
                let tentative_g = curr_g + STEP_COST + step_penalty;

                let next_state = StateKey { coord: nb, last_dir_code: move_dir_code(next_dir) };
                if closed.contains(&next_state) {
                    continue;
                }
                let g_old = g_score.get(&next_state).copied();
                if g_old.is_none() || tentative_g < g_old.unwrap() {
                    came_from.insert(next_state, state);
                    g_score.insert(next_state, tentative_g);
                    let f = tentative_g + manhattan(nb, dst) * STEP_COST;
                    open.push(QueueEntry { f, state: next_state });
                }
            }
        }

        // If search fails, return empty (no path)

        Err(PlannerError::NoViablePath)
    }
}
