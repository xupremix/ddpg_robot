use crate::gym::robot::GymRobot;
use crate::gym::state::State;
use crate::utils::args::Mode;
use crate::utils::consts::{
    CONTENT_TARGETS, EVAL_PLOT_PATH, FONT_SIZE, LABEL_AREA_SIZE, PLOT_FONT, PLOT_HEIGHT,
    PLOT_WIDTH, TRAIN_PLOT_PATH, X_LABELS, Y_LABELS,
};
use ghost_journey_journal::JourneyJournal;
use plotters::prelude::{
    BitMapBackend, ChartBuilder, IntoDrawingArea, IntoFont, LineSeries, BLACK, WHITE,
};
use robotics_lib::interface::{look_at_sky, robot_view, Direction};
use robotics_lib::runner::Runnable;
use robotics_lib::utils::calculate_cost_go_with_environment;
use robotics_lib::world::tile::{Content, Tile};
use robotics_lib::world::World;
use std::cell::RefMut;
use tch::nn::VarStore;
use tch::no_grad;

// weighted sum of two trainable variables
pub fn update_vs(dst: &mut VarStore, src: &VarStore, tau: f64) {
    no_grad(|| {
        for (dest, src) in dst
            .trainable_variables()
            .iter_mut()
            .zip(src.trainable_variables().iter())
        {
            dest.copy_(&(tau * src + (1.0 - tau) * &*dest));
        }
    })
}

pub fn plot(mode: &Mode, memory: Vec<f64>, min_rw: f64, max_rw: f64) {
    let path = match mode {
        Mode::Init => println!("Cannot plot in init mode"),
        Mode::Train { .. } => TRAIN_PLOT_PATH.to_string(),
        Mode::Eval => EVAL_PLOT_PATH.to_string(),
    };
    // plot background
    let root = BitMapBackend::new(&path, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // if the min reward > 0. the y axis will be shifted to 0. otherwise it will stay at that value
    let min_rw = min_rw.min(0.0);

    // create the chart
    let mut chart = ChartBuilder::on(&root)
        .caption("Action-Reward Plot", (PLOT_FONT, FONT_SIZE).into_font())
        .x_label_area_size(LABEL_AREA_SIZE)
        .y_label_area_size(LABEL_AREA_SIZE)
        .build_cartesian_2d(0f64..memory.len() as f64, min_rw..max_rw)
        .unwrap();

    // draw the grid
    chart
        .configure_mesh()
        .x_labels(X_LABELS)
        .y_labels(Y_LABELS)
        .draw()
        .unwrap();

    // create the line points
    let data_points: Vec<(f64, f64)> = (0..memory.len()).map(|i| (i as f64, memory[i])).collect();

    // draw the lines
    chart
        .draw_series(LineSeries::new(data_points, &BLACK))
        .unwrap();
}

pub fn update_closest(robot: &mut GymRobot, world: &mut World) {
    // do a scan of the adj tiles
    let _ = robot_view(robot, world);
    // reset bank and coin adj
    robot.state.borrow_mut().coin_adj = [0.0; 4];
    robot.state.borrow_mut().bank_adj = [0.0; 4];

    let mut journal = JourneyJournal::new(&[], &CONTENT_TARGETS, false);

    // get the closest coin and bank
    let closest_coin: Option<(usize, usize)> = journal
        .contents_closest_coords(&Content::Coin(0), robot, world)
        .unwrap();
    let closest_bank: Option<(usize, usize)> = journal
        .contents_closest_coords(&Content::Bank(0..0), robot, world)
        .unwrap();

    // update the coin and bank direction and adjacency
    let robot_i = robot.get_coordinate().get_row();
    let robot_j = robot.get_coordinate().get_col();

    let update_coin_dir_fn =
        |i: usize, robot_i: usize, j: usize, robot_j: usize, robot: &mut GymRobot| {
            if i < robot_i {
                robot.state.borrow_mut().coin_dir[0] = 1.0;
            } else if i > robot_i {
                robot.state.borrow_mut().coin_dir[2] = 1.0;
            }
            if j < robot_j {
                robot.state.borrow_mut().coin_dir[3] = 1.0;
            } else if j > robot_j {
                robot.state.borrow_mut().coin_dir[1] = 1.0;
            }
        };

    let update_bank_dir_fn =
        |i: usize, robot_i: usize, j: usize, robot_j: usize, robot: &mut GymRobot| {
            if i < robot_i {
                robot.state.borrow_mut().bank_dir[0] = 1.0;
            } else if i > robot_i {
                robot.state.borrow_mut().bank_dir[2] = 1.0;
            }
            if j < robot_j {
                robot.state.borrow_mut().bank_dir[3] = 1.0;
            } else if j > robot_j {
                robot.state.borrow_mut().bank_dir[1] = 1.0;
            }
        };

    if let Some((i, j)) = closest_coin {
        match robot.closest_coin {
            None => {
                robot.closest_coin = Some((i, j));
                robot.state.borrow_mut().coin_dir = [0.0; 4];
                update_coin_dir_fn(i, robot_i, j, robot_j, robot);
            }
            Some((curr_coin_i, curr_coin_j)) => {
                let curr_dist = dist_from_robot(robot_i, robot_j, curr_coin_i, curr_coin_j);
                let new_dist = dist_from_robot(robot_i, robot_j, i, j);
                if new_dist < curr_dist {
                    robot.closest_coin = Some((i, j));
                    robot.state.borrow_mut().coin_dir = [0.0; 4];
                    update_coin_dir_fn(i, robot_i, j, robot_j, robot);
                }
            }
        }
        // update adjacency
        let (i, j) = robot.closest_coin.unwrap();
        if (robot_i as i32 - i as i32).abs() + (robot_j as i32 - j as i32).abs() == 1 {
            match (robot_i as i32 - i as i32, robot_j as i32 - j as i32) {
                (1, 0) => robot.state.borrow_mut().coin_adj[0] = 1.0,
                (0, -1) => robot.state.borrow_mut().coin_adj[1] = 1.0,
                (-1, 0) => robot.state.borrow_mut().coin_adj[2] = 1.0,
                (0, 1) => robot.state.borrow_mut().coin_adj[3] = 1.0,
                _ => {}
            }
        }
    }

    if let Some((i, j)) = closest_bank {
        match robot.closest_bank {
            None => {
                robot.closest_bank = Some((i, j));
                robot.state.borrow_mut().bank_dir = [0.0; 4];
                update_bank_dir_fn(i, robot_i, j, robot_j, robot);
            }
            Some((curr_bank_i, bank_j)) => {
                let curr_dist = dist_from_robot(robot_i, robot_j, curr_bank_i, bank_j);
                let new_dist = dist_from_robot(robot_i, robot_j, i, j);
                if new_dist < curr_dist {
                    robot.closest_bank = Some((i, j));
                    robot.state.borrow_mut().bank_dir = [0.0; 4];
                    update_bank_dir_fn(i, robot_i, j, robot_j, robot);
                }
            }
        }
        // update adjacency
        let (i, j) = robot.closest_bank.unwrap();
        if (robot_i as i32 - i as i32).abs() + (robot_j as i32 - j as i32).abs() == 1 {
            match (robot_i as i32 - i as i32, robot_j as i32 - j as i32) {
                (1, 0) => robot.state.borrow_mut().bank_adj[0] = 1.0,
                (0, -1) => robot.state.borrow_mut().bank_adj[1] = 1.0,
                (-1, 0) => robot.state.borrow_mut().bank_adj[2] = 1.0,
                (0, 1) => robot.state.borrow_mut().bank_adj[3] = 1.0,
                _ => {}
            }
        }
    }
}

pub fn update_danger(robot: &mut GymRobot, world: &mut World) {
    // adj tiles
    let surroundings = robot_view(robot, world);
    // reset the state
    robot.state.borrow_mut().danger = [0.0; 4];
    // update the danger vec for borders and unwalkable tiles
    for (i, row) in surroundings.iter().enumerate() {
        for (j, tile) in row.iter().enumerate() {
            // even manhattan distance -> diagonal or under the robot
            if ((i as i32 - 1).abs() + (j as i32 - 1).abs()) % 2 == 0 {
                continue;
            }
            match tile {
                // border -> that direction must be set to 1.0 (danger)
                None => update_danger_adj(robot, i, j),
                // we have to check if that tile is walkable and the robot has enough energy for it
                Some(tile) => {
                    let tiletype = &tile.tile_type;
                    // danger for non walkable tiles || cost is too high
                    let environmental_conditions = look_at_sky(world);
                    let new_elevation = tile.elevation;
                    let current_elevation = surroundings[1][1].as_ref().unwrap().elevation;

                    // Calculate cost
                    let base_cost = calculate_cost_go_with_environment(
                        tiletype.properties().cost(),
                        environmental_conditions,
                        tiletype.clone(),
                    );
                    // Consider elevation cost only if we are going from a lower tile to a higher tile
                    let cost = if new_elevation > current_elevation {
                        base_cost + (new_elevation - current_elevation).pow(2)
                    } else {
                        base_cost
                    };
                    if !tiletype.properties().walk() || robot.get_energy().get_energy_level() < cost
                    {
                        update_danger_adj(robot, i, j);
                    }
                }
            }
        }
    }
}

fn update_danger_adj(robot: &mut GymRobot, i: usize, j: usize) {
    let mut state = robot.state.borrow_mut();
    match (i, j) {
        // danger up
        (0, 1) => state.danger[0] = 1.0,
        // danger right
        (1, 2) => state.danger[1] = 1.0,
        // danger down
        (2, 1) => state.danger[2] = 1.0,
        // danger left
        (1, 0) => state.danger[3] = 1.0,
        _ => {}
    }
}

fn dist_from_robot(r_i: usize, r_j: usize, i: usize, j: usize) -> f64 {
    ((r_i as f64 - i as f64).powi(2) + (r_j as f64 - j as f64).powi(2)).sqrt()
}

pub fn scan_reward(
    robot: &mut GymRobot,
    rect: Vec<Vec<Tile>>,
    dir: Direction,
    world: &mut World,
) -> f64 {
    let (mut n_coins, mut n_banks) = (0, 0);
    let mut journal = JourneyJournal::new(&[], &CONTENT_TARGETS, false);

    let coins = journal
        .contents_list_coords(&Content::Coin(0), world)
        .unwrap();
    let banks = journal
        .contents_list_coords(&Content::Bank(0..0), world)
        .unwrap();

    let robot_i = robot.get_coordinate().get_row() as i64;
    let robot_j = robot.get_coordinate().get_col() as i64;

    let row_len = rect.len() as i64;
    for (i, row) in rect.iter().enumerate() {
        let col_len = row.len() as i64;
        for (j, tile) in row.iter().enumerate() {
            let (relative_i, relative_j) = match dir {
                // i offset is -row_len + i - 1
                // j offset is -1
                Direction::Up => (-row_len + i as i64 - 1, j as i64 - 1),
                // i offset is -1
                // j offset is col_len - j + 1
                Direction::Right => (i as i64 - 1, col_len - j as i64 + 1),
                // i offset is row_len - i + 1
                // j offset is -1
                Direction::Down => (row_len - i as i64 + 1, j as i64 - 1),
                // i offset is -1
                // j offset is -row_len + j - 1
                Direction::Left => (i as i64 - 1, -row_len + j as i64 - 1),
            };
            let coord = (
                (robot_i + relative_i) as usize,
                (robot_j + relative_j) as usize,
            );
            if let Content::Coin(_) = tile.content {
                if !coins.contains(&coord) {
                    n_coins += 1;
                }
            }
            if let Content::Bank(_) = tile.content {
                if !banks.contains(&coord) {
                    n_banks += 1;
                }
            }
        }
    }
    // TODO
    0.
}

pub fn destroy_reward(amount: usize) -> f64 {
    // TODO
    0.
}

pub fn put_reward(amount: usize) -> f64 {
    // TODO
    0.
}
