use crate::gym::robot::GymRobot;
use crate::utils::args::Mode;
use crate::utils::consts::{
    CONTENT_TARGETS, EVAL_PLOT_PATH, FONT_SIZE, LABEL_AREA_SIZE, PLOT_FONT, PLOT_HEIGHT,
    PLOT_WIDTH, TRAIN_PLOT_PATH, X_LABELS, Y_LABELS,
};
use ghost_journey_journal::JourneyJournal;
use plotters::prelude::{
    BitMapBackend, ChartBuilder, IntoDrawingArea, IntoFont, LineSeries, BLACK, WHITE,
};
use robotics_lib::interface::look_at_sky;
use robotics_lib::runner::Runnable;
use robotics_lib::utils::calculate_cost_go_with_environment;
use robotics_lib::world::tile::{Content, Tile};
use robotics_lib::world::World;
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
        Mode::Init => {
            panic!("Cannot plot in init mode")
        }
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

pub fn update_with_surroundings(
    robot: &mut GymRobot,
    surroundings: Vec<Vec<Option<Tile>>>,
    world: &mut World,
) {
    // reset the state
    robot.state.borrow_mut().reset();

    let mut journal = JourneyJournal::new(&[], &CONTENT_TARGETS);

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
    if let Some((i, j)) = closest_coin {
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
    }
    if let Some((i, j)) = closest_bank {
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
    }

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
                    let mut base_cost = tiletype.properties().cost();
                    let mut elevation_cost = 0;
                    let environmental_conditions = look_at_sky(world);
                    let new_elevation = tile.elevation;
                    let current_elevation = surroundings[1][1].as_ref().unwrap().elevation;

                    // Calculate cost
                    base_cost = calculate_cost_go_with_environment(
                        base_cost,
                        environmental_conditions,
                        tiletype.clone(),
                    );
                    // Consider elevation cost only if we are going from a lower tile to a higher tile
                    if new_elevation > current_elevation {
                        elevation_cost = (new_elevation - current_elevation).pow(2);
                    }
                    let cost = base_cost + elevation_cost;
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
