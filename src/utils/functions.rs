use crate::utils::args::Mode;
use crate::utils::consts::{
    EVAL_PLOT_PATH, FONT_SIZE, LABEL_AREA_SIZE, PLOT_FONT, PLOT_HEIGHT, PLOT_WIDTH,
    TRAIN_PLOT_PATH, X_LABELS, Y_LABELS,
};
use plotters::prelude::{
    BitMapBackend, ChartBuilder, IntoDrawingArea, IntoFont, LineSeries, BLACK, WHITE,
};
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
