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
