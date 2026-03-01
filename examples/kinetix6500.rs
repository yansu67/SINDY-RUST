use sindy_rs::{SINDy, STLSQ, PolynomialLibrary, FiniteDifference, TimeStep};
use ndarray::{Array1, Array2};
use rand::thread_rng;
use rand_distr::{Normal, Distribution};

fn main() {
    println!("=============================================================");
    println!("  SINDy-RS: Servo Motor Kinetix 6500 — Governing Equations  ");
    println!("=============================================================\n");

    let kt = 1.8;
    let ke = 1.8;
    let r_arm = 2.5;
    let l_arm = 0.012;
    let j_total = 0.005;
    let b_friction = 0.008;

    let dt = 0.0001;
    let t_total = 100.0; // 100 seconds per profile
    let n_steps = (t_total / dt) as usize; // 1,000,000 steps per profile

    // We will generate 10 profiles to reach 10,000,000 total samples
    // (10 profiles * 1,000,000 samples = 10M samples)
    let load_profiles: Vec<(&str, f64)> = vec![
        ("No Load",               0.0),
        ("Micro Vibrations",      0.2),
        ("Light Variable",        1.5),
        ("Medium Dynamic",        4.0),
        ("Heavy Shock",           8.0),
        ("Extreme Overload",     15.0),
        ("Chaotic Load A",        5.0),
        ("Chaotic Load B",       10.0),
        ("Resonance Frequency",   2.0),
        ("Catastrophic Jam",     25.0),
    ];

    let mut all_trajectories_x: Vec<Array2<f64>> = Vec::new();
    let mut all_trajectories_xdot: Vec<Array2<f64>> = Vec::new();
    let mut all_trajectories_u: Vec<Array2<f64>> = Vec::new();
    let mut all_time_steps: Vec<TimeStep> = Vec::new();

    println!("Generating synthetic data for {} load profiles...\n", load_profiles.len());

    let start_gen = std::time::Instant::now();

    for (profile_name, base_load) in &load_profiles {
        let subsample = 1; // Keep EVERY sample
        let n_out = n_steps / subsample;

        let mut x_data = Array2::<f64>::zeros((n_out, 3));
        let mut xdot_data = Array2::<f64>::zeros((n_out, 3));
        let mut u_data = Array2::<f64>::zeros((n_out, 2));

        let mut theta: f64 = 0.0;
        let mut omega: f64 = 0.0;
        let mut current: f64 = 0.0;
        let mut idx = 0;

        for step in 0..n_steps {
            let t = step as f64 * dt;

            // Voltage input is a complex PWM-like or sweeping sine wave
            let voltage = 24.0 * (2.0 * std::f64::consts::PI * 5.0 * t).sin() 
                        + 5.0 * (2.0 * std::f64::consts::PI * 50.0 * t).cos(); 

            // Highly complex load with multiple harmonics to simulate real factory noise
            let load_variation = base_load
                + 0.3 * base_load * (2.0 * std::f64::consts::PI * 1.5 * t).sin()
                + 0.2 * base_load * (2.0 * std::f64::consts::PI * 7.0 * t).cos()
                + 0.1 * base_load * (2.0 * std::f64::consts::PI * 23.0 * t).sin()
                + 0.05 * base_load * (2.0 * std::f64::consts::PI * 103.0 * t).cos(); // High freq chatter
                
            let t_load = load_variation.max(0.0);

            let d_theta = omega;
            let d_omega = (kt * current - b_friction * omega - t_load) / j_total;
            let d_current = (voltage - r_arm * current - ke * omega) / l_arm;

            // Inject Gaussian Sensor Noise (e.g., 1% relative noise or fixed variance)
            let mut rng = thread_rng();
            let noise_theta = Normal::new(0.0, 0.01).unwrap().sample(&mut rng); // 0.01 rad noise
            let noise_omega = Normal::new(0.0, 0.1).unwrap().sample(&mut rng);  // 0.1 rad/s noise
            let noise_curr  = Normal::new(0.0, 0.05).unwrap().sample(&mut rng); // 0.05 A noise

            if step % subsample == 0 && idx < n_out {
                x_data[[idx, 0]] = theta + noise_theta;
                x_data[[idx, 1]] = omega + noise_omega;
                x_data[[idx, 2]] = current + noise_curr;

                xdot_data[[idx, 0]] = d_theta;
                xdot_data[[idx, 1]] = d_omega;
                xdot_data[[idx, 2]] = d_current;

                u_data[[idx, 0]] = voltage;
                u_data[[idx, 1]] = t_load;

                idx += 1;
            }

            theta += d_theta * dt;
            omega += d_omega * dt;
            current += d_current * dt;
        }

        println!("  [{}] {} samples | Load = {:.1} Nm | Final ω = {:.2} rad/s | Final I = {:.2} A",
            profile_name, n_out, base_load, omega, current);

        all_trajectories_x.push(x_data);
        all_trajectories_xdot.push(xdot_data);
        all_trajectories_u.push(u_data);
        all_time_steps.push(TimeStep::Uniform(dt * subsample as f64));
    }

    let total_samples: usize = all_trajectories_x.iter().map(|a| a.nrows()).sum();
    let n_state = all_trajectories_x[0].ncols();
    let n_control = all_trajectories_u[0].ncols();

    println!("\n-------------------------------------------------------------");
    println!("  Data Summary");
    println!("-------------------------------------------------------------");
    println!("  Total Trajectories : {}", all_trajectories_x.len());
    println!("  Total Samples      : {}", total_samples);
    println!("  State Variables    : {} (θ, ω, I)", n_state);
    println!("  Control Inputs     : {} (V, T_load)", n_control);
    println!("  Matrix Dimension   : {} x {}", total_samples, n_state + n_control);
    println!("-------------------------------------------------------------\n");

    println!("Running SINDy identification...\n");

    let mut model = SINDy::new(
        Box::new(PolynomialLibrary::new(1).with_bias(false).with_interaction(false)),
        Box::new(STLSQ::new(1e-3, 0.01).unwrap()),
        Box::new(FiniteDifference::default()),
    );

    let start_fit = std::time::Instant::now();

    model.fit(
        &all_trajectories_x,
        &all_time_steps,
        Some(&all_trajectories_xdot),
        Some(&all_trajectories_u),
        Some(&["theta", "omega", "I", "V", "T_load"]),
    ).unwrap();

    let fit_duration = start_fit.elapsed();

    println!("=============================================================");
    println!("  DISCOVERED GOVERNING EQUATIONS");
    println!("=============================================================\n");
    model.print_model(4).unwrap();

    println!("\n-------------------------------------------------------------");
    println!("  Expected (True) Equations:");
    println!("-------------------------------------------------------------");
    println!("  dθ/dt = ω");
    println!("  dω/dt = ({:.1}/J)*I + ({:.3}/J)*ω + (-1/J)*T_load", kt, -b_friction);
    println!("        = {:.1}*I + {:.1}*ω + {:.1}*T_load", kt/j_total, -b_friction/j_total, -1.0/j_total);
    println!("  dI/dt = (1/L)*V + ({:.1}/L)*I + ({:.1}/L)*ω", -r_arm, -ke);
    println!("        = {:.1}*V + {:.1}*I + {:.1}*ω", 1.0/l_arm, -r_arm/l_arm, -ke/l_arm);

    println!("\n-------------------------------------------------------------");
    println!("  Motor Parameters (Kinetix 6500)");
    println!("-------------------------------------------------------------");
    println!("  Torque Constant  Kt  = {:.2} Nm/A", kt);
    println!("  Back-EMF Const.  Ke  = {:.2} V·s/rad", ke);
    println!("  Resistance       R   = {:.2} Ω", r_arm);
    println!("  Inductance       L   = {:.4} H", l_arm);
    println!("  Inertia          J   = {:.4} kg·m²", j_total);
    println!("  Friction         B   = {:.4} Nm·s/rad", b_friction);

    let score = model.score(
        &all_trajectories_x,
        &all_time_steps,
        Some(&all_trajectories_xdot),
        Some(&all_trajectories_u),
    ).unwrap();

    println!("\n=============================================================");
    println!("  PERFORMANCE METRICS");
    println!("=============================================================");
    println!("  Data Generation Time : {:.2?}", start_gen.elapsed());
    println!("  SINDy Fitting Time   : {:.2?}", fit_duration);
    println!("  MODEL ACCURACY: R²   = {:.6}", score);
    println!("  MODEL COMPLEXITY     : {} active terms", model.complexity());
    println!("=============================================================\n");

    println!("Simulating with discovered model (No Load profile)...\n");

    let sim_dt = 0.001;
    let sim_steps = 500;
    let t_sim: Vec<f64> = (0..sim_steps).map(|i| i as f64 * sim_dt).collect();

    let mut u_sim = Array2::<f64>::zeros((sim_steps, 2));
    for i in 0..sim_steps {
        let t = t_sim[i];
        u_sim[[i, 0]] = 24.0 * (2.0 * std::f64::consts::PI * 5.0 * t).sin();
        u_sim[[i, 1]] = 0.0;
    }

    let x0 = Array1::from_vec(vec![0.0, 0.0, 0.0]);
    let sim_result = model.simulate(&x0, &t_sim, Some(&u_sim)).unwrap();

    println!("  Time(s)  |  θ (rad)   |  ω (rad/s)  |  I (A)");
    println!("  ---------+------------+-------------+---------");
    for &i in &[0, 50, 100, 200, 300, 499] {
        if i < sim_steps {
            println!("  {:.4}   | {:>9.4}  | {:>10.4}  | {:>7.4}",
                t_sim[i], sim_result[[i, 0]], sim_result[[i, 1]], sim_result[[i, 2]]);
        }
    }

    println!("\n=============================================================");
    println!("  ANALYSIS COMPLETE");
    println!("=============================================================");
}
