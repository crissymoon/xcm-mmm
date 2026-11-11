//! Run MOEA/D on ZDT1 and score the front against the true front.
//!
//! cargo run --release --example moead_front

use xcm_mmm::benchmarks::{Benchmark, Kind};
use xcm_mmm::metrics::{hypervolume, igd, reference_front, spacing};
use xcm_mmm::moead::{minimize, MoeadConfig};

fn main() {
    let problem = Benchmark::build(Kind::Zdt1, 30).unwrap();
    let config = MoeadConfig {
        population: 100,
        budget: 50_000,
        seed: 1,
        ..MoeadConfig::default()
    };
    let out = minimize(&problem, &config);
    let front = out.front();
    let reference = reference_front(Kind::Zdt1, 500);
    println!("subproblems {}", out.population);
    println!("points      {}", front.len());
    println!("igd         {:.6}", igd(&front, &reference));
    println!("hypervolume {:.6}", hypervolume(&front, &[1.1, 1.1]));
    println!("spacing     {:.6}", spacing(&front));
    println!("\n f1        f2");
    for point in front.iter().take(8) {
        println!("{:.5}   {:.5}", point[0], point[1]);
    }
    if front.len() > 8 {
        println!("... {} more", front.len() - 8);
    }
}
