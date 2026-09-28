use std::env;
use std::fs;

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed.max(1) } }
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x; x
    }
    fn usize(&mut self, n: usize) -> usize { (self.next() as usize) % n.max(1) }
    fn unit(&mut self) -> f64 { (self.next() as f64) / (u64::MAX as f64) }
}

fn read_matrix(path: &str, d: usize) -> Vec<f64> {
    let text = fs::read_to_string(path).expect("evidence file");
    let values: Vec<f64> = text.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    assert_eq!(values.len(), d*d, "evidence matrix has wrong size"); values
}

fn source_count(m: &[u64], d: usize) -> usize {
    (0..d).filter(|&s| m[s] == (1u64 << s)).count()
}

fn score(m: &[u64], b: &[f64], d: usize, prior: f64) -> f64 {
    let mut total = -(source_count(m, d) as f64) * prior;
    for i in 0..d { for j in (i+1)..d {
        if m[i] & m[j] != 0 { total += b[i*d+j]; }
    }} total
}

fn delta_toggle(m: &[u64], b: &[f64], d: usize, node: usize, bit: u64, add: bool) -> f64 {
    let old = m[node]; let new = if add { old | bit } else { old & !bit };
    let mut delta = 0.0;
    for other in 0..d { if other == node { continue; }
        let before = if old & m[other] != 0 { b[other.min(node)*d + other.max(node)] } else { 0.0 };
        let after = if new & m[other] != 0 { b[other.min(node)*d + other.max(node)] } else { 0.0 };
        delta += after - before;
    }
    delta
}

fn propose_special(m: &[u64], rng: &mut Rng, d: usize, kind: usize) -> Option<Vec<u64>> {
    let mut out = m.to_vec();
    if kind == 0 { // create a singleton source at a non-source node
        let candidates: Vec<usize> = (0..d).filter(|&i| m[i] != (1u64 << i)).collect();
        if candidates.is_empty() { return None; }
        let t = candidates[rng.usize(candidates.len())]; out[t] = 1u64 << t; return Some(out);
    }
    if kind == 1 { // delete a source whose node can retain another source label
        let candidates: Vec<usize> = (0..d).filter(|&s| m[s] == (1u64 << s) && m.iter().any(|&x| x & !(1u64 << s) != 0)).collect();
        if candidates.is_empty() { return None; }
        let s = candidates[rng.usize(candidates.len())];
        let replacement = (0..d).find(|&x| x != s && m[x] != 0)?;
        for x in 0..d { if x == s { out[x] = 1u64 << replacement; } else if out[x] & (1u64 << s) != 0 { out[x] &= !(1u64 << s); if out[x] == 0 { out[x] = 1u64 << replacement; } } }
        return Some(out);
    }
    if kind == 2 { // merge source labels
        let sources: Vec<usize> = (0..d).filter(|&s| m[s] == (1u64 << s)).collect();
        if sources.len() < 2 { return None; }
        let a = sources[0]; let b = sources[1];
        for x in 0..d { if x == a { out[x] = 1u64 << a; } else if out[x] & (1u64 << b) != 0 { out[x] |= 1u64 << a; out[x] &= !(1u64 << b); if out[x] == 0 { out[x] = 1u64 << a; } } }
        return Some(out);
    }
    // split a source label by creating a new source and attaching it to one node.
    let sources: Vec<usize> = (0..d).filter(|&s| m[s] == (1u64 << s)).collect();
    if sources.is_empty() { return None; }
    let candidates: Vec<usize> = (0..d).filter(|&i| m[i] != (1u64 << i)).collect();
    if candidates.is_empty() { return None; }
    let t = candidates[rng.usize(candidates.len())]; let s = sources[rng.usize(sources.len())];
    out[t] = 1u64 << t;
    let attach = (0..d).find(|&x| x != t && (m[x] & (1u64 << s)) != 0)?;
    out[attach] |= 1u64 << t; Some(out)
}

fn random_state(rng: &mut Rng, d: usize) -> Vec<u64> {
    let k = 1 + rng.usize(d);
    let mut m = vec![0u64; d];
    for s in 0..k { m[s] = 1u64 << s; }
    for i in k..d { m[i] = 1u64 << rng.usize(k); for s in 0..k { if rng.unit() < 0.25 { m[i] |= 1u64 << s; } } }
    m
}

fn run_restart(b: &[f64], d: usize, seed: u64, budget: usize, anneal: bool, prior: f64) -> (Vec<u64>, f64, usize, usize, usize, usize) {
    let mut rng = Rng::new(seed); let mut m = random_state(&mut rng, d);
    let mut best = m.clone(); let mut cur = score(&m, b, d, prior); let mut best_score = cur;
    let mut accepted = 0; let mut proposed = 0; let mut incremental = 0; let mut full = 1;
    for step in 0..budget {
        proposed += 1;
        let special = step % 11 == 0;
        let candidate = if special { propose_special(&m, &mut rng, d, (step / 11) % 4) } else { None };
        let (candidate, delta, inc) = if let Some(c) = candidate {
            (c.clone(), score(&c,b,d,prior) - cur, 0usize)
        } else {
            let candidates: Vec<usize> = (0..d).filter(|&i| m[i] != (1u64 << i)).collect();
            if candidates.is_empty() { continue; }
            let node = candidates[rng.usize(candidates.len())]; let source = rng.usize(d); let bit = 1u64 << source;
            if m[node] == bit { continue; }
            let add = m[node] & bit == 0;
            if !add && m[node].count_ones() <= 1 { continue; }
            let mut c = m.clone(); if add { c[node] |= bit; } else { c[node] &= !bit; }
            let delta = delta_toggle(&m, b, d, node, bit, add)
                - prior * ((source_count(&c,d) as f64) - (source_count(&m,d) as f64));
            (c, delta, d-1)
        };
        if special { full += 1; }
        incremental += inc;
        let temp = if anneal { 0.25 * (1.0 - (step as f64)/(budget.max(1) as f64)) + 1e-6 } else { 0.0 };
        if delta >= 0.0 || (anneal && rng.unit() < (delta/temp).exp().min(1.0)) {
            m = candidate;
            cur += delta; accepted += 1;
            if cur > best_score { best_score = cur; best = m.clone(); }
        }
    }
    (best, best_score, accepted, proposed, incremental, full)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args[1].parse().unwrap(); let path = &args[2];
    let seed: u64 = args[3].parse().unwrap(); let restarts: usize = args[4].parse().unwrap();
    let budget: usize = args[5].parse().unwrap(); let anneal = args[6] == "anneal";
    let prior: f64 = args[7].parse().unwrap(); let b = read_matrix(path, d);
    let mut winner = vec![0u64; d]; let mut winner_score = f64::NEG_INFINITY;
    let mut accepted=0; let mut proposed=0; let mut incremental=0; let mut full=0;
    for r in 0..restarts { let (m,s,a,p,e,f)=run_restart(&b,d,seed+r as u64,budget,anneal,prior);
        if s > winner_score { winner=m; winner_score=s; } accepted+=a; proposed+=p; incremental+=e; full+=f; }
    let masks = winner.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    println!("masks={}", masks); println!("score={}", winner_score);
    println!("source_count={}", source_count(&winner,d)); println!("accepted={}", accepted);
    println!("proposed={}", proposed); println!("incremental_evaluations={}", incremental);
    println!("full_evaluations={}", full);
}
