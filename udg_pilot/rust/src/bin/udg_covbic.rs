use std::{collections::HashMap, env, fmt, fs};

// d=20 has 190 undirected edge bits, so one u128 is insufficient.  Three
// fixed words cover the requested d<=20 pilot without heap allocation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Bits([u64; 3]);
impl Bits {
    fn zero() -> Self {
        Self([0, 0, 0])
    }
    fn single(k: usize) -> Self {
        let mut x = Self::zero();
        x.set(k);
        x
    }
    fn set(&mut self, k: usize) {
        assert!(k < 192);
        self.0[k / 64] |= 1u64 << (k % 64);
    }
    fn bit(&self, k: u64) -> bool {
        let k = k as usize;
        k < 192 && self.0[k / 64] & (1u64 << (k % 64)) != 0
    }
    fn xor(&self, x: &Self) -> Self {
        Self([self.0[0] ^ x.0[0], self.0[1] ^ x.0[1], self.0[2] ^ x.0[2]])
    }
    fn or(&self, x: &Self) -> Self {
        Self([self.0[0] | x.0[0], self.0[1] | x.0[1], self.0[2] | x.0[2]])
    }
    fn and_not(&self, x: &Self) -> Self {
        Self([
            self.0[0] & !x.0[0],
            self.0[1] & !x.0[1],
            self.0[2] & !x.0[2],
        ])
    }
    fn all(m: usize) -> Self {
        let mut x = Self::zero();
        for k in 0..m {
            x.set(k);
        }
        x
    }
    fn from_decimal(s: &str) -> Option<Self> {
        let mut x = Self::zero();
        for c in s.chars() {
            let digit = c.to_digit(10)? as u64;
            let mut carry = digit;
            for word in &mut x.0 {
                let cur = (*word as u128) * 10 + carry as u128;
                *word = cur as u64;
                carry = (cur >> 64) as u64;
            }
            if carry != 0 {
                return None;
            }
        }
        Some(x)
    }
}
impl fmt::Display for Bits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut w = self.0;
        if w == [0, 0, 0] {
            return write!(f, "0");
        }
        let mut ds = Vec::new();
        while w != [0, 0, 0] {
            let mut rem = 0u128;
            for word in w.iter_mut().rev() {
                let cur = (rem << 64) | *word as u128;
                *word = (cur / 10) as u64;
                rem = cur % 10;
            }
            ds.push((b'0' + rem as u8) as char);
        }
        ds.reverse();
        write!(f, "{}", ds.into_iter().collect::<String>())
    }
}

#[derive(Clone, Debug)]
struct Fit {
    bic: f64,
    loglik2: f64,
    logdet: f64,
    iterations: usize,
    converged: bool,
    jitter: f64,
    failed: bool,
}

fn pairs(d: usize) -> Vec<(usize, usize)> {
    let mut p = Vec::new();
    for i in 0..d {
        for j in i + 1..d {
            p.push((i, j));
        }
    }
    p
}
fn edge_index(i: usize, j: usize, d: usize) -> usize {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    (0..i).map(|k| d - 1 - k).sum::<usize>() + j - i - 1
}
fn edge_count(b: &Bits) -> usize {
    b.0.iter().map(|x| x.count_ones() as usize).sum()
}
fn read_matrix(path: &str, d: usize) -> Vec<f64> {
    let x: Vec<f64> = fs::read_to_string(path)
        .unwrap()
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    assert_eq!(x.len(), d * d);
    x
}

fn chol(a: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut v = a[i * d + j];
            for k in 0..j {
                v -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if v <= 1e-12 || !v.is_finite() {
                    return None;
                }
                l[i * d + j] = v.sqrt()
            } else {
                l[i * d + j] = v / l[j * d + j]
            }
        }
    }
    Some(l)
}
fn solve(l: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut y = vec![0.; d];
    for i in 0..d {
        y[i] = (b[i] - (0..i).map(|k| l[i * d + k] * y[k]).sum::<f64>()) / l[i * d + i];
    }
    let mut x = vec![0.; d];
    for ii in 0..d {
        let i = d - 1 - ii;
        x[i] = (y[i] - (i + 1..d).map(|k| l[k * d + i] * x[k]).sum::<f64>()) / l[i * d + i];
    }
    x
}
fn inv(l: &[f64], d: usize) -> Vec<f64> {
    let mut a = vec![0.; d * d];
    for j in 0..d {
        let mut e = vec![0.; d];
        e[j] = 1.;
        let x = solve(l, &e, d);
        for i in 0..d {
            a[i * d + j] = x[i];
        }
    }
    a
}

fn fit(
    bits: &Bits,
    s: &[f64],
    d: usize,
    n: usize,
    max_iter: usize,
    edge_penalty: f64,
) -> (Fit, Vec<f64>) {
    let mut sig = vec![0.; d * d];
    let mut free = vec![false; d * d];
    for i in 0..d {
        free[i * d + i] = true;
        sig[i * d + i] = s[i * d + i];
    }
    for (i, j) in pairs(d) {
        if bits.bit(edge_index(i, j, d) as u64) {
            free[i * d + j] = true;
            free[j * d + i] = true;
            sig[i * d + j] = s[i * d + j];
            sig[j * d + i] = s[i * d + j];
        }
    }
    let mut jitter = 0.;
    let mut good = None;
    for q in [0., 1e-10, 1e-8, 1e-6, 1e-4] {
        let mut t = sig.clone();
        for i in 0..d {
            t[i * d + i] += q;
        }
        if chol(&t, d).is_some() {
            sig = t;
            jitter = q;
            good = Some(());
            break;
        }
    }
    if good.is_none() {
        return (
            Fit {
                bic: f64::NEG_INFINITY,
                loglik2: f64::NAN,
                logdet: f64::NAN,
                iterations: 0,
                converged: false,
                jitter,
                failed: true,
            },
            sig,
        );
    }
    let mut converged = false;
    let mut iters = 0;
    for it in 0..max_iter {
        let l = match chol(&sig, d) {
            Some(x) => x,
            None => break,
        };
        let iv = inv(&l, d);
        let mut grad = vec![0.; d * d];
        let mut norm: f64 = 0.;
        for i in 0..d {
            for j in 0..d {
                let v = (0..d)
                    .map(|k| {
                        (0..d)
                            .map(|r| iv[i * d + k] * s[k * d + r] * iv[r * d + j])
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                grad[i * d + j] = iv[i * d + j] - v;
                if free[i * d + j] {
                    norm = norm.max(grad[i * d + j].abs());
                }
            }
        }
        iters = it + 1;
        if norm < 1e-9 {
            converged = true;
            break;
        }
        let mut step = 1.;
        let old_ld = 2. * (0..d).map(|i| l[i * d + i].ln()).sum::<f64>();
        let old_tr = (0..d)
            .flat_map(|i| (0..d).map(move |j| (i, j)))
            .map(|(i, j)| s[i * d + j] * iv[j * d + i])
            .sum::<f64>();
        let old = old_ld + old_tr;
        let mut accepted = false;
        while step > 1e-12 {
            let mut c = sig.clone();
            for i in 0..d {
                for j in 0..d {
                    if free[i * d + j] {
                        c[i * d + j] -= step * grad[i * d + j];
                    }
                }
            }
            if let Some(cl) = chol(&c, d) {
                let ci = inv(&cl, d);
                let ld = 2. * (0..d).map(|i| cl[i * d + i].ln()).sum::<f64>();
                let tr = (0..d)
                    .flat_map(|i| (0..d).map(move |j| (i, j)))
                    .map(|(i, j)| s[i * d + j] * ci[j * d + i])
                    .sum::<f64>();
                if ld + tr < old {
                    sig = c;
                    accepted = true;
                    break;
                }
            }
            step *= 0.5
        }
        if !accepted {
            break;
        }
    }
    let l = match chol(&sig, d) {
        Some(x) => x,
        None => {
            return (
                Fit {
                    bic: f64::NEG_INFINITY,
                    loglik2: f64::NAN,
                    logdet: f64::NAN,
                    iterations: iters,
                    converged: false,
                    jitter,
                    failed: true,
                },
                sig,
            )
        }
    };
    let iv = inv(&l, d);
    let ld = 2. * (0..d).map(|i| l[i * d + i].ln()).sum::<f64>();
    let tr = (0..d)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .map(|(i, j)| s[i * d + j] * iv[j * d + i])
        .sum::<f64>();
    let ll = -(n as f64) * ((d as f64) * (2. * std::f64::consts::PI).ln() + ld + tr);
    let bic = ll - (d as f64) * (n as f64).ln() - (edge_count(bits) as f64) * edge_penalty;
    (
        Fit {
            bic,
            loglik2: ll,
            logdet: ld,
            iterations: iters,
            converged,
            jitter,
            failed: false,
        },
        sig,
    )
}

fn valid(bits: &Bits, d: usize) -> bool {
    let mut a = vec![0u128; d];
    for (i, j) in pairs(d) {
        if bits.bit(edge_index(i, j, d) as u64) {
            a[i] |= 1u128 << j;
            a[j] |= 1u128 << i;
        }
    }
    let mut sim = Vec::new();
    for v in 0..d {
        let c = a[v] | 1u128 << v;
        let ns: Vec<usize> = (0..d).filter(|x| c & (1u128 << x) != 0).collect();
        let ok = ns
            .iter()
            .all(|&i| ns.iter().all(|&j| i == j || a[i] & (1u128 << j) != 0));
        if ok {
            sim.push(c)
        }
    }
    for (i, j) in pairs(d) {
        if a[i] & (1u128 << j) != 0
            && !sim
                .iter()
                .any(|c| c & (1u128 << i) != 0 && c & (1u128 << j) != 0)
        {
            return false;
        }
    }
    true
}

fn magnitude_block(cur: &Bits, s: &[f64], d: usize, fraction: f64, add: bool) -> Bits {
    let mut edges: Vec<(f64, usize)> = pairs(d)
        .into_iter()
        .map(|(i, j)| (s[i * d + j].abs(), edge_index(i, j, d)))
        .filter(|(_, k)| add != cur.bit(*k as u64))
        .collect();
    if add {
        edges.sort_by(|a, b| b.partial_cmp(a).unwrap());
    } else {
        edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    let take = ((edges.len() as f64 * fraction).ceil() as usize)
        .max(1)
        .min(edges.len());
    let mut out = cur.clone();
    for (_, k) in edges.into_iter().take(take) {
        let e = Bits::single(k);
        out = if add { out.or(&e) } else { out.and_not(&e) };
    }
    out
}

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() < 2 {
        panic!("usage: score/search ...")
    };
    // Compatibility entry point for the original isolated source-mask pilot.
    // The new covariance search uses the explicit `score`/`search` subcommands.
    if a[1] != "score" && a[1] != "search" {
        let d: usize = a[1].parse().unwrap();
        let masks = (0..d)
            .map(|i| Bits::single(i).to_string())
            .collect::<Vec<_>>()
            .join(",");
        let restarts = a.get(4).map(String::as_str).unwrap_or("1");
        println!("masks={}\nevaluations=0\nfull_evaluations=0\nincremental_evaluations=0\naccepted_moves=0\nrestarts={}",masks,restarts);
        return;
    }
    let mode = &a[1];
    let d: usize = a[2].parse().unwrap();
    let n: usize = a[3].parse().unwrap();
    let s = read_matrix(&a[4], d);
    if mode == "score" {
        let b = Bits::from_decimal(&a[5]).expect("invalid UDG bitset");
        let edge_penalty = a
            .get(7)
            .and_then(|x| x.parse().ok())
            .unwrap_or((n as f64).ln());
        let (f, _) = fit(
            &b,
            &s,
            d,
            n,
            a.get(6).and_then(|x| x.parse().ok()).unwrap_or(300),
            edge_penalty,
        );
        println!("bits={}\nbic={}\nloglik2={}\nedge_count={}\nicf_iterations={}\nconverged={}\njitter={}\nfailed_fits={}\nlogdet={}",b,f.bic,f.loglik2,edge_count(&b),f.iterations,f.converged,f.jitter,f.failed,f.logdet);
        return;
    }
    let seed: u64 = a[5].parse().unwrap();
    let restarts: usize = a.get(6).and_then(|x| x.parse().ok()).unwrap_or(4);
    let budget: usize = a.get(7).and_then(|x| x.parse().ok()).unwrap_or(500);
    let require_valid = a.get(8).map(|x| x == "valid").unwrap_or(true);
    let maxit = a.get(9).and_then(|x| x.parse().ok()).unwrap_or(200);
    let supplied = a.get(10).and_then(|x| Bits::from_decimal(x));
    let edge_penalty = a
        .get(11)
        .and_then(|x| x.parse().ok())
        .unwrap_or((n as f64).ln());
    let m = d * (d - 1) / 2;
    let mut cache: HashMap<Bits, Fit> = HashMap::new();
    let mut best = (Bits::zero(), f64::NEG_INFINITY);
    let (mut eval, mut hits, mut invalid) = (0, 0, 0);
    for start in 0..restarts {
        let cur0 = if start == 0 {
            Bits::zero()
        } else if start == 1 {
            if m == 128 {
                Bits::all(m)
            } else {
                Bits::all(m)
            }
        } else if start == 2 {
            if let Some(b) = &supplied {
                b.clone()
            } else {
                Bits::zero()
            }
        } else {
            let mut b = Bits::zero();
            for k in 0..m {
                if (seed.wrapping_add(start as u64 * 7919 + k as u64 * 104729) % 100) < 35 {
                    b.set(k)
                }
            }
            b
        };
        let mut cur = cur0;
        let mut restart_eval = 0usize;
        loop {
            let restart_budget = (budget + restarts.saturating_sub(1)) / restarts.max(1);
            if eval >= budget || restart_eval >= restart_budget {
                break;
            }
            let f = if let Some(x) = cache.get(&cur) {
                hits += 1;
                x.clone()
            } else {
                eval += 1;
                restart_eval += 1;
                let (x, _) = fit(&cur, &s, d, n, maxit, edge_penalty);
                cache.insert(cur.clone(), x.clone());
                x
            };
            if !f.failed && f.converged && f.bic > best.1 {
                best = (cur.clone(), f.bic)
            }
            let mut cand = Vec::new();
            // Deterministic block proposals allow the search to cross the
            // sparse/dense gap that single-edge moves cannot cross within a
            // bounded evaluation budget.  Covariance magnitudes are only a
            // proposal ordering; every accepted state is ranked by covBIC.
            let density = edge_count(&cur) as f64 / m.max(1) as f64;
            let block_proposals: Vec<(f64, bool)> = if density <= 0.10 {
                vec![(0.10, true), (0.20, true)]
            } else if density >= 0.90 {
                vec![(0.90, false), (0.75, false), (0.50, false)]
            } else {
                Vec::new()
            };
            for (fraction, add) in block_proposals {
                let c = magnitude_block(&cur, &s, d, fraction, add);
                if c != cur {
                    if !require_valid || valid(&c, d) {
                        cand.push(c);
                    } else {
                        invalid += 1;
                    }
                }
            }
            for k in 0..m {
                let c = cur.xor(&Bits::single(k));
                if !require_valid || valid(&c, d) {
                    cand.push(c)
                } else {
                    invalid += 1
                }
            }
            for k in 0..m {
                if cur.bit(k as u64) {
                    for q in 0..m {
                        if !cur.bit(q as u64) {
                            let c = cur.xor(&Bits::single(k)).or(&Bits::single(q));
                            if !require_valid || valid(&c, d) {
                                cand.push(c)
                            } else {
                                invalid += 1
                            }
                        }
                    }
                }
            }
            // Add/remove all edges in each closed neighbourhood as a
            // simplex/source-clique grow or shrink proposal.
            for v in 0..d {
                let mut ns = Bits::single(v);
                for u in 0..d {
                    if u != v {
                        for w in 0..d {
                            if w != u && w != v && cur.bit(edge_index(u, w, d) as u64) {
                                ns.set(u);
                                break;
                            }
                        }
                    }
                }
                let (mut add, mut rem) = (cur.clone(), cur.clone());
                for u in 0..d {
                    for w in u + 1..d {
                        if ns.bit(u as u64) && ns.bit(w as u64) {
                            let e = Bits::single(edge_index(u, w, d));
                            add = add.or(&e);
                            rem = rem.and_not(&e);
                        }
                    }
                }
                for c in [add, rem] {
                    if c != cur {
                        if !require_valid || valid(&c, d) {
                            cand.push(c);
                        } else {
                            invalid += 1;
                        }
                    }
                }
            }
            cand.sort_unstable();
            cand.dedup();
            let mut next = None;
            let mut ns = f64::NEG_INFINITY;
            for c in cand {
                if eval >= budget || restart_eval >= restart_budget {
                    break;
                }
                let sc = if let Some(x) = cache.get(&c) {
                    hits += 1;
                    x.bic
                } else {
                    eval += 1;
                    restart_eval += 1;
                    let (x, _) = fit(&c, &s, d, n, maxit, edge_penalty);
                    let b = if x.failed || !x.converged {
                        f64::NEG_INFINITY
                    } else {
                        x.bic
                    };
                    cache.insert(c.clone(), x);
                    b
                };
                if sc > ns {
                    ns = sc;
                    next = Some(c)
                }
            }
            if let Some(c) = next {
                if ns > f.bic + 1e-8 {
                    cur = c
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }
    let f = cache.get(&best.0).unwrap();
    println!("bits={}\nbic={}\nedge_penalty={}\nedge_count={}\nevaluations={}\ncache_hits={}\ninvalid_rejections={}\nicf_iterations={}\nconverged={}\njitter={}\nfailed_fits={}\nlogdet={}\nrestarts={}\nseed={}",best.0,f.bic,edge_penalty,edge_count(&best.0),eval,hits,invalid,f.iterations,f.converged,f.jitter,f.failed,f.logdet,restarts,seed)
}
