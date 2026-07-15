//! Runtime cache detection and analytical blocking-parameter selection for
//! the blocked matrix multiplication.
//!
//! Implements the analytical model from Low et al., *"Analytical Modeling Is
//! Enough for High-Performance BLIS"* (ACM TOMS 2016), as popularised by the
//! `gemm`/`faer` crates: instead of hand-tuning the `kc`/`mc`/`nc` blocking
//! parameters per machine, derive them from the cache hierarchy so that
//!
//! - one `MR × kc` A micropanel plus one `kc × NR` B micropanel fit in L1
//!   without evicting each other (sets `kc`),
//! - the packed `mc × kc` A block stays resident in L2 while B micropanels
//!   stream through it (sets `mc`),
//! - the packed `kc × nc` B block stays resident in L3 (sets `nc`).
//!
//! Cache geometry is detected once, trying in order:
//!
//! 1. `/sys/devices/system/cpu` on Linux (most precise: reports the actual
//!    set of threads sharing each cache, and sees every core of hybrid
//!    P/E-core parts),
//! 2. the CPUID deterministic cache parameters (leaf `0x4` on Intel, leaf
//!    `0x8000001D` on AMD) on any OS — this module is x86-only, so the
//!    instruction is always available,
//! 3. conservative defaults.
//!
//! Sizes of caches shared between SMT threads are divided by the number of
//! sharers (each thread can only count on its share); L3 is kept whole since
//! the packed B block is read-shared.

#[cfg(target_arch = "x86")]
use std::arch::x86::{__cpuid_count, CpuidResult};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__cpuid_count, CpuidResult};

use std::sync::OnceLock;

/// Geometry of one cache level.
#[derive(Debug, Copy, Clone)]
pub(crate) struct CacheInfo {
    pub(crate) associativity: usize,
    pub(crate) cache_bytes: usize,
    pub(crate) cache_line_bytes: usize,
}

/// Blocking parameters for the macrokernel loops.
#[derive(Debug, Copy, Clone)]
pub(crate) struct KernelParams {
    pub(crate) kc: usize,
    pub(crate) mc: usize,
    pub(crate) nc: usize,
}

/// Conservative fallback (small x86_64 core: 32K L1d, 256K L2, 2M L3).
const CACHE_INFO_DEFAULT: [CacheInfo; 3] = [
    CacheInfo {
        associativity: 8,
        cache_bytes: 32 * 1024,
        cache_line_bytes: 64,
    },
    CacheInfo {
        associativity: 8,
        cache_bytes: 256 * 1024,
        cache_line_bytes: 64,
    },
    CacheInfo {
        associativity: 8,
        cache_bytes: 2 * 1024 * 1024,
        cache_line_bytes: 64,
    },
];

/// Parses a `/sys` cache size string like `48K`, `2M` or a plain byte count.
#[cfg(target_os = "linux")]
fn parse_size(s: &str) -> Option<usize> {
    if let Some(v) = s.strip_suffix('G') {
        Some(v.parse::<usize>().ok()? * 1024 * 1024 * 1024)
    } else if let Some(v) = s.strip_suffix('M') {
        Some(v.parse::<usize>().ok()? * 1024 * 1024)
    } else if let Some(v) = s.strip_suffix('K') {
        Some(v.parse::<usize>().ok()? * 1024)
    } else {
        s.parse::<usize>().ok()
    }
}

/// Counts CPUs in a `shared_cpu_list` string like `0-1,8` (→ 3).
#[cfg(target_os = "linux")]
fn parse_shared_count(s: &str) -> Option<usize> {
    let mut count = 0;
    for item in s.split(',') {
        if let Some((start, end)) = item.split_once('-') {
            let start = start.trim().parse::<usize>().ok()?;
            let end = end.trim().parse::<usize>().ok()?;
            count += end + 1 - start;
        } else {
            count += 1;
        }
    }
    Some(count)
}

/// Reads data/unified cache geometry for levels 1–3 from
/// `/sys/devices/system/cpu/cpu*/cache/index*`.
///
/// L1/L2 sizes are divided by the number of SMT threads sharing the cache
/// (each thread can only count on its share); L3 is kept whole since the
/// packed B block is read-shared. When several CPU types report different
/// geometry (hybrid P/E-core parts), the entry with the largest line size
/// wins, matching the `gemm` crate's behaviour.
#[cfg(target_os = "linux")]
fn cache_info_linux() -> Option<[CacheInfo; 3]> {
    use std::fs;

    let mut all_info = [CacheInfo {
        associativity: 8,
        cache_bytes: 0,
        cache_line_bytes: 64,
    }; 3];
    let mut found = false;

    for cpu in fs::read_dir("/sys/devices/system/cpu").ok()? {
        let cpu = cpu.ok()?.path();
        let is_cpu_dir = cpu
            .file_name()
            .and_then(|f| f.to_str())
            .is_some_and(|name| {
                name.strip_prefix("cpu")
                    .is_some_and(|n| n.parse::<usize>().is_ok())
            });
        let cache_dir = cpu.join("cache");
        if !is_cpu_dir || !cache_dir.is_dir() {
            continue;
        }

        for index in fs::read_dir(cache_dir).ok()? {
            let index = index.ok()?.path();
            let read = |file: &str| -> Option<String> {
                Some(fs::read_to_string(index.join(file)).ok()?.trim().to_owned())
            };

            let Some(cache_type) = read("type") else {
                continue;
            };
            if !matches!(cache_type.as_str(), "Data" | "Unified") {
                continue;
            }
            let Some(level) = read("level").and_then(|s| s.parse::<usize>().ok()) else {
                continue;
            };
            if !(1..=3).contains(&level) {
                continue;
            }

            let Some(cache_bytes) = read("size").and_then(|s| parse_size(&s)) else {
                continue;
            };
            let associativity = read("ways_of_associativity")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(8);
            let cache_line_bytes = read("coherency_line_size")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(64);
            let shared_count = if level == 3 {
                1
            } else {
                read("shared_cpu_list")
                    .and_then(|s| parse_shared_count(&s))
                    .unwrap_or(1)
                    .max(1)
            };

            let entry = &mut all_info[level - 1];
            if cache_line_bytes >= entry.cache_line_bytes {
                entry.associativity = associativity;
                entry.cache_line_bytes = cache_line_bytes;
                entry.cache_bytes = cache_bytes / shared_count;
                found = true;
            }
        }
    }

    if !found {
        return None;
    }
    for (info, default) in all_info.iter_mut().zip(CACHE_INFO_DEFAULT) {
        if info.cache_bytes == 0 {
            *info = default;
        }
    }
    Some(all_info)
}

#[cfg(not(target_os = "linux"))]
fn cache_info_linux() -> Option<[CacheInfo; 3]> {
    None
}

/// Reads data/unified cache geometry for levels 1–3 from the CPUID
/// deterministic cache parameters — the OS-independent fallback.
///
/// Intel reports them on leaf `0x4`; AMD (Zen and newer) on the identically
/// laid out extended leaf `0x8000001D`. Each subleaf describes one cache:
/// type, level, ways, line size and set count, plus the number of logical
/// processors sharing it (used to divide L1/L2 into per-thread shares,
/// mirroring the `/sys` path).
///
/// One caveat on hybrid P/E-core parts: CPUID describes the core executing
/// the query, so the result matches whichever core type detection ran on
/// (the `/sys` path, preferred on Linux, sees all core types instead).
fn cache_info_cpuid() -> Option<[CacheInfo; 3]> {
    // Pick the leaf: Intel's 0x4 if it reports at least one cache, else
    // AMD's 0x8000001D when the extended range reaches it.
    let leaf = if __cpuid_count(0, 0).eax >= 0x4 && __cpuid_count(0x4, 0).eax & 0x1f != 0 {
        0x4
    } else if __cpuid_count(0x8000_0000, 0).eax >= 0x8000_001D {
        0x8000_001D
    } else {
        return None;
    };

    let mut all_info = [CacheInfo {
        associativity: 8,
        cache_bytes: 0,
        cache_line_bytes: 64,
    }; 3];
    let mut found = false;

    // Real CPUs list well under a dozen caches; the bound only guards
    // against a hypothetical enumeration that never terminates.
    for subleaf in 0..64 {
        let CpuidResult { eax, ebx, ecx, .. } = __cpuid_count(leaf, subleaf);

        // EAX[4:0]: cache type — 0 = end of list, 1 = data, 2 = instruction,
        // 3 = unified.
        let cache_type = eax & 0x1f;
        if cache_type == 0 {
            break;
        }
        let level = ((eax >> 5) & 0x7) as usize;
        if !matches!(cache_type, 1 | 3) || !(1..=3).contains(&level) {
            continue;
        }

        let ways = ((ebx >> 22) & 0x3ff) as usize + 1;
        let partitions = ((ebx >> 12) & 0x3ff) as usize + 1;
        let cache_line_bytes = (ebx & 0xfff) as usize + 1;
        let sets = ecx as usize + 1;
        let cache_bytes = ways * partitions * cache_line_bytes * sets;

        // EAX[25:14]: logical processors sharing this cache, minus one
        // (rounded up to a power of two — close enough for the per-thread
        // share).
        let shared_count = if level == 3 {
            1
        } else {
            ((eax >> 14) & 0xfff) as usize + 1
        };

        all_info[level - 1] = CacheInfo {
            associativity: ways,
            cache_bytes: cache_bytes / shared_count,
            cache_line_bytes,
        };
        found = true;
    }

    if !found {
        return None;
    }
    for (info, default) in all_info.iter_mut().zip(CACHE_INFO_DEFAULT) {
        if info.cache_bytes == 0 {
            *info = default;
        }
    }
    Some(all_info)
}

/// Detected L1/L2/L3 data-cache geometry (detection runs once): `/sys` on
/// Linux, then CPUID, then [`CACHE_INFO_DEFAULT`].
pub(crate) fn cache_info() -> &'static [CacheInfo; 3] {
    static CACHE_INFO: OnceLock<[CacheInfo; 3]> = OnceLock::new();
    CACHE_INFO.get_or_init(|| {
        cache_info_linux()
            .or_else(cache_info_cpuid)
            .unwrap_or(CACHE_INFO_DEFAULT)
    })
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn round_down(a: usize, b: usize) -> usize {
    a / b * b
}

/// Derives `kc`/`mc`/`nc` for an `m × n × k` product with an `mr × nr`
/// microkernel over `sizeof`-byte scalars, per the TOMS analytical model.
///
/// The final `kc`/`mc`/`nc` are additionally balanced so the last block of
/// each loop is not degenerately small (e.g. `k = 513` becomes two blocks of
/// ~257 rather than 512 + 1).
pub(crate) fn kernel_params(
    m: usize,
    n: usize,
    k: usize,
    mr: usize,
    nr: usize,
    sizeof: usize,
) -> KernelParams {
    if m == 0 || n == 0 || k == 0 {
        return KernelParams {
            kc: k.max(1),
            mc: m.max(mr),
            nc: n.max(nr),
        };
    }

    let info = cache_info();

    let l1_cache_bytes = info[0].cache_bytes.max(32 * 1024);
    let l2_cache_bytes = info[1].cache_bytes;
    let l3_cache_bytes = info[2].cache_bytes;

    let l1_line_bytes = info[0].cache_line_bytes.max(64);
    let l1_assoc = info[0].associativity.max(2);
    let l2_assoc = info[1].associativity.max(2);
    let l3_assoc = info[2].associativity.max(2);

    let l1_n_sets = l1_cache_bytes / (l1_line_bytes * l1_assoc);

    // kc: successive A micropanels (mr × kc) must map to *all* L1 sets so a
    // new micropanel exactly evicts the previous one, i.e. mr×kc×sizeof must
    // be a multiple of line_bytes×n_sets. Of the l1_assoc ways per set,
    // c_lhs go to the A micropanel and c_rhs to the B micropanel; the rest
    // stay free for C and loop temporaries.
    let gcd_a = gcd(mr * sizeof, l1_line_bytes * l1_n_sets);
    let kc_0 = (l1_line_bytes * l1_n_sets) / gcd_a;
    let c_lhs = (mr * sizeof) / gcd_a;
    let c_rhs = (nr * kc_0 * sizeof) / (l1_line_bytes * l1_n_sets);
    let kc_multiplier = (l1_assoc / (c_lhs + c_rhs)).max(1);

    // The 512 floor amortizes the per-kernel-call C tile load/store over a
    // deeper accumulation, matching the gemm crate. Measured on f64 too:
    // scaling the floor down by element size (256 for f64, keeping the same
    // byte budget) was neutral-to-slower — the deeper accumulation beats
    // the extra L1 pressure of the wider B micropanel.
    let auto_kc = (kc_0 * kc_multiplier.next_power_of_two()).max(512).min(k);
    // Balance the k blocks: ceil-divide k by the block count.
    let k_iter = k.div_ceil(auto_kc);
    let auto_kc = k.div_ceil(k_iter);

    // mc: the packed mc × kc A block must stay L2-resident while B
    // micropanels (nr × kc, ~1 way) stream through, keeping one way free.
    let auto_mc = {
        let rhs_micropanel_bytes = nr * auto_kc * sizeof;
        let rhs_l2_assoc = rhs_micropanel_bytes.div_ceil(l2_cache_bytes / l2_assoc);
        let lhs_l2_assoc = l2_assoc.saturating_sub(1 + rhs_l2_assoc).max(1);

        let auto_mc = round_down(
            (lhs_l2_assoc * l2_cache_bytes) / (l2_assoc * sizeof * auto_kc),
            mr,
        )
        .max(mr);
        let m_iter = m.div_ceil(auto_mc);
        m.div_ceil(m_iter * mr) * mr
    };
    // Cap mc so the A block also stays comfortably inside L2 on machines
    // with very large L2. The gemm crate uses `8 * mr` with mr = 24; with
    // our mr = 16 the equivalent row budget is 16 * mr.
    let auto_mc = auto_mc.min(16 * mr).max(mr);

    // nc: the packed kc × nc B block must stay L3-resident (all ways but
    // one, leaving room for the streaming A block).
    let auto_nc = if l3_cache_bytes == 0 {
        n.next_multiple_of(nr)
    } else {
        let rhs_l3_assoc = l3_assoc - 1;
        let rhs_macropanel_max_bytes = (rhs_l3_assoc * l3_cache_bytes) / l3_assoc;

        let auto_nc = round_down(rhs_macropanel_max_bytes / (sizeof * auto_kc), nr).max(nr);
        let n_iter = n.div_ceil(auto_nc);
        n.div_ceil(n_iter * nr) * nr
    };

    KernelParams {
        kc: auto_kc,
        mc: auto_mc,
        nc: auto_nc.max(nr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_size() {
        assert_eq!(parse_size("48K"), Some(48 * 1024));
        assert_eq!(parse_size("2M"), Some(2 * 1024 * 1024));
        assert_eq!(parse_size("1G"), Some(1024 * 1024 * 1024));
        assert_eq!(parse_size("32768"), Some(32768));
        assert_eq!(parse_size("junk"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_shared_count() {
        assert_eq!(parse_shared_count("0"), Some(1));
        assert_eq!(parse_shared_count("0-1"), Some(2));
        assert_eq!(parse_shared_count("0-3,8-11"), Some(8));
        assert_eq!(parse_shared_count("0,2,4"), Some(3));
    }

    /// CPUID detection must succeed on any x86 machine the tests run on and
    /// report plausible geometry.
    #[test]
    fn test_cache_info_cpuid_sane() {
        let info = cache_info_cpuid().expect("CPUID cache detection failed");
        println!("cpuid: {info:?}");
        println!("linux: {:?}", cache_info_linux());

        // Per-thread L1d share: real L1d caches are 16K-64K, halved by SMT.
        assert!(
            info[0].cache_bytes >= 8 * 1024,
            "L1d too small: {}",
            info[0].cache_bytes
        );
        for level in info {
            assert!(
                level.cache_line_bytes.is_power_of_two() && level.cache_line_bytes >= 32,
                "bad line size: {}",
                level.cache_line_bytes
            );
            assert!(level.associativity >= 2, "bad associativity");
            assert!(level.cache_bytes > 0, "empty cache level");
        }
    }

    #[test]
    fn test_kernel_params_sane() {
        for &(m, n, k) in &[
            (1usize, 1usize, 1usize),
            (7, 9, 13),
            (64, 64, 64),
            (513, 513, 513),
            (2048, 2048, 2048),
        ] {
            let p = kernel_params(m, n, k, 8, 8, size_of::<f32>());
            assert!(p.kc >= 1 && p.kc <= k.max(1), "kc={} for k={k}", p.kc);
            assert!(p.mc >= 8 && p.mc.is_multiple_of(8), "mc={}", p.mc);
            assert!(p.nc >= 8 && p.nc.is_multiple_of(8), "nc={}", p.nc);
        }
    }

    /// The k blocks must be balanced: k = 513 must not produce kc = 512
    /// (which would leave a degenerate block of depth 1).
    #[test]
    fn test_kernel_params_balanced_k() {
        let p = kernel_params(256, 256, 513, 8, 8, size_of::<f32>());
        let iters = 513usize.div_ceil(p.kc);
        assert!(p.kc * (iters - 1) < 513, "last k block is empty");
        assert!(
            513 - p.kc * (iters - 1) >= p.kc / 2,
            "last k block is degenerate: kc={}",
            p.kc
        );
    }
}
