//! Live system-memory probe: what the machine can actually hand out RIGHT
//! NOW, as opposed to the configured fraction of installed RAM the budget
//! is cut from (SPEC §2.3).
//!
//! The budget alone is blind to memory other processes already claimed: a
//! 16 GB machine under daily-use load admitted an 11.5 GB model against a
//! 13.7 GB budget while the OS was 4.4 GB into swap, and generation ran
//! ~150-200x slower than benchmarked (2026-07-21 field finding). Admission
//! decisions therefore also consult this probe, which reports:
//!
//! - `available_bytes` — pages the OS can grant or reclaim without touching
//!   the compressor or swap: free + speculative + inactive (`vm_stat`).
//!   These three queues are disjoint (`vm_stat` prints free excluding
//!   speculative). Dirty anonymous pages inside the inactive queue still
//!   need compression to reclaim, so this slightly overstates what is
//!   painlessly available — `memory.min_available_bytes` is the guard band.
//! - `swap_used_bytes` — `sysctl vm.swapusage`. Reported for observability,
//!   deliberately NOT a gate: macOS keeps swap allocated long after the
//!   pressure that created it has passed (this dev machine idles with
//!   ~2.5 GB "used" at pressure level normal), so gating on it would refuse
//!   loads on a machine with gigabytes genuinely free.
//! - `pressure_level` — `kern.memorystatus_vm_pressure_level`, the kernel's
//!   own live signal: 1 normal, 2 warning (compressor/swap actively
//!   working), 4 critical. This is the honest version of "swap is active".
//!
//! Shelled out (`/usr/bin/vm_stat`, `/usr/sbin/sysctl`) because native
//! reads need libc and unsafe code is confined to kiln-mlx (CLAUDE.md),
//! mirroring `lifecycle::total_unified_memory` and the supervisor's
//! `/bin/kill` precedent. A probe is two process spawns (~ms); callers
//! cache and rate-limit (`Lifecycle`).

/// One point-in-time reading. All zeros are legal (a partial probe fails
/// open field-by-field); a probe that cannot establish `available_bytes`
/// returns `None` instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemMemory {
    /// Free + speculative + inactive pages, in bytes (see module docs).
    pub available_bytes: u64,
    /// Swap currently allocated by the OS (`vm.swapusage` "used").
    pub swap_used_bytes: u64,
    /// `kern.memorystatus_vm_pressure_level`: 1 normal, 2 warning,
    /// 4 critical; 0 = unavailable (treated as normal — fail open).
    pub pressure_level: u32,
}

impl SystemMemory {
    /// The kernel says the machine is already under memory pressure
    /// (warning or critical): the compressor/swap is actively working, and
    /// admitting new model bytes would compound it.
    pub fn pressure_elevated(&self) -> bool {
        self.pressure_level >= 2
    }
}

/// Reads the current system memory state; `None` when availability cannot
/// be established (callers fail open to budget-only admission and log).
pub fn probe() -> Option<SystemMemory> {
    #[cfg(target_os = "macos")]
    {
        let vm_stat = run("/usr/bin/vm_stat", &[])?;
        let available_bytes = parse_vm_stat_available(&vm_stat)?;
        let swap_used_bytes = run("/usr/sbin/sysctl", &["-n", "vm.swapusage"])
            .as_deref()
            .and_then(parse_swapusage_used)
            .unwrap_or(0);
        let pressure_level = run(
            "/usr/sbin/sysctl",
            &["-n", "kern.memorystatus_vm_pressure_level"],
        )
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
        Some(SystemMemory {
            available_bytes,
            swap_used_bytes,
            pressure_level,
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Linux is a compile-check target only (SPEC §1.2); MemAvailable is
        // the kernel's own reclaimable-without-swapping estimate.
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let field = |name: &str| -> Option<u64> {
            text.lines()
                .find_map(|line| line.strip_prefix(name))?
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<u64>()
                .ok()
                .map(|kb| kb * 1024)
        };
        Some(SystemMemory {
            available_bytes: field("MemAvailable:")?,
            swap_used_bytes: field("SwapTotal:")
                .zip(field("SwapFree:"))
                .map(|(total, free)| total.saturating_sub(free))
                .unwrap_or(0),
            pressure_level: 0,
        })
    }
}

#[cfg(target_os = "macos")]
fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Available bytes from `vm_stat` output: page size from the header line,
/// then free + speculative + inactive page counts.
fn parse_vm_stat_available(text: &str) -> Option<u64> {
    // "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
    let header = text.lines().next()?;
    let page_size: u64 = header
        .split("page size of")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let count = |prefix: &str| -> Option<u64> {
        text.lines()
            .find_map(|line| line.strip_prefix(prefix))?
            .trim()
            .trim_end_matches('.')
            .parse()
            .ok()
    };
    let pages = count("Pages free:")?
        .checked_add(count("Pages speculative:")?)?
        .checked_add(count("Pages inactive:")?)?;
    pages.checked_mul(page_size)
}

/// "used" bytes from `sysctl -n vm.swapusage`:
/// `total = 4096.00M  used = 2493.12M  free = 1602.88M  (encrypted)`.
fn parse_swapusage_used(text: &str) -> Option<u64> {
    let value = text.split("used =").nth(1)?.split_whitespace().next()?;
    let (number, unit) = value.split_at(value.len().saturating_sub(1));
    let scale: f64 = match unit {
        "K" => 1024.0,
        "M" => 1024.0 * 1024.0,
        "G" => 1024.0 * 1024.0 * 1024.0,
        "T" => 1024.0f64.powi(4),
        // No suffix: the whole token is a plain byte count.
        _ => return value.parse::<f64>().ok().map(|b| b as u64),
    };
    let number: f64 = number.parse().ok()?;
    Some((number * scale) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_plausible_numbers() {
        let mem = probe().expect("probe works on dev/CI machines");
        assert!(
            mem.available_bytes > 64 << 20,
            "implausibly little available memory: {}",
            mem.available_bytes
        );
        // 0 = unavailable is legal; real macOS levels are 1, 2, or 4.
        assert!(
            matches!(mem.pressure_level, 0 | 1 | 2 | 4),
            "unexpected pressure level: {}",
            mem.pressure_level
        );
    }

    #[test]
    fn vm_stat_available_is_free_plus_speculative_plus_inactive() {
        let sample = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                      Pages free:                                    26750.\n\
                      Pages active:                                 176330.\n\
                      Pages inactive:                               161591.\n\
                      Pages speculative:                             13625.\n\
                      Pages throttled:                                   0.\n\
                      Pages purgeable:                                1489.\n";
        assert_eq!(
            parse_vm_stat_available(sample),
            Some((26750 + 13625 + 161591) * 16384)
        );
        assert_eq!(parse_vm_stat_available("garbage"), None);
        // A missing counter is a parse failure, never a silent zero.
        assert_eq!(
            parse_vm_stat_available(
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                 Pages free: 10.\n"
            ),
            None
        );
    }

    #[test]
    fn swapusage_used_parses_units() {
        let sample = "total = 4096.00M  used = 2493.12M  free = 1602.88M  (encrypted)";
        assert_eq!(
            parse_swapusage_used(sample),
            Some((2493.12 * 1024.0 * 1024.0) as u64)
        );
        assert_eq!(
            parse_swapusage_used("total = 8.00G  used = 1.50G  free = 6.50G"),
            Some((1.5 * 1024.0 * 1024.0 * 1024.0) as u64)
        );
        assert_eq!(parse_swapusage_used("total = 0.00M  used = 0.00M"), Some(0));
        assert_eq!(parse_swapusage_used("no swap here"), None);
    }

    #[test]
    fn pressure_elevation_threshold() {
        let mem = |level| SystemMemory {
            available_bytes: 0,
            swap_used_bytes: 0,
            pressure_level: level,
        };
        assert!(!mem(0).pressure_elevated()); // unavailable fails open
        assert!(!mem(1).pressure_elevated());
        assert!(mem(2).pressure_elevated());
        assert!(mem(4).pressure_elevated());
    }
}
