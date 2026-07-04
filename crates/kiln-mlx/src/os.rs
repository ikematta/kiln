//! Process-level OS facilities that need `libc` — here because unsafe is
//! confined to this crate (CLAUDE.md), like [`crate::io`].

#![allow(unsafe_code)]

/// Raises the soft `RLIMIT_NOFILE` to at least `min` (capped at the hard
/// limit) and returns the effective soft limit. Workers call this at startup
/// (CLAUDE.md sharp edge: mmap'd slabs + sockets on macOS).
pub fn raise_nofile_limit(min: u64) -> std::io::Result<u64> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: valid pointer to an rlimit out-param.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    if limit.rlim_cur >= min {
        return Ok(limit.rlim_cur);
    }
    limit.rlim_cur = min.min(limit.rlim_max);
    // SAFETY: valid pointer; only the soft limit is raised, never past hard.
    if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(limit.rlim_cur)
}

/// Current resident set size of this process in bytes (0 if unavailable) —
/// feeds `MemoryReport.process_rss_bytes` (SPEC §2.3).
pub fn process_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        // SAFETY: buffer sized for proc_taskinfo; the call writes at most
        // `size` bytes and returns how many it wrote.
        let written = unsafe {
            libc::proc_pidinfo(
                std::process::id() as libc::c_int,
                libc::PROC_PIDTASKINFO,
                0,
                (&raw mut info).cast(),
                size,
            )
        };
        if written == size {
            return info.pti_resident_size;
        }
        0
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn nofile_limit_is_at_least_requested() {
        let effective = super::raise_nofile_limit(1024).expect("getrlimit/setrlimit");
        assert!(effective >= 1024);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rss_is_nonzero_on_macos() {
        assert!(super::process_rss_bytes() > 0);
    }
}
