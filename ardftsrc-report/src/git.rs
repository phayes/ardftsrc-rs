use std::process::Command;

/// The repo's current revision, for display in Markdown reports only (kept out of JSON
/// reports so it doesn't perturb a checked-in regression baseline on every commit).
/// Prefers a `v*` tag that points exactly at `HEAD`; falls back to the short commit hash.
/// Returns `None` if `git` isn't installed or the cwd isn't inside a git repo.
pub fn git_revision() -> Option<String> {
    let head = Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;
    if !head.status.success() {
        return None;
    }
    let head_hash = String::from_utf8(head.stdout).ok()?.trim().to_string();
    if head_hash.is_empty() {
        return None;
    }

    if let Ok(tags) = Command::new("git").args(["tag", "--points-at", "HEAD"]).output()
        && tags.status.success()
        && let Ok(tag_list) = String::from_utf8(tags.stdout)
        && let Some(v_tag) = tag_list.lines().map(str::trim).find(|t| t.starts_with('v'))
    {
        return Some(v_tag.to_string());
    }

    if let Ok(short) = Command::new("git").args(["rev-parse", "--short", "HEAD"]).output()
        && short.status.success()
        && let Ok(s) = String::from_utf8(short.stdout)
        && !s.trim().is_empty()
    {
        return Some(s.trim().to_string());
    }

    Some(head_hash)
}
