//! Combined THD+N, pre-ringing, and HydrogenAudio quality reporting for ardftsrc's `f64`
//! resamplers.
//!
//! See [`thdn`] for the THD+N sweep/analysis, [`preringing`] for impulse pre-ringing and
//! below-rolloff leakage, [`hydrogen`] for the HydrogenAudio Test Suite stage, and
//! [`report`] for the combined Markdown renderer.
pub mod git;
pub mod hydrogen;
pub mod preringing;
pub mod preset;
pub mod report;
pub mod thdn;
