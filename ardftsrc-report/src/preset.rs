//! The four `ardftsrc` quality presets, shared by the `thdn` and `hydrogen_src` report
//! stages (and by [`crate::report`], which joins their per-preset output).

use ardftsrc::{Config, PRESET_EXTREME, PRESET_FAST, PRESET_GOOD, PRESET_HIGH};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    Fast,
    Good,
    High,
    Extreme,
}

impl Preset {
    pub const ALL: [Preset; 4] = [Preset::Fast, Preset::Good, Preset::High, Preset::Extreme];

    pub fn label(self) -> &'static str {
        match self {
            Preset::Fast => "fast",
            Preset::Good => "good",
            Preset::High => "high",
            Preset::Extreme => "extreme",
        }
    }

    pub fn base_config(self) -> Config {
        match self {
            Preset::Fast => PRESET_FAST,
            Preset::Good => PRESET_GOOD,
            Preset::High => PRESET_HIGH,
            Preset::Extreme => PRESET_EXTREME,
        }
    }
}
