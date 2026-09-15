//! Lightweight, always-on regression coverage: a couple of representative cases with a
//! loose threshold, so a serious THD+N regression fails `cargo test` without paying for
//! the full sweep on every run. Run `ardftsrc-report run thdn` for the full sweep + report.

use ardftsrc_report::preset::Preset;
use ardftsrc_report::thdn::report::{HighPrecisionVariant, run_case};

#[test]
fn preset_good_44100_to_48000_1khz_is_well_below_zero_dbfs_thdn() {
    let case = run_case(44_100, 48_000, Preset::Good, HighPrecisionVariant::Off, false, 1_000.0, -1.0);
    assert!(
        case.thdn_broadband_db < -80.0,
        "thdn_broadband_db={}",
        case.thdn_broadband_db
    );
    assert!(
        case.thdn_audio_band_db < -80.0,
        "thdn_audio_band_db={}",
        case.thdn_audio_band_db
    );
    assert!(case.gain_error_db.abs() < 1.0, "gain_error_db={}", case.gain_error_db);
}

#[test]
fn preset_extreme_48000_to_44100_1khz_is_well_below_zero_dbfs_thdn() {
    let case = run_case(48_000, 44_100, Preset::Extreme, HighPrecisionVariant::Off, false, 1_000.0, -1.0);
    assert!(
        case.thdn_broadband_db < -100.0,
        "thdn_broadband_db={}",
        case.thdn_broadband_db
    );
    assert!(
        case.thdn_audio_band_db < -100.0,
        "thdn_audio_band_db={}",
        case.thdn_audio_band_db
    );
    assert!(case.gain_error_db.abs() < 1.0, "gain_error_db={}", case.gain_error_db);
}

#[test]
fn preset_good_192000_to_48000_decimated_1khz_is_well_below_zero_dbfs_thdn() {
    let case = run_case(192_000, 48_000, Preset::Good, HighPrecisionVariant::Off, true, 1_000.0, -1.0);
    assert!(
        case.thdn_broadband_db < -80.0,
        "thdn_broadband_db={}",
        case.thdn_broadband_db
    );
    assert!(
        case.thdn_audio_band_db < -80.0,
        "thdn_audio_band_db={}",
        case.thdn_audio_band_db
    );
    assert!(case.gain_error_db.abs() < 1.0, "gain_error_db={}", case.gain_error_db);
}

#[cfg(feature = "high_precision")]
fn assert_high_precision_preset_extreme_44100_to_48000_1khz_is_well_below_zero_dbfs_thdn(
    high_precision: HighPrecisionVariant,
) {
    let case = run_case(44_100, 48_000, Preset::Extreme, high_precision, false, 1_000.0, -1.0);
    assert!(
        case.thdn_broadband_db < -100.0,
        "thdn_broadband_db={}",
        case.thdn_broadband_db
    );
    assert!(
        case.thdn_audio_band_db < -100.0,
        "thdn_audio_band_db={}",
        case.thdn_audio_band_db
    );
}

#[cfg(feature = "high_precision")]
#[test]
fn double_double_preset_extreme_44100_to_48000_1khz_is_well_below_zero_dbfs_thdn() {
    assert_high_precision_preset_extreme_44100_to_48000_1khz_is_well_below_zero_dbfs_thdn(
        HighPrecisionVariant::DoubleDouble,
    );
}

#[cfg(feature = "high_precision")]
#[test]
fn f128_preset_extreme_44100_to_48000_1khz_is_well_below_zero_dbfs_thdn() {
    assert_high_precision_preset_extreme_44100_to_48000_1khz_is_well_below_zero_dbfs_thdn(HighPrecisionVariant::F128);
}

#[cfg(feature = "high_precision")]
#[test]
fn f256_preset_extreme_44100_to_48000_1khz_is_well_below_zero_dbfs_thdn() {
    assert_high_precision_preset_extreme_44100_to_48000_1khz_is_well_below_zero_dbfs_thdn(HighPrecisionVariant::F256);
}
