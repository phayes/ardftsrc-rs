//! Lightweight, always-on regression coverage: a couple of representative cases with a
//! loose threshold, so a serious THD+N or pre-ringing regression fails `cargo test` without
//! paying for the full sweep on every run. Run `ardftsrc-report run thdn` /
//! `ardftsrc-report run preringing` for the full sweeps + reports.

use ardftsrc_report::preringing::run_config;
use ardftsrc_report::preset::Preset;
use ardftsrc_report::thdn::report::{HighPrecisionVariant, run_case};

#[test]
fn preset_good_44100_to_48000_1khz_is_well_below_zero_dbfs_thdn() {
    let case = run_case(
        44_100,
        48_000,
        Preset::Good,
        HighPrecisionVariant::Off,
        false,
        1_000.0,
        -1.0,
    );
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
    let case = run_case(
        48_000,
        44_100,
        Preset::Extreme,
        HighPrecisionVariant::Off,
        false,
        1_000.0,
        -1.0,
    );
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
    let case = run_case(
        192_000,
        48_000,
        Preset::Good,
        HighPrecisionVariant::Off,
        true,
        1_000.0,
        -1.0,
    );
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
fn preset_good_44100_to_48000_pre_ringing_stays_above_the_rolloff() {
    let result = run_config(Preset::Good, 44_100, 48_000, false);

    for transient in &result.transients {
        assert!(
            transient.pre_echo_db < -200.0,
            "carrier={} position={} pre_echo_db={}",
            transient.carrier_hz,
            transient.chunk_position,
            transient.pre_echo_db
        );
        assert!(
            transient.delay_offset_samples.abs() < 1e-6,
            "delay_offset_samples={}",
            transient.delay_offset_samples
        );
    }

    let worst = result.worst_impulse().expect("impulses were measured");
    assert!(
        worst.ringing_freq_hz > result.band.passband_edge_hz && worst.ringing_freq_hz < result.band.stopband_edge_hz,
        "ringing_freq_hz={} outside transition band {:?}",
        worst.ringing_freq_hz,
        result.band
    );
    assert!(
        worst.pre_ring_ms.iter().all(|&ms| ms < 10.0),
        "pre_ring_ms={:?}",
        worst.pre_ring_ms
    );
}

#[test]
fn preset_good_192000_to_48000_decimated_below_rolloff_pre_echo_is_small() {
    let result = run_config(Preset::Good, 192_000, 48_000, true);
    for transient in &result.transients {
        assert!(
            transient.pre_echo_db < -120.0,
            "carrier={} position={} pre_echo_db={}",
            transient.carrier_hz,
            transient.chunk_position,
            transient.pre_echo_db
        );
    }
}

#[test]
fn preset_fast_192000_to_48000_decimated_output_has_no_delay_offset() {
    // Fast's decimation cascade delays by 16.5 output samples, so whole-sample trimming alone
    // would leave the output half a sample early.
    let result = run_config(Preset::Fast, 192_000, 48_000, true);
    for transient in &result.transients {
        assert!(
            transient.delay_offset_samples.abs() < 1e-6,
            "position={} delay_offset_samples={}",
            transient.chunk_position,
            transient.delay_offset_samples
        );
    }
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
