//! Alias-floor coverage through the public resampler wrappers.
//!
//! The DSP behavior itself (folding/imaging levels, passband protection) is tested against the
//! core in `src/spectral.rs`; these tests check that every wrapper path picks up the alias floor
//! and keeps stream length/timing unchanged.

use ardftsrc::{AliasFloor, Config, InterleavedResampler, PlanarResampler};

/// Half a second of a 1 kHz tone plus a tone near the lower Nyquist (22.05 kHz).
fn signal(rate: usize, channels: usize, near_nyquist_hz: f64) -> Vec<f64> {
    let frames = rate / 2;
    let mut samples = Vec::with_capacity(frames * channels);
    for frame in 0..frames {
        let t = frame as f64 / rate as f64;
        for channel in 0..channels {
            let low = (2.0 * std::f64::consts::PI * 1_000.0 * t + channel as f64).sin();
            let high = (2.0 * std::f64::consts::PI * near_nyquist_hz * t).sin();
            samples.push(0.3 * low + 0.3 * high);
        }
    }
    samples
}

fn run_interleaved(config: Config, input: &[f64], context: Option<(Vec<f64>, Vec<f64>)>) -> Vec<f64> {
    let mut resampler = InterleavedResampler::<f64>::new(config).unwrap();
    if let Some((pre, post)) = context {
        resampler.pre(pre).unwrap();
        resampler.post(post).unwrap();
    }

    let input_size = resampler.input_buffer_size();
    let mut buffer = vec![0.0; resampler.output_buffer_size()];
    let mut output = Vec::new();
    let mut offset = 0;
    while offset + input_size <= input.len() {
        let written = resampler
            .process_chunk(&input[offset..offset + input_size], &mut buffer)
            .unwrap();
        output.extend_from_slice(&buffer[..written]);
        offset += input_size;
    }
    let written = resampler.process_chunk_final(&input[offset..], &mut buffer).unwrap();
    output.extend_from_slice(&buffer[..written]);
    let written = resampler.finalize(&mut buffer).unwrap();
    output.extend_from_slice(&buffer[..written]);

    assert!(output.iter().all(|sample| sample.is_finite()), "non-finite output");
    output
}

/// Runs the planar resampler and returns its output re-interleaved.
fn run_planar(config: Config, input: &[f64]) -> Vec<f64> {
    let channels = config.channels;
    let planes: Vec<Vec<f64>> = (0..channels)
        .map(|channel| input.iter().skip(channel).step_by(channels).copied().collect())
        .collect();

    let mut resampler = PlanarResampler::<f64>::new(config).unwrap();
    // Planar buffer sizes are totals across channels.
    let input_size = resampler.input_buffer_size() / channels;
    let mut buffers = vec![vec![0.0; resampler.output_buffer_size() / channels]; channels];
    let mut output_planes = vec![Vec::new(); channels];
    let frames = planes[0].len();

    // `written` is also a total across channels.
    let mut collect = |buffers: &mut Vec<Vec<f64>>, written: usize| {
        for (plane, buffer) in output_planes.iter_mut().zip(buffers.iter()) {
            plane.extend_from_slice(&buffer[..written / channels]);
        }
    };

    let mut offset = 0;
    while offset + input_size <= frames {
        let chunk: Vec<&[f64]> = planes.iter().map(|plane| &plane[offset..offset + input_size]).collect();
        let mut outputs: Vec<&mut [f64]> = buffers.iter_mut().map(Vec::as_mut_slice).collect();
        let written = resampler.process_chunk(&chunk, &mut outputs).unwrap();
        collect(&mut buffers, written);
        offset += input_size;
    }
    let chunk: Vec<&[f64]> = planes.iter().map(|plane| &plane[offset..]).collect();
    let mut outputs: Vec<&mut [f64]> = buffers.iter_mut().map(Vec::as_mut_slice).collect();
    let written = resampler.process_chunk_final(&chunk, &mut outputs).unwrap();
    collect(&mut buffers, written);
    let mut outputs: Vec<&mut [f64]> = buffers.iter_mut().map(Vec::as_mut_slice).collect();
    let written = resampler.finalize(&mut outputs).unwrap();
    collect(&mut buffers, written);

    let output_frames = output_planes[0].len();
    (0..output_frames)
        .flat_map(|frame| output_planes.iter().map(move |plane| plane[frame]))
        .collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |max, (x, y)| max.max((x - y).abs()))
}

/// Asserts `aliased` has the same length as `strict` but measurably different content.
fn assert_alias_floor_applied(strict: &[f64], aliased: &[f64]) {
    assert_eq!(strict.len(), aliased.len(), "alias floor must not change output length");
    // Identical configs are bit-identical, so anything well above rounding noise means the
    // alias floor took effect.
    let diff = max_abs_diff(strict, aliased);
    assert!(diff > 1e-7, "alias floor had no effect (max diff {diff})");
}

#[test]
fn interleaved_and_planar_apply_alias_floor_in_both_directions() {
    // (input rate, output rate, near-Nyquist tone): folds when downsampling, images when upsampling.
    for (input_rate, output_rate, near_nyquist_hz) in [(96_000, 44_100, 22_300.0), (44_100, 96_000, 21_800.0)] {
        let strict = Config::new(input_rate, output_rate, 2);
        let aliased = strict.clone().with_alias_floor_db(-3.0);
        let input = signal(input_rate, 2, near_nyquist_hz);

        let strict_out = run_interleaved(strict, &input, None);
        let aliased_out = run_interleaved(aliased.clone(), &input, None);
        assert_alias_floor_applied(&strict_out, &aliased_out);

        let planar_out = run_planar(aliased, &input);
        assert_eq!(planar_out.len(), aliased_out.len());
        assert!(
            max_abs_diff(&planar_out, &aliased_out) < 1e-12,
            "planar and interleaved outputs differ"
        );
    }
}

#[test]
fn decimation_keeps_length_with_alias_floor() {
    let strict = Config::new(192_000, 44_100, 1).with_decimate(true);
    let aliased = strict.clone().with_alias_floor_db(-3.0);
    let input = signal(192_000, 1, 22_300.0);

    assert_alias_floor_applied(
        &run_interleaved(strict, &input, None),
        &run_interleaved(aliased, &input, None),
    );
}

#[test]
fn gapless_context_keeps_length_with_alias_floor() {
    let strict = Config::new(96_000, 44_100, 2);
    let aliased = strict.clone().with_alias_floor_db(-3.0);
    let input = signal(96_000, 2, 22_300.0);
    let context_len = InterleavedResampler::<f64>::new(strict.clone())
        .unwrap()
        .input_buffer_size();
    let context = || Some((input[..context_len].to_vec(), input[..context_len].to_vec()));

    assert_alias_floor_applied(
        &run_interleaved(strict, &input, context()),
        &run_interleaved(aliased, &input, context()),
    );
}

#[cfg(feature = "dd_fft")]
#[test]
fn dd_fft_applies_alias_floor() {
    let strict = Config::new(96_000, 44_100, 1).with_dd_fft(true);
    let aliased = strict.clone().with_alias_floor_db(-3.0);
    let input = signal(96_000, 1, 22_300.0);

    assert_alias_floor_applied(
        &run_interleaved(strict, &input, None),
        &run_interleaved(aliased, &input, None),
    );
}
