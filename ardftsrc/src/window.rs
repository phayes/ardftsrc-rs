//! Shared FFT-window construction logic.
//!
//! [`CpuCore`](crate::cpu_core::CpuCore) and the GPU streaming core build the same overlap-add
//! FFT windows (steady-state, start-edge priming, short-final-chunk, and finalize-tail), the
//! only difference being what happens to the window afterward (an in-process CPU FFT vs. an
//! upload to a GPU buffer). This module holds that window-construction logic once so the two
//! cores cannot drift apart on it -- pulled out of what was originally `ArdftsrcCore`-only
//! code, unchanged in behavior (every `cpu_core`/`spectral` test still exercises it exactly as
//! before) *and* in allocation profile: every function writes into a caller-provided `window`
//! buffer rather than returning an owned `Vec`, so [`write_normal_window`] -- called on every
//! chunk, including on a realtime audio thread via `RealtimeResampler` -- allocates nothing,
//! exactly like the original hand-inlined version did. The other functions here only run once
//! per stream (start priming, a short final chunk, or finalize), where the small `Vec`s they
//! allocate match what the original code already allocated in those same rare paths.

use num_traits::Float;

use crate::extrapolation::Extrapolation;

/// Copies up to `dst.len()` trailing samples from `pre` into `dst`'s tail. Returns the number
/// of samples copied.
pub(crate) fn copy_pre_tail<T: Float>(pre: Option<&[T]>, dst: &mut [T]) -> usize {
    let Some(pre) = pre else {
        return 0;
    };
    let copied = pre.len().min(dst.len());
    let start = pre.len() - copied;
    let dst_start = dst.len() - copied;
    dst[dst_start..].copy_from_slice(&pre[start..start + copied]);
    copied
}

/// Copies up to `dst.len()` leading samples from `post` into `dst`'s head. Returns the number
/// of samples copied.
pub(crate) fn copy_post_head<T: Float>(post: Option<&[T]>, dst: &mut [T]) -> usize {
    let Some(post) = post else {
        return 0;
    };
    let copied = post.len().min(dst.len());
    dst[..copied].copy_from_slice(&post[..copied]);
    copied
}

/// Fills a synthetic forward tail from `post` first, then `extrapolation` fallback.
pub(crate) fn build_tail_prediction<T: Float>(
    post: Option<&[T]>,
    base: &[T],
    needed: usize,
    extrapolation: Extrapolation,
) -> Vec<T> {
    let mut predicted = vec![T::zero(); needed];
    let copied = copy_post_head(post, &mut predicted);
    if copied < needed {
        let mut seed = Vec::with_capacity(base.len() + copied);
        seed.extend_from_slice(base);
        seed.extend_from_slice(&predicted[..copied]);
        let fallback = extrapolation.forward(&seed, needed - copied);
        predicted[copied..].copy_from_slice(&fallback);
    }
    predicted
}

/// Writes a steady-state FFT window into `window` (`input` placed at `input_offset`, zero
/// elsewhere). Allocates nothing; this runs on every chunk, including on a realtime audio
/// thread.
pub(crate) fn write_normal_window<T: Float>(window: &mut [T], input_offset: usize, input: &[T]) {
    window.fill(T::zero());
    window[input_offset..input_offset + input.len()].copy_from_slice(input);
}

/// Writes the start-edge priming window used once, before the first real chunk, to seed
/// overlap state from real `pre` context (if any) or backward `extrapolation` from `input`
/// (the first real chunk) otherwise.
///
/// The synthetic predicted samples are placed at `input_chunk_frames..input_chunk_frames +
/// input_offset` in `window` (length `fft_size`) -- the position that makes this window's
/// second half land exactly where the real first chunk's overlap contribution belongs.
pub(crate) fn write_start_window<T: Float>(
    window: &mut [T],
    input_chunk_frames: usize,
    input_offset: usize,
    pre: Option<&[T]>,
    input: &[T],
    extrapolation: Extrapolation,
) {
    let mut predicted = vec![T::zero(); input_offset];
    let copied = copy_pre_tail(pre, &mut predicted);
    if copied < input_offset {
        let fallback_len = input_offset - copied;
        let fallback = extrapolation.reverse(input, fallback_len);
        predicted[..fallback_len].copy_from_slice(&fallback);
    }

    window.fill(T::zero());
    let tail_start = input_chunk_frames;
    window[tail_start..tail_start + predicted.len()].copy_from_slice(&predicted);
}

/// Writes the window for a short final chunk (fewer than `input_chunk_frames` real samples),
/// padding with prior real history (`prev_input_window`) and a synthetic forward tail (real
/// `post` context, or forward `extrapolation`). Updates `prev_input_window` in place so a
/// subsequent [`write_finalize_tail_window`] call (if the stream needs one) has correct
/// history; a short final chunk otherwise never calls [`save_current_window`].
///
/// `prev_input_window` must be `input_chunk_frames * 2` samples, matching
/// [`save_current_window`]'s own buffer convention; `window` must be `fft_size` samples.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_short_final_window<T: Float>(
    window: &mut [T],
    prev_input_window: &mut [T],
    input: &[T],
    input_chunk_frames: usize,
    input_offset: usize,
    post: Option<&[T]>,
    extrapolation: Extrapolation,
) {
    let input_samples = input.len();
    let pad_samples = input_chunk_frames - input_samples;

    let mut work = vec![T::zero(); input_chunk_frames * 2];
    work[..pad_samples].copy_from_slice(&prev_input_window[input_samples..input_samples + pad_samples]);
    work[pad_samples..pad_samples + input_samples].copy_from_slice(input);

    let predicted = build_tail_prediction(post, &work[..input_chunk_frames], input_chunk_frames, extrapolation);
    work[input_chunk_frames..input_chunk_frames * 2].copy_from_slice(&predicted);

    if input_samples > 0 {
        prev_input_window[..input_samples].copy_from_slice(&predicted[pad_samples..pad_samples + input_samples]);
        prev_input_window[input_samples..input_chunk_frames].fill(T::zero());
        prev_input_window[input_chunk_frames..input_chunk_frames * 2].fill(T::zero());
    }

    window.fill(T::zero());
    window[input_offset..input_offset + input_chunk_frames]
        .copy_from_slice(&work[pad_samples..pad_samples + input_chunk_frames]);
}

/// Persists the current window so a later [`write_finalize_tail_window`] call has sample-local
/// history. Must be called after every full (non-short-final) chunk.
///
/// `prev_input_window` must be `input_chunk_frames * 2` samples.
pub(crate) fn save_current_window<T: Float>(
    prev_input_window: &mut [T],
    window: &[T],
    input_offset: usize,
    input_chunk_frames: usize,
) {
    let history_start = input_offset;
    let history_end = history_start + input_chunk_frames;
    prev_input_window[..input_chunk_frames].copy_from_slice(&window[history_start..history_end]);
    prev_input_window[input_chunk_frames..].fill(T::zero());
}

/// Writes the finalize-tail window: a synthetic forward tail (real `post` context, or forward
/// `extrapolation`) appended after the last saved real window, run once at the very end
/// of a stream (in [`crate::cpu_core`]'s terms, `TransformMode::End`) to recover the final
/// overlap contribution that a steady-state chunk would otherwise still be holding back.
///
/// `prev_input_window` must be `input_chunk_frames * 2` samples, previously populated by
/// [`save_current_window`] (or left as the identity/zero state [`write_short_final_window`]
/// leaves it in, in which case this is a no-op tail of zeros); `window` must be `fft_size`
/// samples.
pub(crate) fn write_finalize_tail_window<T: Float>(
    window: &mut [T],
    prev_input_window: &mut [T],
    input_chunk_frames: usize,
    input_offset: usize,
    post: Option<&[T]>,
    extrapolation: Extrapolation,
) {
    let base = prev_input_window[..input_chunk_frames].to_vec();
    let predicted = build_tail_prediction(post, &base, input_offset, extrapolation);
    prev_input_window[input_chunk_frames..input_chunk_frames * 2].fill(T::zero());
    prev_input_window[input_chunk_frames..input_chunk_frames + predicted.len()].copy_from_slice(&predicted);

    window.fill(T::zero());
    window[input_offset..input_offset + input_chunk_frames]
        .copy_from_slice(&prev_input_window[input_chunk_frames..input_chunk_frames * 2]);
}
