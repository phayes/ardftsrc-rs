use num_complex::Complex;
use num_traits::Zero;

use crate::f128_fft::vendor::rustfft::{Direction, Fft, FftNum, Length};
use crate::f128_fft::vendor::rustfft::{FftDirection, twiddles};

/// Naive O(n^2 ) Discrete Fourier Transform implementation
///
/// This implementation is primarily used to test other FFT algorithms.
///
/// ~~~
/// // Computes a naive DFT of size 123
/// use rustfft::algorithm::Dft;
/// use rustfft::{Fft, FftDirection};
/// use rustfft::num_complex::Complex;
///
/// let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 123];
///
/// let dft = Dft::new(123, FftDirection::Forward);
/// dft.process(&mut buffer);
/// ~~~
pub struct Dft<T> {
    twiddles: Vec<Complex<T>>,
    direction: FftDirection,
}

impl<T: FftNum> Dft<T> {
    /// Preallocates necessary arrays and precomputes necessary data to efficiently compute Dft
    pub fn new(len: usize, direction: FftDirection) -> Self {
        let twiddles = (0..len).map(|i| twiddles::compute_twiddle(i, len, direction)).collect();
        Self { twiddles, direction }
    }

    fn inplace_scratch_len(&self) -> usize {
        self.len()
    }
    fn outofplace_scratch_len(&self) -> usize {
        0
    }
    fn immut_scratch_len(&self) -> usize {
        0
    }

    fn perform_fft_immut(&self, signal: &[Complex<T>], spectrum: &mut [Complex<T>], _scratch: &mut [Complex<T>]) {
        for k in 0..spectrum.len() {
            let output_cell = spectrum.get_mut(k).unwrap();

            *output_cell = Zero::zero();
            let mut twiddle_index = 0;

            for input_cell in signal {
                let twiddle = self.twiddles[twiddle_index];
                *output_cell = *output_cell + twiddle * input_cell;

                twiddle_index += k;
                if twiddle_index >= self.twiddles.len() {
                    twiddle_index -= self.twiddles.len();
                }
            }
        }
    }

    fn perform_fft_out_of_place(
        &self,
        signal: &[Complex<T>],
        spectrum: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        self.perform_fft_immut(signal, spectrum, _scratch);
    }
}
boilerplate_fft_oop!(Dft, |this: &Dft<_>| this.twiddles.len());
