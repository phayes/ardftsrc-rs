use crate::Error;
#[cfg(feature = "audioadapter")]
use audioadapter::Adapter;
#[cfg(feature = "audioadapter")]
use audioadapter::AdapterMut;
use num_traits::cast;

#[derive(Clone, Debug, PartialEq)]
/// A channel-major container of audio samples with equal frame counts.
///
/// Each inner vector is one channel, and all channels must contain the same
/// number of frames.
///
/// # Examples
///
/// ```
/// use ardftsrc::PlanarVecs;
///
/// let planar = PlanarVecs::<f32>::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
/// let interleaved = planar.interleave();
///
/// assert_eq!(interleaved, vec![1.0, 3.0, 2.0, 4.0]);
/// ```
pub struct PlanarVecs<T> {
    buf: Vec<Vec<T>>,
    frames: usize,
}

pub struct PlanarVecsIter<'a, T> {
    inner: std::slice::Iter<'a, Vec<T>>,
}

pub struct PlanarVecsIterMut<'a, T> {
    inner: std::slice::IterMut<'a, Vec<T>>,
}

impl<T> PlanarVecs<T> {
    /// Builds a planar buffer from channel vectors.
    ///
    /// Returns an error when channels do not all have the same frame count.
    pub fn new(buf: Vec<Vec<T>>) -> Result<Self, Error> {
        let frames = buf.first().map_or(0, Vec::len);
        if let Some(channel) = buf.iter().find(|channel| channel.len() != frames) {
            return Err(Error::WrongFrameCount {
                expected: frames,
                actual: channel.len(),
            });
        }

        Ok(Self { buf, frames })
    }

    #[must_use]
    #[inline]
    /// Returns the number of channels.
    pub fn channels(&self) -> usize {
        self.buf.len()
    }

    #[must_use]
    #[inline]
    /// Returns the number of frames per channel.
    pub fn frames(&self) -> usize {
        self.frames
    }

    #[must_use]
    #[inline]
    /// Returns `true` when there are no channels or no frames.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty() || self.frames == 0
    }

    #[must_use]
    #[inline]
    /// Returns an immutable view of a channel by index.
    pub fn get_channel(&self, index: usize) -> Option<&[T]> {
        self.buf.get(index).map(Vec::as_slice)
    }

    #[must_use]
    #[inline]
    /// Returns a mutable view of a channel by index.
    pub fn get_channel_mut(&mut self, index: usize) -> Option<&mut [T]> {
        self.buf.get_mut(index).map(Vec::as_mut_slice)
    }

    #[must_use]
    #[inline]
    /// Returns the underlying channel vectors as a slice.
    pub fn as_slice(&self) -> &[Vec<T>] {
        &self.buf
    }

    #[must_use]
    #[inline]
    /// Consumes `self` and returns the underlying channel vectors.
    pub fn into_inner(self) -> Vec<Vec<T>> {
        self.buf
    }

    #[must_use]
    #[inline]
    /// Removes and returns the last channel.
    ///
    /// When the last channel is removed, the frame count is reset to `0`.
    pub fn pop_channel(&mut self) -> Option<Vec<T>> {
        let channel = self.buf.pop();
        if self.buf.is_empty() {
            self.frames = 0;
        }
        channel
    }

    /// Returns an interleaved buffer in frame-major order.
    ///
    /// The returned layout is:
    /// `[L1, R1, L2, R2, ...]`
    ///
    /// For example, channels `[[1, 2], [3, 4]]` become `[1, 3, 2, 4]`.
    ///
    /// See also `interleave_into()` for converting to a different sample type (eg `f64` -> `f32`) while interleaving.
    pub fn interleave(&self) -> Vec<T>
    where
        T: Clone,
    {
        let mut interleaved = Vec::with_capacity(self.channels() * self.frames());

        for frame in 0..self.frames {
            for channel in &self.buf {
                interleaved.push(channel[frame].clone());
            }
        }

        interleaved
    }
}

impl<T> PlanarVecs<T>
where
    T: Clone + num_traits::NumCast,
{
    /// Returns an interleaved buffer converted to `S` in frame-major order.
    ///
    /// This is useful when output needs a different sample type, such as
    /// converting internal `f64` samples into `f32` playback samples.
    ///
    /// Panics when a sample cannot be represented in `S`.
    pub fn interleave_into<S>(&self) -> Vec<S>
    where
        S: num_traits::NumCast,
    {
        let mut interleaved = Vec::with_capacity(self.channels() * self.frames());

        for frame in 0..self.frames {
            for channel in &self.buf {
                let sample = channel[frame].clone();
                interleaved.push(cast(sample).expect("interleave_into target type cannot represent source sample"));
            }
        }

        interleaved
    }
}

impl<'a, T> Iterator for PlanarVecsIter<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(Vec::as_slice)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for PlanarVecsIter<'_, T> {}

impl<'a, T> Iterator for PlanarVecsIterMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(Vec::as_mut_slice)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for PlanarVecsIterMut<'_, T> {}

impl<T> From<PlanarVecs<T>> for Vec<Vec<T>> {
    #[inline]
    fn from(value: PlanarVecs<T>) -> Self {
        value.into_inner()
    }
}

impl<T> IntoIterator for PlanarVecs<T> {
    type Item = Vec<T>;
    type IntoIter = std::vec::IntoIter<Vec<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_inner().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a PlanarVecs<T> {
    type Item = &'a [T];
    type IntoIter = PlanarVecsIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        PlanarVecsIter { inner: self.buf.iter() }
    }
}

impl<'a, T> IntoIterator for &'a mut PlanarVecs<T> {
    type Item = &'a mut [T];
    type IntoIter = PlanarVecsIterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        PlanarVecsIterMut {
            inner: self.buf.iter_mut(),
        }
    }
}

#[cfg(feature = "audioadapter")]
unsafe impl<'a, T> Adapter<'a, T> for PlanarVecs<T>
where
    T: Clone + 'a,
{
    #[inline]
    unsafe fn read_sample_unchecked(&self, channel: usize, frame: usize) -> T {
        unsafe { self.buf.get_unchecked(channel).get_unchecked(frame).clone() }
    }

    #[inline]
    fn channels(&self) -> usize {
        self.channels()
    }

    #[inline]
    fn frames(&self) -> usize {
        self.frames()
    }

    fn copy_from_channel_to_slice(&self, channel: usize, skip: usize, slice: &mut [T]) -> usize {
        if channel >= self.channels() || skip >= self.frames {
            return 0;
        }

        let frames_to_write = (self.frames - skip).min(slice.len());
        slice[..frames_to_write].clone_from_slice(&self.buf[channel][skip..skip + frames_to_write]);
        frames_to_write
    }
}

#[cfg(feature = "audioadapter")]
unsafe impl<'a, T> AdapterMut<'a, T> for PlanarVecs<T>
where
    T: Clone + 'a,
{
    unsafe fn write_sample_unchecked(&mut self, channel: usize, frame: usize, value: &T) -> bool {
        unsafe {
            *self.buf.get_unchecked_mut(channel).get_unchecked_mut(frame) = value.clone();
        }
        false
    }

    fn copy_from_slice_to_channel(&mut self, channel: usize, skip: usize, slice: &[T]) -> (usize, usize) {
        if channel >= self.channels() || skip >= self.frames {
            return (0, 0);
        }

        let frames_to_read = (self.frames - skip).min(slice.len());
        self.buf[channel][skip..skip + frames_to_read].clone_from_slice(&slice[..frames_to_read]);
        (frames_to_read, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::PlanarVecs;

    #[test]
    fn mono_interleave_matches_pop_channel() {
        let mut planar = PlanarVecs::new(vec![vec![1_i32, 2, 3, 5, 8]]).unwrap();
        let interleaved = planar.interleave();
        let popped = planar.pop_channel().expect("mono has one channel");
        assert_eq!(interleaved, popped);
    }

    #[test]
    fn interleave_into_converts_sample_type() {
        let planar = PlanarVecs::new(vec![vec![1.5_f64, 2.25], vec![3.75, 4.5]]).unwrap();
        let interleaved: Vec<f32> = planar.interleave_into();
        assert_eq!(interleaved, vec![1.5_f32, 3.75, 2.25, 4.5]);
    }
}
