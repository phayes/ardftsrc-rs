use audioadapter::Adapter;
use audioadapter::AdapterMut;
use crate::Error;

#[derive(Clone, Debug, PartialEq)]
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
    pub fn channels(&self) -> usize {
        self.buf.len()
    }

    #[must_use]
    #[inline]
    pub fn frames(&self) -> usize {
        self.frames
    }

    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty() || self.frames == 0
    }

    #[must_use]
    #[inline]
    pub fn get_channel(&self, index: usize) -> Option<&[T]> {
        self.buf.get(index).map(Vec::as_slice)
    }

    #[must_use]
    #[inline]
    pub fn get_channel_mut(&mut self, index: usize) -> Option<&mut [T]> {
        self.buf.get_mut(index).map(Vec::as_mut_slice)
    }

    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[Vec<T>] {
        &self.buf
    }

    #[must_use]
    #[inline]
    pub fn into_inner(self) -> Vec<Vec<T>> {
        self.buf
    }

    #[must_use]
    #[inline]
    pub fn pop_channel(&mut self) -> Option<Vec<T>> {
        let channel = self.buf.pop();
        if self.buf.is_empty() {
            self.frames = 0;
        }
        channel
    }

    /// Returns a single interleaved buffer in frame-major order.
    ///
    /// The returned layout is:
    /// `[L1, R1, L2, R2, ...]`
    ///
    /// For example, channels `[[1, 2], [3, 4]]` become `[1, 3, 2, 4]`.
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