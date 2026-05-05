use audioadapter::Adapter;
use audioadapter::AdapterMut;
use crate::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct SequentialVecOfVecs<T> {
    buf: Vec<Vec<T>>,
    frames: usize,
}

pub struct SequentialVecOfVecsIter<'a, T> {
    inner: std::slice::Iter<'a, Vec<T>>,
}

pub struct SequentialVecOfVecsIterMut<'a, T> {
    inner: std::slice::IterMut<'a, Vec<T>>,
}

impl<T> SequentialVecOfVecs<T> {
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
}

impl<'a, T> Iterator for SequentialVecOfVecsIter<'a, T> {
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

impl<T> ExactSizeIterator for SequentialVecOfVecsIter<'_, T> {}

impl<'a, T> Iterator for SequentialVecOfVecsIterMut<'a, T> {
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

impl<T> ExactSizeIterator for SequentialVecOfVecsIterMut<'_, T> {}

impl<T> From<SequentialVecOfVecs<T>> for Vec<Vec<T>> {
    #[inline]
    fn from(value: SequentialVecOfVecs<T>) -> Self {
        value.into_inner()
    }
}

impl<T> IntoIterator for SequentialVecOfVecs<T> {
    type Item = Vec<T>;
    type IntoIter = std::vec::IntoIter<Vec<T>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_inner().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a SequentialVecOfVecs<T> {
    type Item = &'a [T];
    type IntoIter = SequentialVecOfVecsIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        SequentialVecOfVecsIter { inner: self.buf.iter() }
    }
}

impl<'a, T> IntoIterator for &'a mut SequentialVecOfVecs<T> {
    type Item = &'a mut [T];
    type IntoIter = SequentialVecOfVecsIterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        SequentialVecOfVecsIterMut {
            inner: self.buf.iter_mut(),
        }
    }
}

unsafe impl<'a, T> Adapter<'a, T> for SequentialVecOfVecs<T>
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

unsafe impl<'a, T> AdapterMut<'a, T> for SequentialVecOfVecs<T>
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