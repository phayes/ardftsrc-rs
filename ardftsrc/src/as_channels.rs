use crate::PlanarVecs;

/// A read-only, per-channel view over planar audio data.
///
/// Implemented for the common shapes a caller already holds -- a slice of channel slices, a
/// `Vec<Vec<T>>`/`[Vec<T>]`, or a [`PlanarVecs`] -- so APIs that need per-channel input can
/// accept any of them directly instead of requiring the caller to first collect a temporary
/// `Vec<&[T]>`.
pub trait AsChannels<'a, T> {
    /// Number of channels.
    fn channel_count(&self) -> usize;

    /// Borrows one channel's samples.
    fn channel(&self, index: usize) -> &'a [T];
}

impl<'a, T> AsChannels<'a, T> for &'a [&'a [T]] {
    fn channel_count(&self) -> usize {
        self.len()
    }

    fn channel(&self, index: usize) -> &'a [T] {
        self[index]
    }
}

impl<'a, T> AsChannels<'a, T> for &'a Vec<Vec<T>> {
    fn channel_count(&self) -> usize {
        self.len()
    }

    fn channel(&self, index: usize) -> &'a [T] {
        self[index].as_slice()
    }
}

impl<'a, T> AsChannels<'a, T> for &'a [Vec<T>] {
    fn channel_count(&self) -> usize {
        self.len()
    }

    fn channel(&self, index: usize) -> &'a [T] {
        self[index].as_slice()
    }
}

impl<'a, T> AsChannels<'a, T> for &'a PlanarVecs<T> {
    fn channel_count(&self) -> usize {
        self.channels()
    }

    fn channel(&self, index: usize) -> &'a [T] {
        self.get_channel(index).expect("channel index in range")
    }
}

/// A mutable, per-channel view over planar audio data.
///
/// The write-side counterpart to [`AsChannels`], for APIs that fill in per-channel output.
pub trait AsChannelsMut<'a, T> {
    /// Number of channels.
    fn channel_count(&self) -> usize;

    /// Mutably borrows one channel's samples.
    fn channel_mut(&mut self, index: usize) -> &mut [T];
}

impl<'a, T> AsChannelsMut<'a, T> for &'a mut [&'a mut [T]] {
    fn channel_count(&self) -> usize {
        self.len()
    }

    fn channel_mut(&mut self, index: usize) -> &mut [T] {
        &mut self[index][..]
    }
}

impl<'a, T> AsChannelsMut<'a, T> for &'a mut Vec<Vec<T>> {
    fn channel_count(&self) -> usize {
        self.len()
    }

    fn channel_mut(&mut self, index: usize) -> &mut [T] {
        self[index].as_mut_slice()
    }
}

impl<'a, T> AsChannelsMut<'a, T> for &'a mut [Vec<T>] {
    fn channel_count(&self) -> usize {
        self.len()
    }

    fn channel_mut(&mut self, index: usize) -> &mut [T] {
        self[index].as_mut_slice()
    }
}

impl<'a, T> AsChannelsMut<'a, T> for &'a mut PlanarVecs<T> {
    fn channel_count(&self) -> usize {
        self.channels()
    }

    fn channel_mut(&mut self, index: usize) -> &mut [T] {
        self.get_channel_mut(index).expect("channel index in range")
    }
}

#[cfg(test)]
mod tests {
    use super::{AsChannels, AsChannelsMut};
    use crate::PlanarVecs;

    fn collect_channels<'a, T: Copy + 'a>(input: impl AsChannels<'a, T>) -> Vec<Vec<T>> {
        (0..input.channel_count()).map(|c| input.channel(c).to_vec()).collect()
    }

    #[test]
    fn slice_of_slices_yields_channels() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let refs: &[&[i32]] = &[&a, &b];
        assert_eq!(collect_channels(refs), vec![vec![1, 2, 3], vec![4, 5, 6]]);
    }

    #[test]
    fn vec_of_vec_yields_channels() {
        let v: Vec<Vec<i32>> = vec![vec![1, 2], vec![3, 4]];
        assert_eq!(collect_channels(&v), vec![vec![1, 2], vec![3, 4]]);
    }

    #[test]
    fn planar_vecs_yields_channels() {
        let planar = PlanarVecs::new(vec![vec![1, 2], vec![3, 4]]).unwrap();
        assert_eq!(collect_channels(&planar), vec![vec![1, 2], vec![3, 4]]);
    }

    #[test]
    fn mut_variants_write_into_channels() {
        let mut v: Vec<Vec<i32>> = vec![vec![0, 0], vec![0, 0]];
        {
            let mut view: &mut Vec<Vec<i32>> = &mut v;
            for c in 0..AsChannelsMut::<i32>::channel_count(&view) {
                view.channel_mut(c).copy_from_slice(&[9, 9]);
            }
        }
        assert_eq!(v, vec![vec![9, 9], vec![9, 9]]);

        let mut planar = PlanarVecs::new(vec![vec![0, 0]]).unwrap();
        {
            let mut view: &mut PlanarVecs<i32> = &mut planar;
            view.channel_mut(0).copy_from_slice(&[7, 7]);
        }
        assert_eq!(planar.get_channel(0), Some(&[7, 7][..]));
    }
}
