#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
use bevy_derive::{Deref, DerefMut};

/// A wrapper to safely make `wgpu` types Send / Sync on web with atomics enabled.
///
/// On web with `atomics` enabled the inner value can only be accessed on the thread that
/// created it or else a panic will occur. It may be dropped on any thread: a drop elsewhere
/// parks the value until [`drop_parked_wgpu_values`] runs on the owning thread.
/// On other platforms the wrapper simply contains the wrapped value.
#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
#[derive(Debug, Clone, Deref, DerefMut)]
pub struct WgpuWrapper<T>(T);
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub struct WgpuWrapper<T: 'static> {
    value: core::mem::ManuallyDrop<T>,
    thread: std::thread::ThreadId,
}

// SAFETY: the value is only accessed on the thread that created it (checked on every deref)
// and only dropped there (drops from other threads are parked, see `Drop`).
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[expect(unsafe_code, reason = "Blanket-impl Send requires unsafe.")]
unsafe impl<T> Send for WgpuWrapper<T> {}
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[expect(unsafe_code, reason = "Blanket-impl Sync requires unsafe.")]
unsafe impl<T> Sync for WgpuWrapper<T> {}

#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
impl<T> WgpuWrapper<T> {
    /// Constructs a new instance of `WgpuWrapper` which will wrap the specified value.
    pub fn new(t: T) -> Self {
        Self(t)
    }

    /// Unwraps the value.
    pub fn into_inner(self) -> T {
        self.0
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
impl<T> WgpuWrapper<T> {
    /// Constructs a new instance of `WgpuWrapper` which will wrap the specified value.
    pub fn new(t: T) -> Self {
        Self {
            value: core::mem::ManuallyDrop::new(t),
            thread: std::thread::current().id(),
        }
    }

    /// Unwraps the value.
    pub fn into_inner(mut self) -> T {
        self.assert_thread();
        #[expect(unsafe_code, reason = "ManuallyDrop::take requires unsafe.")]
        // SAFETY: `self` is forgotten right after, so the value is never dropped twice.
        let value = unsafe { core::mem::ManuallyDrop::take(&mut self.value) };
        core::mem::forget(self);
        value
    }

    #[inline]
    fn assert_thread(&self) {
        if self.thread != std::thread::current().id() {
            panic!(
                "Accessed a WgpuWrapper<{}> from a thread different to the one it has been created with.",
                core::any::type_name::<T>()
            );
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
impl<T> core::ops::Deref for WgpuWrapper<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.assert_thread();
        &self.value
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
impl<T> core::ops::DerefMut for WgpuWrapper<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.assert_thread();
        &mut self.value
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
impl<T: Clone> Clone for WgpuWrapper<T> {
    fn clone(&self) -> Self {
        Self::new(T::clone(self))
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
impl<T: core::fmt::Debug> core::fmt::Debug for WgpuWrapper<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("WgpuWrapper").field(&**self).finish()
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
impl<T> Drop for WgpuWrapper<T> {
    fn drop(&mut self) {
        #[expect(unsafe_code, reason = "ManuallyDrop::take requires unsafe.")]
        // SAFETY: `value` is never touched again after being taken.
        let value = unsafe { core::mem::ManuallyDrop::take(&mut self.value) };
        if self.thread == std::thread::current().id() {
            drop(value);
        } else {
            bevy_log::debug!(
                "parking a WgpuWrapper<{}> dropped off its owning thread",
                core::any::type_name::<T>()
            );
            parked_wgpu_values::park(value, self.thread);
        }
    }
}

/// Drops the wgpu values that were dropped on a thread other than the one that created them
/// (see [`WgpuWrapper`]). Only values owned by the current thread are dropped; the rest stay
/// parked. Runs every frame on the render thread.
pub fn drop_parked_wgpu_values() {
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    parked_wgpu_values::drop_owned_by_current_thread();
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
mod parked_wgpu_values {
    use std::sync::Mutex;
    use std::thread::ThreadId;

    struct Parked {
        _value: Box<dyn core::any::Any>,
        thread: ThreadId,
    }

    // SAFETY: the value is only ever dropped on `thread`, the thread that created it.
    #[expect(unsafe_code, reason = "Send for a value only dropped on its owning thread.")]
    unsafe impl Send for Parked {}

    static PARKED: Mutex<Vec<Parked>> = Mutex::new(Vec::new());

    pub fn park<T: 'static>(value: T, thread: ThreadId) {
        PARKED.lock().unwrap().push(Parked {
            _value: Box::new(value),
            thread,
        });
    }

    pub fn drop_owned_by_current_thread() {
        let current = std::thread::current().id();
        // drop outside the lock: a value may itself hold wrappers that park on drop
        let parked = core::mem::take(&mut *PARKED.lock().unwrap());
        let (owned, other): (Vec<_>, Vec<_>) =
            parked.into_iter().partition(|p| p.thread == current);
        PARKED.lock().unwrap().extend(other);
        drop(owned);
    }
}
