//! A value kept between runs for each device, such as a GPU context.
//!
//! Creating a context takes 65 to 76 ms on an NVIDIA laptop GPU, far more than a small run, so
//! the backends keep one per device instead of making one per run. A run that fails removes
//! the entry, so a sticky driver error does not reach the next run. The table lives in a
//! `static`, which Rust never drops, so a context is not destroyed while the driver unloads.

use std::sync::{Arc, Mutex, PoisonError};

/// Values by key, one each. The key is a device index.
pub struct Kept<T> {
    slots: Mutex<Vec<(usize, Arc<T>)>>,
}

impl<T> Kept<T> {
    pub const fn new() -> Self {
        Kept {
            slots: Mutex::new(Vec::new()),
        }
    }

    /// The value kept for `key`. `make` builds it when there is none (cudarc already hands out
    /// an `Arc`), and the table keeps what it built. A `make` that fails keeps nothing. The table stays locked while `make`
    /// runs, so two first runs on one device build one value between them.
    pub fn get_or_make<E>(
        &self,
        key: usize,
        make: impl FnOnce() -> Result<Arc<T>, E>,
    ) -> Result<Arc<T>, E> {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, v)) = slots.iter().find(|(k, _)| *k == key) {
            return Ok(Arc::clone(v));
        }
        let v = make()?;
        slots.push((key, Arc::clone(&v)));
        Ok(v)
    }

    /// Removes the value kept for `key` and returns it, so that the caller drops it where a
    /// panic is caught. A run that still holds the value keeps it alive until it ends.
    pub fn forget(&self, key: usize) -> Option<Arc<T>> {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        let at = slots.iter().position(|(k, _)| *k == key)?;
        Some(slots.swap_remove(at).1)
    }

    /// How many values are kept.
    pub fn len(&self) -> usize {
        self.slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for Kept<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn counted(built: &AtomicUsize, v: u32) -> impl FnOnce() -> Result<Arc<u32>, String> + '_ {
        move || {
            built.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(v))
        }
    }

    #[test]
    fn the_second_call_for_a_key_reuses_the_value() {
        let kept = Kept::new();
        let built = AtomicUsize::new(0);
        let a = kept.get_or_make(0, counted(&built, 7)).unwrap();
        let b = kept.get_or_make(0, counted(&built, 8)).unwrap();
        assert_eq!((*a, *b), (7, 7));
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(built.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn each_key_has_its_own_value() {
        let kept = Kept::new();
        let built = AtomicUsize::new(0);
        let a = kept.get_or_make(0, counted(&built, 1)).unwrap();
        let b = kept.get_or_make(1, counted(&built, 2)).unwrap();
        assert_eq!((*a, *b), (1, 2));
        assert_eq!(kept.len(), 2);
        assert_eq!(built.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_failed_make_keeps_nothing_and_the_next_call_tries_again() {
        let kept: Kept<u32> = Kept::new();
        let err = kept.get_or_make(0, || Err::<Arc<u32>, _>("no device".to_owned()));
        assert_eq!(err.unwrap_err(), "no device");
        assert!(kept.is_empty());
        let built = AtomicUsize::new(0);
        let v = kept.get_or_make(0, counted(&built, 3)).unwrap();
        assert_eq!(*v, 3);
        assert_eq!(built.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn forget_removes_the_value_and_the_next_call_builds_a_new_one() {
        let kept = Kept::new();
        let built = AtomicUsize::new(0);
        let first = kept.get_or_make(0, counted(&built, 1)).unwrap();
        let gone = kept.forget(0).expect("a value was kept");
        assert!(Arc::ptr_eq(&first, &gone));
        assert!(kept.is_empty());
        let second = kept.get_or_make(0, counted(&built, 2)).unwrap();
        assert_eq!(*second, 2);
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(built.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn forget_of_a_key_with_no_value_is_none_and_keeps_the_others() {
        let kept = Kept::new();
        let built = AtomicUsize::new(0);
        kept.get_or_make(1, counted(&built, 5)).unwrap();
        assert!(kept.forget(0).is_none());
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn a_run_that_holds_the_value_keeps_it_after_forget() {
        let kept = Kept::new();
        let built = AtomicUsize::new(0);
        let held = kept.get_or_make(0, counted(&built, 9)).unwrap();
        drop(kept.forget(0));
        assert_eq!(*held, 9);
        assert_eq!(Arc::strong_count(&held), 1);
    }

    #[test]
    fn a_panic_in_make_does_not_poison_the_table() {
        let kept: Kept<u32> = Kept::new();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = kept.get_or_make(0, || -> Result<Arc<u32>, String> { panic!("driver") });
        }));
        assert!(r.is_err());
        let built = AtomicUsize::new(0);
        let v = kept.get_or_make(0, counted(&built, 4)).unwrap();
        assert_eq!(*v, 4);
    }

    #[test]
    fn threads_that_start_together_build_one_value() {
        let kept = Arc::new(Kept::new());
        let built = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let kept = Arc::clone(&kept);
                let built = Arc::clone(&built);
                std::thread::spawn(move || {
                    let v = kept
                        .get_or_make(0, || {
                            built.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(std::time::Duration::from_millis(20));
                            Ok::<_, String>(Arc::new(11u32))
                        })
                        .unwrap();
                    *v
                })
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), 11);
        }
        assert_eq!(built.load(Ordering::SeqCst), 1);
    }
}
