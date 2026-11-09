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

    /// The value kept for `key`, if any.
    pub fn get(&self, key: usize) -> Option<Arc<T>> {
        let slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        slots
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| Arc::clone(v))
    }

    /// Removes the value kept for `key` and returns it, so that the caller drops it where a
    /// panic is caught. A run that still holds the value keeps it alive until it ends.
    pub fn forget(&self, key: usize) -> Option<Arc<T>> {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        let at = slots.iter().position(|(k, _)| *k == key)?;
        Some(slots.swap_remove(at).1)
    }

    /// Removes the value kept for `key` only if it is the one `seen` points to, and returns it.
    /// A run that failed removes the value it used. Another run may have replaced that value
    /// after an earlier failure, and the new one must stay.
    pub fn forget_if(&self, key: usize, seen: &Arc<T>) -> Option<Arc<T>> {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        let at = slots
            .iter()
            .position(|(k, v)| *k == key && Arc::ptr_eq(v, seen))?;
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

/// A short list of values by key that drops the least recently used one when it is full. The
/// backends keep their compiled kernels in one of these inside the kept device entry.
///
/// `V` is cheap to clone (an `Arc`, or a handle that holds one), so a run takes a clone and
/// keeps using it after the list dropped its own. The list stays locked while `make` runs, so
/// threads that need one value compile it once. A value that the list drops is dropped here,
/// so a caller must call `get_or_make` where a panic is caught.
pub struct Lru<K, V> {
    bound: usize,
    /// Oldest first, most recently used last.
    slots: Mutex<Vec<(K, V)>>,
}

impl<K: PartialEq, V: Clone> Lru<K, V> {
    /// An empty list that holds at most `bound` values, and at least one.
    pub fn new(bound: usize) -> Self {
        Lru {
            bound: bound.max(1),
            slots: Mutex::new(Vec::new()),
        }
    }

    /// The value for `key`, which becomes the most recently used. `make` builds it when there is
    /// none. A `make` that fails keeps nothing and drops nothing.
    pub fn get_or_make<E>(&self, key: K, make: impl FnOnce() -> Result<V, E>) -> Result<V, E> {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(at) = slots.iter().position(|(k, _)| *k == key) {
            let hit = slots.remove(at);
            let value = hit.1.clone();
            slots.push(hit);
            return Ok(value);
        }
        let value = make()?;
        if slots.len() >= self.bound {
            slots.remove(0);
        }
        slots.push((key, value.clone()));
        Ok(value)
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
    fn get_peeks_without_making() {
        let kept = Kept::new();
        assert!(kept.get(0).is_none());
        let built = AtomicUsize::new(0);
        let a = kept.get_or_make(0, counted(&built, 5)).unwrap();
        let b = kept.get(0).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert!(kept.get(1).is_none());
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
    fn forget_if_removes_only_the_value_that_was_seen() {
        let kept = Kept::new();
        let built = AtomicUsize::new(0);
        let old = kept.get_or_make(0, counted(&built, 1)).unwrap();
        drop(kept.forget(0));
        let new = kept.get_or_make(0, counted(&built, 2)).unwrap();
        // A second failing run still holds the old value and asks to remove it.
        assert!(kept.forget_if(0, &old).is_none());
        assert_eq!(kept.len(), 1);
        assert!(Arc::ptr_eq(&kept.get(0).unwrap(), &new));
        let gone = kept.forget_if(0, &new).expect("the seen value is kept");
        assert!(Arc::ptr_eq(&gone, &new));
        assert!(kept.is_empty());
        assert!(kept.forget_if(1, &new).is_none());
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

#[cfg(test)]
mod lru_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn make(built: &AtomicUsize, v: u32) -> impl FnOnce() -> Result<Arc<u32>, String> + '_ {
        move || {
            built.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(v))
        }
    }

    #[test]
    fn a_second_call_with_the_same_key_reuses_the_value() {
        let lru: Lru<String, Arc<u32>> = Lru::new(4);
        let built = AtomicUsize::new(0);
        let a = lru.get_or_make("k".to_owned(), make(&built, 1)).unwrap();
        let b = lru.get_or_make("k".to_owned(), make(&built, 2)).unwrap();
        assert_eq!((*a, *b), (1, 1));
        assert_eq!(built.load(Ordering::SeqCst), 1);
        assert_eq!(lru.len(), 1);
    }

    #[test]
    fn keys_that_differ_in_any_part_get_their_own_value() {
        let lru: Lru<(String, u8), Arc<u32>> = Lru::new(4);
        let built = AtomicUsize::new(0);
        for (i, key) in [("a", 0), ("a", 1), ("b", 0)].into_iter().enumerate() {
            let v = lru
                .get_or_make((key.0.to_owned(), key.1), make(&built, i as u32))
                .unwrap();
            assert_eq!(*v, i as u32);
        }
        assert_eq!(lru.len(), 3);
        assert_eq!(built.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn the_list_never_holds_more_than_its_bound() {
        let lru: Lru<u32, Arc<u32>> = Lru::new(3);
        let built = AtomicUsize::new(0);
        for k in 0..10 {
            lru.get_or_make(k, make(&built, k)).unwrap();
            assert!(lru.len() <= 3);
        }
        assert_eq!(lru.len(), 3);
    }

    #[test]
    fn the_least_recently_used_value_goes_first() {
        let lru: Lru<u32, Arc<u32>> = Lru::new(3);
        let built = AtomicUsize::new(0);
        for k in 0..3 {
            lru.get_or_make(k, make(&built, k)).unwrap();
        }
        // Use 0 again, so 1 is now the oldest.
        lru.get_or_make(0, make(&built, 0)).unwrap();
        assert_eq!(built.load(Ordering::SeqCst), 3);
        lru.get_or_make(3, make(&built, 3)).unwrap();
        assert_eq!(built.load(Ordering::SeqCst), 4);
        // 0, 2 and 3 are kept and 1 was dropped.
        for k in [0, 2, 3] {
            lru.get_or_make(k, make(&built, k)).unwrap();
        }
        assert_eq!(built.load(Ordering::SeqCst), 4);
        lru.get_or_make(1, make(&built, 1)).unwrap();
        assert_eq!(built.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn a_failed_make_keeps_nothing_and_does_not_push_out_a_value() {
        let lru: Lru<u32, Arc<u32>> = Lru::new(2);
        let built = AtomicUsize::new(0);
        lru.get_or_make(0, make(&built, 0)).unwrap();
        lru.get_or_make(1, make(&built, 1)).unwrap();
        let err = lru.get_or_make(2, || Err::<Arc<u32>, _>("compile".to_owned()));
        assert_eq!(err.unwrap_err(), "compile");
        assert_eq!(lru.len(), 2);
        for k in [0, 1] {
            lru.get_or_make(k, make(&built, k)).unwrap();
        }
        assert_eq!(built.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_holder_keeps_a_value_that_the_list_dropped() {
        let lru: Lru<u32, Arc<u32>> = Lru::new(1);
        let built = AtomicUsize::new(0);
        let held = lru.get_or_make(0, make(&built, 7)).unwrap();
        lru.get_or_make(1, make(&built, 8)).unwrap();
        assert_eq!(*held, 7);
        assert_eq!(Arc::strong_count(&held), 1);
    }

    #[test]
    fn a_panic_in_make_does_not_poison_the_list() {
        let lru: Lru<u32, Arc<u32>> = Lru::new(2);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = lru.get_or_make(0, || -> Result<Arc<u32>, String> { panic!("driver") });
        }));
        assert!(r.is_err());
        let built = AtomicUsize::new(0);
        assert_eq!(*lru.get_or_make(0, make(&built, 4)).unwrap(), 4);
        assert_eq!(lru.len(), 1);
    }

    #[test]
    fn threads_that_need_one_kernel_compile_it_once() {
        let lru = Arc::new(Lru::<u32, Arc<u32>>::new(2));
        let built = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let lru = Arc::clone(&lru);
                let built = Arc::clone(&built);
                std::thread::spawn(move || {
                    let v = lru
                        .get_or_make(5, || {
                            built.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(std::time::Duration::from_millis(20));
                            Ok::<_, String>(Arc::new(5u32))
                        })
                        .unwrap();
                    *v
                })
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), 5);
        }
        assert_eq!(built.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_bound_of_zero_is_treated_as_one() {
        let lru: Lru<u32, Arc<u32>> = Lru::new(0);
        let built = AtomicUsize::new(0);
        lru.get_or_make(0, make(&built, 0)).unwrap();
        lru.get_or_make(0, make(&built, 0)).unwrap();
        assert_eq!(built.load(Ordering::SeqCst), 1);
        assert_eq!(lru.len(), 1);
    }
}
