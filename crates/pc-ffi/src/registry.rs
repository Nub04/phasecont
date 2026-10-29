//! Handle registry. Objects cross the C ABI as opaque u64 ids. An unknown or
//! freed id, or an id of the wrong type, is an error, never a dereference.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::error::{Error, Result};

struct Entry {
    kind: &'static str,
    object: Arc<dyn Any + Send + Sync>,
}

static OBJECTS: Mutex<BTreeMap<u64, Entry>> = Mutex::new(BTreeMap::new());
static NEXT: AtomicU64 = AtomicU64::new(1);

/// Tag in the high 16 bits of every id, so 0 and small integers are never
/// valid. Ids are `TAG | (counter << 1) | 1`, so every id is odd. Every double
/// in [2^62, 2^63), the range of the tagged ids, is a multiple of 1024 and so
/// even, which means an id that went through a double never matches a live
/// object. This holds while the counter stays below 2^47.
const TAG: u64 = 0x5043_0000_0000_0000;

fn objects() -> MutexGuard<'static, BTreeMap<u64, Entry>> {
    // Only insert, clone and remove run under the lock, so the map is still
    // consistent after a panic poisoned it.
    OBJECTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Stores `value` under a new id. `kind` is a short name ("model", "run")
/// that error messages use instead of a Rust type path.
pub fn insert<T: Any + Send + Sync>(kind: &'static str, value: T) -> u64 {
    let id = TAG | (NEXT.fetch_add(1, Ordering::Relaxed) << 1) | 1;
    objects().insert(
        id,
        Entry {
            kind,
            object: Arc::new(value),
        },
    );
    id
}

/// Returns a shared reference, so a long computation does not hold the lock.
pub fn get<T: Any + Send + Sync>(id: u64, kind: &'static str) -> Result<Arc<T>> {
    let (actual, object) = {
        let map = objects();
        let e = map
            .get(&id)
            .ok_or_else(|| Error::BadHandle(format!("unknown handle 0x{id:016x}")))?;
        (e.kind, Arc::clone(&e.object))
    };
    object
        .downcast::<T>()
        .map_err(|_| Error::BadHandle(format!("handle 0x{id:016x} is a {actual}, not a {kind}")))
}

/// Removes the entry. The object is dropped after the lock is released.
pub fn remove(id: u64) -> Result<()> {
    let entry = objects().remove(&id);
    match entry {
        Some(entry) => {
            drop(entry);
            Ok(())
        }
        None => Err(Error::BadHandle(format!("unknown handle 0x{id:016x}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_tagged_typed_and_single_use() {
        let id = insert("number", 42u32);
        assert_eq!(id & TAG, TAG);
        assert_eq!(id & 1, 1);
        assert_eq!(*get::<u32>(id, "number").unwrap(), 42);
        let e = get::<f64>(id, "float").unwrap_err();
        assert_eq!(
            e.to_string(),
            format!("handle 0x{id:016x} is a number, not a float")
        );
        remove(id).unwrap();
        assert!(matches!(get::<u32>(id, "number"), Err(Error::BadHandle(_))));
        assert!(matches!(remove(id), Err(Error::BadHandle(_))));
        assert!(matches!(remove(0), Err(Error::BadHandle(_))));
    }

    #[test]
    fn survives_a_poisoned_lock() {
        let r = std::thread::spawn(|| {
            let _guard = OBJECTS.lock().unwrap();
            panic!("poison the registry");
        })
        .join();
        assert!(r.is_err());
        assert!(OBJECTS.is_poisoned());
        let id = insert("number", 7u8);
        assert_eq!(*get::<u8>(id, "number").unwrap(), 7);
        remove(id).unwrap();
    }
}
