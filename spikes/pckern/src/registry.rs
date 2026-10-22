//! Handle registry. Objects cross the C ABI as opaque u64 ids. An unknown or
//! freed id, or an id of the wrong type, is an error, never a dereference.

use std::any::{Any, type_name};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::ffi::{Error, Result};

type Object = Arc<dyn Any + Send + Sync>;

static OBJECTS: Mutex<BTreeMap<u64, Object>> = Mutex::new(BTreeMap::new());
static NEXT: AtomicU64 = AtomicU64::new(1);

/// Tag in the high 16 bits of every id, so 0 and small integers are never
/// valid. Ids are `TAG | (counter << 1) | 1`, so every id is odd. Every double
/// in [2^62, 2^63), the range of the tagged ids, is a multiple of 1024 and so
/// even, which means an id that went through a double never matches a live
/// object. This holds while the counter stays below 2^47.
const TAG: u64 = 0x5043_0000_0000_0000;

fn objects() -> MutexGuard<'static, BTreeMap<u64, Object>> {
    // Only insert, clone and remove run under the lock, so the map is still
    // consistent after a panic poisoned it.
    OBJECTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn insert<T: Any + Send + Sync>(value: T) -> u64 {
    let id = TAG | (NEXT.fetch_add(1, Ordering::Relaxed) << 1) | 1;
    objects().insert(id, Arc::new(value));
    id
}

/// Returns a shared reference, so a long computation does not hold the lock.
pub fn get<T: Any + Send + Sync>(id: u64) -> Result<Arc<T>> {
    let obj = objects()
        .get(&id)
        .cloned()
        .ok_or_else(|| Error::BadHandle(format!("unknown handle 0x{id:016x}")))?;
    obj.downcast::<T>()
        .map_err(|_| Error::BadHandle(format!("handle 0x{id:016x} is not a {}", type_name::<T>())))
}

pub fn remove(id: u64) -> Result<()> {
    // The guard is released at the end of this statement, so the object is
    // dropped outside the lock.
    let obj = objects().remove(&id);
    match obj {
        Some(obj) => {
            drop(obj);
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
        let id = insert(42u32);
        assert_eq!(id & TAG, TAG);
        assert_eq!(*get::<u32>(id).unwrap(), 42);
        assert!(matches!(get::<f64>(id), Err(Error::BadHandle(_))));
        remove(id).unwrap();
        assert!(matches!(get::<u32>(id), Err(Error::BadHandle(_))));
        assert!(matches!(remove(id), Err(Error::BadHandle(_))));
        assert!(matches!(remove(0), Err(Error::BadHandle(_))));
    }
}
