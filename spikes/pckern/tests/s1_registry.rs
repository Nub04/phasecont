//! Handle ids that went through a double must never name a live object.
//! MATLAB turns a uint64 into a double when it is mixed with doubles, and every
//! double in [2^62, 2^63) is a multiple of 1024, so the registry hands out odd
//! ids only.

use pckern::ffi::PC_ERR_BAD_HANDLE;
use pckern::registry;

#[test]
fn ids_rounded_through_a_double_are_unknown() {
    // 2048 ids span more than one 1024-wide step between doubles, so with
    // consecutive ids some rounded id would land on a live one.
    let ids: Vec<u64> = (0..2048u32).map(registry::insert).collect();
    for &id in &ids {
        let rounded = (id as f64) as u64;
        match registry::get::<u32>(rounded) {
            Err(e) => assert_eq!(e.code(), PC_ERR_BAD_HANDLE, "id 0x{id:016x}: {e}"),
            Ok(v) => {
                panic!("id 0x{id:016x} through a double is 0x{rounded:016x}, the live object {v}")
            }
        }
    }
    for &id in &ids {
        assert_eq!(id & 1, 1, "id 0x{id:016x} is even");
        assert_eq!(id >> 48, 0x5043, "id 0x{id:016x} has the wrong tag");
    }
    for (k, &id) in ids.iter().enumerate() {
        assert_eq!(*registry::get::<u32>(id).unwrap(), k as u32);
        registry::remove(id).unwrap();
    }
}
