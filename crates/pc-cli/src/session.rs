//! Session files (format version 1) and their headless replay.
//!
//! A session file holds a model's source text and every run that produced a result: the
//! exact problem JSON, the starting point, the final status JSON and the rows. `replay`
//! loads the model through the same C ABI table that MATLAB calls, reruns every run and
//! compares the new rows with the stored ones. The MATLAB reader and writer follow the
//! layout described in `docs/SPEC.md`, section 11.
//!
//! Every array of f64 is a block `{"shape":[r,c],"f64le":"<base64>"}`: the r-by-c matrix
//! as row-major little-endian binary64 in base64 (RFC 4648). The bytes keep NaN payloads,
//! infinities and -0, so a reader sees the bits the writer had. Row blocks have one row
//! per point, so their shape is `[count, stride]`. The problem and the status are stored
//! as the exact strings the engine took and wrote, because a JSON parse and print can
//! change digits.

use std::ffi::{CStr, CString, c_char};
use std::path::Path;
use std::ptr;
use std::time::{Duration, Instant};

use pckern::{PcApiV1, pc_api_v1};
use serde_json::{Map, Value, json};

use crate::base64;

/// The `format` key of a session file.
pub const FORMAT: &str = "phasecont-session";
/// The only version this build reads and writes.
pub const VERSION: u64 = 1;

/// A matrix of f64, stored row-major.
#[derive(Debug, Clone)]
pub struct Block {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Block {
    /// A `rows`-by-`cols` matrix. Panics if `data` has another length.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "block data length");
        Self { rows, cols, data }
    }

    /// A 1-by-n row.
    pub fn row(data: Vec<f64>) -> Self {
        Self::new(1, data.len(), data)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// True when the shapes agree and every value has the same bits.
    pub fn same_bits(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self
                .data
                .iter()
                .zip(&other.data)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    pub fn to_json(&self) -> Value {
        let mut bytes = Vec::with_capacity(self.data.len() * 8);
        for v in &self.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        json!({ "shape": [self.rows, self.cols], "f64le": base64::encode(&bytes) })
    }

    /// Reads a block. `what` names the key for the error text.
    pub fn from_json(v: &Value, what: &str) -> Result<Self, String> {
        let obj = v
            .as_object()
            .ok_or_else(|| format!("{what}: expected an array block"))?;
        let shape = obj
            .get("shape")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| format!("{what}: \"shape\" must be two counts, [rows,cols]"))?;
        let dim = |x: &Value| x.as_u64().and_then(|n| usize::try_from(n).ok());
        let (Some(rows), Some(cols)) = (dim(&shape[0]), dim(&shape[1])) else {
            return Err(format!("{what}: \"shape\" must hold two counts"));
        };
        let text = obj
            .get("f64le")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{what}: \"f64le\" must be a string"))?;
        let bytes = base64::decode(text).map_err(|e| format!("{what}: \"f64le\": {e}"))?;
        let want = rows.checked_mul(cols).and_then(|n| n.checked_mul(8));
        if want != Some(bytes.len()) {
            return Err(format!(
                "{what}: shape [{rows},{cols}] needs {} bytes, \"f64le\" has {}",
                want.map_or_else(|| "too many".to_owned(), |w| w.to_string()),
                bytes.len()
            ));
        }
        let data = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
        Ok(Self { rows, cols, data })
    }
}

/// What kind of run a record is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A branch or curve through `run_start`.
    Continuation,
    /// An initial value run through `run_start`, problem type `IVP`.
    Simulate,
    /// A Newton solve through `model_equilibrium`.
    Equilibrium,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Continuation => "continuation",
            Self::Simulate => "simulate",
            Self::Equilibrium => "equilibrium",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        [Self::Continuation, Self::Simulate, Self::Equilibrium]
            .into_iter()
            .find(|k| k.name() == s)
    }
}

/// The result of a Newton solve: the point and the eigenvalues as real and imaginary rows.
#[derive(Debug, Clone)]
pub struct EqResult {
    pub u: Block,
    pub eig_re: Block,
    pub eig_im: Block,
}

/// One run of a session.
#[derive(Debug, Clone)]
pub struct RunRec {
    pub kind: Kind,
    /// The problem JSON given to `run_start`, or the options JSON given to
    /// `model_equilibrium`.
    pub problem: String,
    /// The final status JSON of a run, as the engine wrote it. Empty for an equilibrium.
    pub status: String,
    /// "EP", "LP", "H", "LC", "LPC", "PD", "NS", "HOM" or "IVP"; empty for an equilibrium.
    pub ty: String,
    /// The free parameter names.
    pub free: Vec<String>,
    /// `[ntst, ncol]` of a run of cycles or a HOM curve, else empty.
    pub grid: Vec<usize>,
    /// Number of states.
    pub n: usize,
    /// The start buffer, or the Newton guess.
    pub u0: Block,
    pub p: Block,
    /// One block per branch, `[count, stride]`.
    pub rows: Vec<Block>,
    /// One block per branch, `[specials, stride]`.
    pub specials: Vec<Block>,
    pub result: Option<EqResult>,
}

impl RunRec {
    fn to_json(&self) -> Value {
        let mut o = Map::new();
        o.insert("kind".into(), json!(self.kind.name()));
        o.insert("problem".into(), json!(self.problem));
        o.insert("type".into(), json!(self.ty));
        o.insert("free".into(), json!(self.free));
        o.insert("grid".into(), json!(self.grid));
        o.insert("n".into(), json!(self.n));
        o.insert("u0".into(), self.u0.to_json());
        o.insert("p".into(), self.p.to_json());
        if self.kind == Kind::Equilibrium {
            if let Some(r) = &self.result {
                o.insert(
                    "result".into(),
                    json!({
                        "u": r.u.to_json(),
                        "eig_re": r.eig_re.to_json(),
                        "eig_im": r.eig_im.to_json(),
                    }),
                );
            }
        } else {
            o.insert("status".into(), json!(self.status));
            o.insert(
                "rows".into(),
                Value::Array(self.rows.iter().map(Block::to_json).collect()),
            );
            o.insert(
                "specials".into(),
                Value::Array(self.specials.iter().map(Block::to_json).collect()),
            );
        }
        Value::Object(o)
    }

    fn from_json(v: &Value, ctx: &str) -> Result<Self, String> {
        let o = v
            .as_object()
            .ok_or_else(|| format!("{ctx}: expected an object"))?;
        let kind_name = str_field(o, "kind", ctx)?;
        let kind = Kind::parse(&kind_name).ok_or_else(|| {
            format!("{ctx}.kind: \"{kind_name}\" is not continuation, simulate or equilibrium")
        })?;
        let free = field(o, "free", ctx)?
            .as_array()
            .and_then(|a| a.iter().map(|x| x.as_str().map(str::to_owned)).collect())
            .ok_or_else(|| format!("{ctx}.free: expected a list of strings"))?;
        let grid = field(o, "grid", ctx)?
            .as_array()
            .and_then(|a| {
                a.iter()
                    .map(|x| x.as_u64().and_then(|n| usize::try_from(n).ok()))
                    .collect()
            })
            .ok_or_else(|| format!("{ctx}.grid: expected a list of counts"))?;
        let mut rec = Self {
            kind,
            problem: str_field(o, "problem", ctx)?,
            status: String::new(),
            ty: str_field(o, "type", ctx)?,
            free,
            grid,
            n: count_field(o, "n", ctx)?,
            u0: block_field(o, "u0", ctx)?,
            p: block_field(o, "p", ctx)?,
            rows: Vec::new(),
            specials: Vec::new(),
            result: None,
        };
        if kind == Kind::Equilibrium {
            let rctx = format!("{ctx}.result");
            let r = field(o, "result", ctx)?
                .as_object()
                .ok_or_else(|| format!("{rctx}: expected an object"))?;
            rec.result = Some(EqResult {
                u: block_field(r, "u", &rctx)?,
                eig_re: block_field(r, "eig_re", &rctx)?,
                eig_im: block_field(r, "eig_im", &rctx)?,
            });
        } else {
            rec.status = str_field(o, "status", ctx)?;
            if !matches!(
                serde_json::from_str::<Value>(&rec.status),
                Ok(Value::Object(_))
            ) {
                return Err(format!("{ctx}.status: not a JSON object"));
            }
            rec.rows = block_list(o, "rows", ctx)?;
            rec.specials = block_list(o, "specials", ctx)?;
            if rec.rows.len() != rec.specials.len() {
                return Err(format!(
                    "{ctx}: {} row blocks but {} special blocks",
                    rec.rows.len(),
                    rec.specials.len()
                ));
            }
        }
        Ok(rec)
    }
}

/// The model of a session.
#[derive(Debug, Clone)]
pub struct ModelRec {
    /// A file name, never a path.
    pub name: String,
    /// The exact text passed to `model_from_ode`.
    pub source: String,
    /// SHA-256 of the source bytes, 64 lower-case hex digits.
    pub sha256: String,
    /// "jit", "fallback" or "aot".
    pub evaluator: String,
}

/// An object the user held: run and branch, both counted from 1. An equilibrium has
/// branch 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Item {
    pub run: usize,
    pub branch: usize,
}

/// A session file.
#[derive(Debug, Clone)]
pub struct Session {
    pub created: String,
    pub engine_name: String,
    pub engine_version: String,
    pub platform: String,
    pub matlab: String,
    pub model: ModelRec,
    pub runs: Vec<RunRec>,
    pub items: Vec<Item>,
}

fn field<'a>(o: &'a Map<String, Value>, key: &str, ctx: &str) -> Result<&'a Value, String> {
    o.get(key)
        .ok_or_else(|| format!("{ctx}: missing \"{key}\""))
}

fn str_field(o: &Map<String, Value>, key: &str, ctx: &str) -> Result<String, String> {
    field(o, key, ctx)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{ctx}.{key}: expected a string"))
}

fn count_field(o: &Map<String, Value>, key: &str, ctx: &str) -> Result<usize, String> {
    field(o, key, ctx)?
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("{ctx}.{key}: expected a count"))
}

fn block_field(o: &Map<String, Value>, key: &str, ctx: &str) -> Result<Block, String> {
    Block::from_json(field(o, key, ctx)?, &format!("{ctx}.{key}"))
}

fn block_list(o: &Map<String, Value>, key: &str, ctx: &str) -> Result<Vec<Block>, String> {
    field(o, key, ctx)?
        .as_array()
        .ok_or_else(|| format!("{ctx}.{key}: expected a list"))?
        .iter()
        .enumerate()
        .map(|(i, v)| Block::from_json(v, &format!("{ctx}.{key}[{i}]")))
        .collect()
}

impl Session {
    pub fn to_json(&self) -> Value {
        json!({
            "format": FORMAT,
            "version": VERSION,
            "created": self.created,
            "engine": { "name": self.engine_name, "version": self.engine_version },
            "platform": self.platform,
            "matlab": self.matlab,
            "precision": "f64",
            "model": {
                "name": self.model.name,
                "source": self.model.source,
                "sha256": self.model.sha256,
                "evaluator": self.model.evaluator,
            },
            "runs": self.runs.iter().map(RunRec::to_json).collect::<Vec<_>>(),
            "items": self
                .items
                .iter()
                .map(|i| json!({ "run": i.run, "branch": i.branch }))
                .collect::<Vec<_>>(),
        })
    }

    /// The file text: indented JSON and a final newline.
    pub fn to_text(&self) -> String {
        let mut s = serde_json::to_string_pretty(&self.to_json()).unwrap_or_default();
        s.push('\n');
        s
    }

    /// Reads a session. Every error names the key that is wrong.
    pub fn parse(text: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("not valid JSON: {e}"))?;
        let o = v
            .as_object()
            .ok_or_else(|| "a session file holds one JSON object".to_owned())?;
        if o.get("format").and_then(Value::as_str) != Some(FORMAT) {
            return Err(format!("\"format\" must be \"{FORMAT}\""));
        }
        let version = field(o, "version", "session")?
            .as_u64()
            .ok_or_else(|| "version: expected a count".to_owned())?;
        if version != VERSION {
            return Err(format!(
                "unsupported session version {version}; this build reads version {VERSION}"
            ));
        }
        if str_field(o, "precision", "session")? != "f64" {
            return Err("precision: only \"f64\" is supported".to_owned());
        }
        let obj = |key: &str| {
            field(o, key, "session")?
                .as_object()
                .ok_or_else(|| format!("{key}: expected an object"))
        };
        let engine = obj("engine")?;
        let model = obj("model")?;
        let runs = field(o, "runs", "session")?
            .as_array()
            .ok_or_else(|| "runs: expected a list".to_owned())?
            .iter()
            .enumerate()
            .map(|(i, r)| RunRec::from_json(r, &format!("runs[{i}]")))
            .collect::<Result<Vec<_>, _>>()?;
        let items = field(o, "items", "session")?
            .as_array()
            .ok_or_else(|| "items: expected a list".to_owned())?
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let ctx = format!("items[{i}]");
                let io = v
                    .as_object()
                    .ok_or_else(|| format!("{ctx}: expected an object"))?;
                let item = Item {
                    run: count_field(io, "run", &ctx)?,
                    branch: count_field(io, "branch", &ctx)?,
                };
                check_item(item, &runs).map_err(|e| format!("{ctx}: {e}"))?;
                Ok(item)
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            created: str_field(o, "created", "session")?,
            engine_name: str_field(engine, "name", "engine")?,
            engine_version: str_field(engine, "version", "engine")?,
            platform: str_field(o, "platform", "session")?,
            matlab: str_field(o, "matlab", "session")?,
            model: ModelRec {
                name: str_field(model, "name", "model")?,
                source: str_field(model, "source", "model")?,
                sha256: str_field(model, "sha256", "model")?,
                evaluator: str_field(model, "evaluator", "model")?,
            },
            runs,
            items,
        })
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::parse(&text)
    }
}

fn check_item(item: Item, runs: &[RunRec]) -> Result<(), String> {
    let Some(run) = item.run.checked_sub(1).and_then(|i| runs.get(i)) else {
        return Err(format!("run {} does not exist", item.run));
    };
    let ok = if run.kind == Kind::Equilibrium {
        item.branch == 0
    } else {
        (1..=run.rows.len()).contains(&item.branch)
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "branch {} does not exist in run {}",
            item.branch, item.run
        ))
    }
}

// ---------------------------------------------------------------------------------------
// The engine through the C ABI table.

/// Which evaluator compiles the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evaluator {
    Jit,
    Fallback,
}

impl Evaluator {
    pub fn name(self) -> &'static str {
        match self {
            Self::Jit => "jit",
            Self::Fallback => "fallback",
        }
    }
}

/// The function table of `pckern`, the library MATLAB loads.
pub struct Engine {
    api: &'static PcApiV1,
}

/// A model handle, freed on drop.
pub struct ModelHandle<'a> {
    engine: &'a Engine,
    id: u64,
    info: Value,
}

/// A run handle, freed on drop. Freeing a running run cancels it.
struct RunHandle<'a> {
    engine: &'a Engine,
    id: u64,
}

impl Drop for ModelHandle<'_> {
    fn drop(&mut self) {
        (self.engine.api.free_handle)(self.id);
    }
}

impl Drop for RunHandle<'_> {
    fn drop(&mut self) {
        (self.engine.api.free_handle)(self.id);
    }
}

/// What a finished run holds.
pub struct Executed {
    /// The final status JSON as the engine wrote it.
    pub status: String,
    pub rows: Vec<Block>,
    pub specials: Vec<Block>,
}

impl Engine {
    pub fn load() -> Result<Self, String> {
        let p = pc_api_v1();
        if p.is_null() {
            return Err("the engine did not start".to_owned());
        }
        Ok(Self {
            api: unsafe { &*p },
        })
    }

    /// The name and the version from `version()`, for example `("pckern", "0.1.0")`.
    pub fn version(&self) -> (String, String) {
        let p = (self.api.version)();
        let text = if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        };
        let mut words = text.split_whitespace();
        let name = words.next().unwrap_or("pckern").to_owned();
        let version = words.next().unwrap_or("unknown").to_owned();
        (name, version)
    }

    fn last_error(&self) -> String {
        let p = (self.api.last_error)();
        if p.is_null() {
            return "unknown engine error".to_owned();
        }
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    fn check(&self, rc: i32) -> Result<(), String> {
        if rc == 0 {
            Ok(())
        } else {
            Err(self.last_error())
        }
    }

    /// Copies an engine string and frees it.
    fn take_string(&self, p: *mut c_char) -> Option<String> {
        if p.is_null() {
            return None;
        }
        let text = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        unsafe { (self.api.free_string)(p) };
        Some(text)
    }

    /// Parses and compiles a model from its source text.
    pub fn model(&self, source: &str, evaluator: Evaluator) -> Result<ModelHandle<'_>, String> {
        let mut id = 0u64;
        let rc = unsafe {
            (self.api.model_from_ode)(
                source.as_ptr().cast(),
                source.len(),
                &mut id,
                ptr::null_mut(),
            )
        };
        self.check(rc)?;
        let mut handle = ModelHandle {
            engine: self,
            id,
            info: Value::Null,
        };
        let opts = CString::new(format!("{{\"evaluator\":\"{}\"}}", evaluator.name()))
            .map_err(|e| e.to_string())?;
        self.check(unsafe { (self.api.model_compile)(id, opts.as_ptr()) })?;
        let mut s: *mut c_char = ptr::null_mut();
        let rc = unsafe { (self.api.model_info)(id, &mut s) };
        let text = self.take_string(s);
        self.check(rc)?;
        let text = text.ok_or_else(|| "model_info returned no text".to_owned())?;
        handle.info = serde_json::from_str(&text).map_err(|e| format!("model_info: {e}"))?;
        Ok(handle)
    }

    fn poll(&self, run: u64) -> Result<String, String> {
        let mut s: *mut c_char = ptr::null_mut();
        let rc = unsafe { (self.api.run_poll)(run, &mut s) };
        let text = self.take_string(s);
        self.check(rc)?;
        text.ok_or_else(|| "run_poll returned no status".to_owned())
    }

    fn copy(
        &self,
        run: u64,
        branch: usize,
        what: u32,
        count: usize,
        stride: usize,
    ) -> Result<Block, String> {
        let mut buf = vec![f64::NAN; count * stride];
        let mut got = 0usize;
        let branch = u32::try_from(branch).map_err(|_| "too many branches".to_owned())?;
        let rc = unsafe {
            (self.api.run_copy)(
                run,
                branch,
                what,
                0,
                count,
                buf.as_mut_ptr(),
                buf.len(),
                &mut got,
            )
        };
        self.check(rc)?;
        if got != count {
            return Err(format!("run_copy gave {got} rows, expected {count}"));
        }
        Ok(Block::new(count, stride, buf))
    }
}

impl ModelHandle<'_> {
    /// The `model_info` JSON.
    pub fn info(&self) -> &Value {
        &self.info
    }

    /// SHA-256 of the source text as the engine computed it.
    pub fn sha256(&self) -> &str {
        self.info["source_sha256"].as_str().unwrap_or("")
    }

    /// Starts a run, waits for the worker and copies the rows of every branch.
    pub fn run(&self, problem: &str, u0: &[f64], p: &[f64]) -> Result<Executed, String> {
        let e = self.engine;
        let c = CString::new(problem).map_err(|_| "the problem JSON holds a NUL byte")?;
        let mut id = 0u64;
        let rc = unsafe {
            (e.api.run_start)(
                self.id,
                c.as_ptr(),
                u0.as_ptr(),
                u0.len(),
                p.as_ptr(),
                p.len(),
                &mut id,
            )
        };
        e.check(rc)?;
        let run = RunHandle { engine: e, id };
        let status = loop {
            let text = e.poll(run.id)?;
            let v: Value = serde_json::from_str(&text).map_err(|e| format!("status: {e}"))?;
            if v["state"] != "running" {
                break (text, v);
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        let (text, v) = status;
        let stride = v["stride"]
            .as_u64()
            .and_then(|s| usize::try_from(s).ok())
            .ok_or("status: missing stride")?;
        let branches = v["branches"].as_array().ok_or("status: missing branches")?;
        let mut rows = Vec::new();
        let mut specials = Vec::new();
        for (b, br) in branches.iter().enumerate() {
            let points = br["points"].as_u64().ok_or("status: missing points")?;
            let nsp = br["specials"].as_array().map_or(0, Vec::len);
            let points = usize::try_from(points).map_err(|_| "status: points too large")?;
            rows.push(e.copy(run.id, b, 0, points, stride)?);
            specials.push(e.copy(run.id, b, 1, nsp, stride)?);
        }
        Ok(Executed {
            status: text,
            rows,
            specials,
        })
    }

    /// Newton's method from `u0`. `opts` is the options JSON of `model_equilibrium`.
    pub fn equilibrium(&self, opts: &str, u0: &[f64], p: &[f64]) -> Result<EqResult, String> {
        let e = self.engine;
        let n = u0.len();
        let c = CString::new(opts).map_err(|_| "the options JSON holds a NUL byte")?;
        let mut u = vec![f64::NAN; n];
        let mut eig = vec![f64::NAN; 2 * n];
        let rc = unsafe {
            (e.api.model_equilibrium)(
                self.id,
                u0.as_ptr(),
                n,
                p.as_ptr(),
                p.len(),
                c.as_ptr(),
                u.as_mut_ptr(),
                n,
                eig.as_mut_ptr(),
                2 * n,
            )
        };
        e.check(rc)?;
        let im = eig.split_off(n);
        Ok(EqResult {
            u: Block::row(u),
            eig_re: Block::row(eig),
            eig_im: Block::row(im),
        })
    }
}

// ---------------------------------------------------------------------------------------
// Replay.

/// How a replay compares.
#[derive(Debug, Clone, Copy)]
pub struct ReplayOptions {
    /// Largest `|a - b| / max(1, |b|)` that still passes.
    pub tol: f64,
    /// Compile with the f64 fallback evaluator instead of the recorded one.
    pub fallback: bool,
}

/// The comparison of one run.
#[derive(Debug, Clone)]
pub struct RunReport {
    /// Position in the session, from 1.
    pub index: usize,
    pub kind: Kind,
    pub ty: String,
    pub pass: bool,
    /// Every value has the same bits.
    pub identical: bool,
    /// Largest `|a - b| / max(1, |b|)` over all values, infinite when a value is NaN in only
    /// one run or an infinity differs.
    pub max_rel_diff: f64,
    /// Block shapes agree, so the row counts do too.
    pub rows_match: bool,
    /// The special points have the same labels and indexes.
    pub labels_match: bool,
    /// The runs ended in the same state and for the same reason.
    pub stop_match: bool,
    pub seconds: f64,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub tol: f64,
    pub engine: String,
    pub evaluator: &'static str,
    pub runs: Vec<RunReport>,
}

impl Report {
    pub fn pass(&self) -> bool {
        self.runs.iter().all(|r| r.pass)
    }

    pub fn to_json(&self) -> Value {
        let num = |x: f64| if x.is_finite() { json!(x) } else { Value::Null };
        json!({
            "pass": self.pass(),
            "tol": self.tol,
            "engine": self.engine,
            "evaluator": self.evaluator,
            "runs": self.runs.iter().map(|r| json!({
                "index": r.index,
                "kind": r.kind.name(),
                "type": r.ty,
                "pass": r.pass,
                "identical": r.identical,
                "max_rel_diff": num(r.max_rel_diff),
                "rows_match": r.rows_match,
                "labels_match": r.labels_match,
                "stop_match": r.stop_match,
                "seconds": r.seconds,
                "message": r.message,
            })).collect::<Vec<_>>(),
        })
    }
}

/// Running comparison of stored blocks with replayed ones.
struct Acc {
    identical: bool,
    max_rel: f64,
    shapes: bool,
    notes: Vec<String>,
}

impl Acc {
    fn new() -> Self {
        Self {
            identical: true,
            max_rel: 0.0,
            shapes: true,
            notes: Vec::new(),
        }
    }

    /// Compares one block. `stored` is the reference, `new` the replayed one.
    fn add(&mut self, what: &str, stored: &Block, new: &Block) {
        if (stored.rows, stored.cols) != (new.rows, new.cols) {
            self.shapes = false;
            self.identical = false;
            self.notes.push(format!(
                "{what}: stored [{},{}], replayed [{},{}]",
                stored.rows, stored.cols, new.rows, new.cols
            ));
            return;
        }
        let mut reported = false;
        for (&b, &a) in stored.data.iter().zip(&new.data) {
            if a.to_bits() == b.to_bits() {
                continue;
            }
            self.identical = false;
            let d = if a.is_nan() && b.is_nan() {
                0.0
            } else if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
                if a == b { 0.0 } else { f64::INFINITY }
            } else {
                (a - b).abs() / b.abs().max(1.0)
            };
            if d.is_infinite() && !reported {
                reported = true;
                self.notes
                    .push(format!("{what}: a value is {b} stored and {a} replayed"));
            }
            self.max_rel = self.max_rel.max(d);
        }
    }
}

struct StatusCheck {
    structure: bool,
    labels: bool,
    stops: bool,
}

/// Compares what the status JSON of two runs says apart from the numbers.
fn compare_status(stored: &Value, new: &Value, notes: &mut Vec<String>) -> StatusCheck {
    let same = |key: &str| stored[key] == new[key];
    let empty = Vec::new();
    let sb = stored["branches"].as_array().unwrap_or(&empty);
    let nb = new["branches"].as_array().unwrap_or(&empty);
    let mut structure = same("kind") && same("n") && same("stride") && sb.len() == nb.len();
    let mut labels = true;
    let mut stops = same("state");
    if !same("state") {
        notes.push(format!(
            "state: \"{}\" stored, \"{}\" replayed",
            stored["state"].as_str().unwrap_or("?"),
            new["state"].as_str().unwrap_or("?")
        ));
    }
    for (i, (s, n)) in sb.iter().zip(nb).enumerate() {
        if s["points"] != n["points"] {
            structure = false;
            notes.push(format!(
                "branch {}: {} points stored, {} replayed",
                i + 1,
                s["points"],
                n["points"]
            ));
        }
        if s["stop"] != n["stop"] {
            stops = false;
            notes.push(format!(
                "branch {}: stop {} stored, {} replayed",
                i + 1,
                s["stop"],
                n["stop"]
            ));
        }
        let key = |v: &Value| -> Vec<(Value, Value)> {
            v["specials"]
                .as_array()
                .unwrap_or(&empty)
                .iter()
                .map(|x| (x["label"].clone(), x["index"].clone()))
                .collect()
        };
        if key(s) != key(n) {
            labels = false;
            notes.push(format!("branch {}: special points differ", i + 1));
        }
    }
    StatusCheck {
        structure,
        labels,
        stops,
    }
}

fn replay_run(model: &ModelHandle<'_>, rec: &RunRec, index: usize, tol: f64) -> RunReport {
    let t0 = Instant::now();
    let mut report = RunReport {
        index,
        kind: rec.kind,
        ty: rec.ty.clone(),
        pass: false,
        identical: false,
        max_rel_diff: f64::INFINITY,
        rows_match: false,
        labels_match: false,
        stop_match: false,
        seconds: 0.0,
        message: String::new(),
    };
    let mut acc = Acc::new();
    let mut notes = Vec::new();
    let (labels, stops, structure) = if let Some(stored) = &rec.result {
        match model.equilibrium(&rec.problem, rec.u0.data(), rec.p.data()) {
            Ok(new) => {
                acc.add("result.u", &stored.u, &new.u);
                acc.add("result.eig_re", &stored.eig_re, &new.eig_re);
                acc.add("result.eig_im", &stored.eig_im, &new.eig_im);
                (true, true, true)
            }
            Err(e) => {
                report.message = format!("replay failed: {e}");
                report.seconds = t0.elapsed().as_secs_f64();
                return report;
            }
        }
    } else {
        match model.run(&rec.problem, rec.u0.data(), rec.p.data()) {
            Ok(new) => {
                let stored_status: Value = serde_json::from_str(&rec.status).unwrap_or(Value::Null);
                let new_status: Value = serde_json::from_str(&new.status).unwrap_or(Value::Null);
                let c = compare_status(&stored_status, &new_status, &mut notes);
                if rec.rows.len() != new.rows.len() {
                    notes.push(format!(
                        "{} branches stored, {} replayed",
                        rec.rows.len(),
                        new.rows.len()
                    ));
                }
                for (b, (s, n)) in rec.rows.iter().zip(&new.rows).enumerate() {
                    acc.add(&format!("rows[{}]", b + 1), s, n);
                }
                for (b, (s, n)) in rec.specials.iter().zip(&new.specials).enumerate() {
                    acc.add(&format!("specials[{}]", b + 1), s, n);
                }
                (
                    c.labels,
                    c.stops,
                    c.structure && rec.rows.len() == new.rows.len(),
                )
            }
            Err(e) => {
                report.message = format!("replay failed: {e}");
                report.seconds = t0.elapsed().as_secs_f64();
                return report;
            }
        }
    };
    report.seconds = t0.elapsed().as_secs_f64();
    report.identical = acc.identical && structure;
    report.max_rel_diff = acc.max_rel;
    report.rows_match = acc.shapes && structure;
    report.labels_match = labels;
    report.stop_match = stops;
    let within = acc.max_rel <= tol;
    if !within && acc.max_rel.is_finite() {
        acc.notes.push(format!(
            "largest relative difference {:.3e} exceeds tol {tol:.3e}",
            acc.max_rel
        ));
    }
    notes.extend(acc.notes);
    report.message = notes.join("; ");
    report.pass = report.rows_match && labels && stops && within;
    report
}

/// Reruns every run of the session and compares. An error means the session cannot be
/// replayed at all: the model does not parse, or its source does not hash to the recorded
/// SHA-256.
pub fn replay(session: &Session, opts: ReplayOptions) -> Result<Report, String> {
    let engine = Engine::load()?;
    let evaluator = if opts.fallback || session.model.evaluator == "fallback" {
        Evaluator::Fallback
    } else {
        // An "aot" library path belongs to the machine that recorded it.
        Evaluator::Jit
    };
    let model = engine
        .model(&session.model.source, evaluator)
        .map_err(|e| format!("the model does not load: {e}"))?;
    if model.sha256() != session.model.sha256 {
        return Err(format!(
            "hash mismatch: the model source hashes to {} but the file records {}",
            model.sha256(),
            session.model.sha256
        ));
    }
    let (name, version) = engine.version();
    let runs = session
        .runs
        .iter()
        .enumerate()
        .map(|(i, rec)| replay_run(&model, rec, i + 1, opts.tol))
        .collect();
    Ok(Report {
        tol: opts.tol,
        engine: format!("{name} {version}"),
        evaluator: evaluator.name(),
        runs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECIAL: [f64; 10] = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,
        0.0,
        5e-324,
        // The largest subnormal, then the smallest normal.
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MAX,
        f64::MIN_POSITIVE,
        std::f64::consts::PI,
    ];

    fn sample_run(kind: Kind) -> RunRec {
        let eq = kind == Kind::Equilibrium;
        RunRec {
            kind,
            problem: if eq {
                r#"{"max_newton":50}"#.to_owned()
            } else {
                r#"{"type":"EP","free":"a"}"#.to_owned()
            },
            status: if eq {
                String::new()
            } else {
                r#"{"state":"done","branches":[]}"#.to_owned()
            },
            ty: if eq { String::new() } else { "EP".to_owned() },
            free: if eq { vec![] } else { vec!["a".to_owned()] },
            grid: vec![],
            n: 2,
            u0: Block::row(vec![0.5, -0.0]),
            p: Block::row(vec![1.0]),
            rows: if eq {
                vec![]
            } else {
                vec![Block::new(2, 5, SPECIAL.to_vec()), Block::new(0, 5, vec![])]
            },
            specials: if eq {
                vec![]
            } else {
                vec![Block::new(0, 5, vec![]), Block::new(0, 5, vec![])]
            },
            result: eq.then(|| EqResult {
                u: Block::row(vec![1.0, 2.0]),
                eig_re: Block::row(vec![-1.0, -2.0]),
                eig_im: Block::row(vec![0.0, 0.5]),
            }),
        }
    }

    fn sample() -> Session {
        Session {
            created: "2026-11-01T00:00:00Z".to_owned(),
            engine_name: "pckern".to_owned(),
            engine_version: "0.1.0".to_owned(),
            platform: "glnxa64".to_owned(),
            matlab: "2026b".to_owned(),
            model: ModelRec {
                name: "m.ode".to_owned(),
                source: "x'=-x\ndone\n".to_owned(),
                sha256: "0".repeat(64),
                evaluator: "jit".to_owned(),
            },
            runs: vec![
                sample_run(Kind::Continuation),
                sample_run(Kind::Equilibrium),
            ],
            items: vec![Item { run: 1, branch: 2 }, Item { run: 2, branch: 0 }],
        }
    }

    #[test]
    fn a_block_keeps_every_bit() {
        let b = Block::new(2, 5, SPECIAL.to_vec());
        let back = Block::from_json(&b.to_json(), "b").unwrap();
        assert!(back.same_bits(&b));
        assert_eq!((back.rows(), back.cols()), (2, 5));
    }

    #[test]
    fn block_bytes_are_row_major_little_endian() {
        let b = Block::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = b.to_json();
        let bytes = base64::decode(v["f64le"].as_str().unwrap()).unwrap();
        let first = f64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(first, 2.0, "the second value of the first row comes second");
        assert_eq!(v["shape"], json!([2, 3]));
    }

    #[test]
    fn empty_blocks_keep_their_shape() {
        for (r, c) in [(0, 4), (3, 0), (0, 0)] {
            let b = Block::new(r, c, vec![]);
            let v = b.to_json();
            assert_eq!(v["f64le"], "");
            let back = Block::from_json(&v, "b").unwrap();
            assert_eq!((back.rows(), back.cols()), (r, c));
        }
    }

    #[test]
    fn bad_blocks_are_errors_that_name_the_key() {
        let bad = [
            json!(3),
            json!({"shape":[1,2,3],"f64le":""}),
            json!({"shape":[-1,2],"f64le":""}),
            json!({"shape":[1.5,2],"f64le":""}),
            json!({"shape":[1,1]}),
            json!({"shape":[1,1],"f64le":7}),
            json!({"shape":[1,1],"f64le":"AAAA"}),
            json!({"shape":[1,1],"f64le":"AAAAAAAAAAA"}),
            json!({"shape":[usize::MAX,2],"f64le":""}),
        ];
        for v in bad {
            let e = Block::from_json(&v, "runs[0].u0").unwrap_err();
            assert!(e.starts_with("runs[0].u0"), "{e}");
        }
    }

    #[test]
    fn a_session_round_trips() {
        let s = sample();
        let back = Session::parse(&s.to_text()).unwrap();
        assert_eq!(back.created, s.created);
        assert_eq!(back.model.source, s.model.source);
        assert_eq!(back.items, s.items);
        assert_eq!(back.runs.len(), 2);
        let (a, b) = (&back.runs[0], &s.runs[0]);
        assert_eq!(a.kind, Kind::Continuation);
        assert_eq!(a.problem, b.problem);
        assert_eq!(a.status, b.status);
        assert_eq!(a.free, b.free);
        assert!(a.u0.same_bits(&b.u0));
        assert!(a.rows[0].same_bits(&b.rows[0]));
        assert_eq!(a.rows[1].rows(), 0);
        let (ra, rb) = (
            back.runs[1].result.as_ref().unwrap(),
            s.runs[1].result.as_ref().unwrap(),
        );
        assert!(ra.eig_im.same_bits(&rb.eig_im));
        // The text is stable: writing what was read gives the same bytes.
        assert_eq!(back.to_text(), s.to_text());
    }

    #[test]
    fn the_file_has_the_documented_keys() {
        let v = sample().to_json();
        assert_eq!(v["format"], "phasecont-session");
        assert_eq!(v["version"], 1);
        assert_eq!(v["precision"], "f64");
        for key in [
            "created", "engine", "platform", "matlab", "model", "runs", "items",
        ] {
            assert!(v.get(key).is_some(), "{key}");
        }
        let eq = &v["runs"][1];
        assert!(eq.get("result").is_some() && eq.get("rows").is_none());
        assert!(v["runs"][0].get("rows").is_some() && v["runs"][0].get("result").is_none());
    }

    #[test]
    fn reading_refuses_what_it_does_not_know() {
        let edit = |f: &dyn Fn(&mut Value)| {
            let mut v = sample().to_json();
            f(&mut v);
            Session::parse(&v.to_string()).unwrap_err()
        };
        assert!(edit(&|v| v["version"] = json!(2)).contains("version 2"));
        assert!(edit(&|v| v["format"] = json!("other")).contains("format"));
        assert!(edit(&|v| v["precision"] = json!("f32")).contains("precision"));
        assert!(
            edit(&|v| {
                v.as_object_mut().unwrap().remove("model");
            })
            .contains("model")
        );
        assert!(edit(&|v| v["runs"][0]["kind"] = json!("walk")).contains("runs[0].kind"));
        assert!(edit(&|v| v["runs"][0]["status"] = json!("[1]")).contains("runs[0].status"));
        assert!(edit(&|v| v["items"][0]["run"] = json!(9)).contains("items[0]"));
        assert!(edit(&|v| v["items"][0]["branch"] = json!(3)).contains("items[0]"));
        assert!(edit(&|v| v["items"][1]["branch"] = json!(1)).contains("items[1]"));
        assert!(Session::parse("[]").is_err());
        assert!(Session::parse("{").unwrap_err().contains("not valid JSON"));
    }

    #[test]
    fn comparison_uses_a_relative_difference_with_a_floor_of_one() {
        let stored = Block::row(vec![1000.0, 1e-12, f64::NAN, f64::INFINITY]);
        let new = Block::row(vec![
            1000.0 + 1e-6,
            5e-12,
            f64::from_bits(f64::NAN.to_bits() ^ 1),
            f64::INFINITY,
        ]);
        let mut acc = Acc::new();
        acc.add("x", &stored, &new);
        assert!(!acc.identical);
        // 1e-6 / 1000 = 1e-9 beats the 4e-12 of the small value, which has floor 1.
        assert!((acc.max_rel - 1e-9).abs() < 1e-12, "{}", acc.max_rel);
        assert!(acc.shapes && acc.notes.is_empty(), "{:?}", acc.notes);
    }

    #[test]
    fn comparison_rejects_shape_nan_and_infinity_differences() {
        let mut acc = Acc::new();
        acc.add("a", &Block::row(vec![1.0]), &Block::row(vec![1.0, 2.0]));
        assert!(!acc.shapes);

        let mut acc = Acc::new();
        acc.add("b", &Block::row(vec![1.0]), &Block::row(vec![f64::NAN]));
        assert!(acc.max_rel.is_infinite());

        let mut acc = Acc::new();
        acc.add(
            "c",
            &Block::row(vec![f64::INFINITY]),
            &Block::row(vec![f64::NEG_INFINITY]),
        );
        assert!(acc.max_rel.is_infinite());

        let mut acc = Acc::new();
        acc.add("d", &Block::row(vec![0.0]), &Block::row(vec![-0.0]));
        assert!(!acc.identical && acc.max_rel == 0.0);
    }

    #[test]
    fn status_comparison_names_the_difference() {
        let a: Value = serde_json::from_str(
            r#"{"state":"done","kind":"EP","n":2,"stride":5,"branches":[{"points":3,"stop":"Bounds",
            "specials":[{"label":"H","index":1}]}]}"#,
        )
        .unwrap();
        let mut b = a.clone();
        let mut notes = Vec::new();
        let c = compare_status(&a, &b, &mut notes);
        assert!(c.structure && c.labels && c.stops && notes.is_empty());

        b["branches"][0]["stop"] = json!("MaxSteps");
        b["branches"][0]["specials"][0]["label"] = json!("LP");
        b["branches"][0]["points"] = json!(4);
        let c = compare_status(&a, &b, &mut notes);
        assert!(!c.structure && !c.labels && !c.stops);
        assert_eq!(notes.len(), 3, "{notes:?}");
    }
}
