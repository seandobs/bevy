//! Running the engine on web workers, started with one call from the page.
//!
//! [`start`] takes the page's canvas and spawns the workers the engine runs on: the render
//! worker, which owns the canvas as an `OffscreenCanvas`, the wgpu device and the render world
//! ([`bevy_render::web_worker`]); and the engine worker, which runs the app
//! world and winit's event loop, with input forwarded from the page. Every worker runs the same
//! script, shipped with this module, which instantiates the wasm module over the page's shared
//! memory and then calls this module's entry for its role. The app supplies one export, the
//! engine worker's, which builds the app with
//! [`RenderCreation::WebWorker`](bevy_render::settings::RenderCreation::WebWorker) and runs it; `App::run` then has `finish` and `cleanup`
//! executed on the render worker before the app runs. Optionally the app supplies a render
//! worker setup export, called there before the render world is created: the place to patch
//! WebGPU before the device exists, or to create the wgpu resources itself and
//! [provide](bevy_render::web_worker::provide_render_resources) them. [`start_render`] and
//! [`start_with_render`] split [`start`] so the render worker (and that setup) can start ahead
//! of the engine.
//!
//! Functions the engine reaches on the page (JS hooks looked up on `self`) are declared in
//! `#[wasm_bindgen(js_namespace = self)]` extern blocks marked [`page_functions`]; each worker
//! gets a same-named function that relays the call to the page's global and resolves with the
//! result; a function returning a value must be `async` or return a `js_sys::Promise`, which the
//! attribute checks. Worker console output is mirrored to the page. Once a worker's last entry
//! has returned it posts `{ __bevy_ready: tag }` to the page, or
//! `{ __bevy_ready: tag, error }` if an entry failed. [`spawn_worker`] spawns further workers of
//! the app's own on the same terms.
//!
//! The engine worker registers a thread spawner with `bevy_tasks` before the app is built
//! ([`enable_threads`]), so the compute pool's threads, and every thread the app starts with
//! [`bevy_tasks::thread::spawn`], are workers of this module too. A thread's worker closes once
//! the thread's body returns, unless the body called [`keep_this_worker_open`].

use bevy_platform::prelude::{Box, String, ToString, Vec, vec};
extern crate std;

use std::collections::BTreeMap;
use core::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, OffscreenCanvas, Worker};

use bevy_render::settings::WgpuSettings;

pub use bevy_derive::page_functions;
pub use bevy_render::web_worker::{
    init_on_render_worker, provide_render_resources, render_worker_main,
};

#[wasm_bindgen(module = "/src/web_worker.js")]
extern "C" {
    #[wasm_bindgen(js_name = spawnWorker, catch)]
    fn spawn_worker_js(init: &Object) -> Result<Worker, JsValue>;
    #[wasm_bindgen(js_name = spawnThreadWorker, catch)]
    fn spawn_thread_worker_js(init: &Object) -> Result<(), JsValue>;
}

/// What every worker is spawned with.
pub struct WebWorkerConfig {
    /// The url of the wasm-bindgen glue module, for the workers to import.
    pub glue_url: String,
    /// The app's engine worker export.
    pub engine_entry: String,
    /// The engine export's argument.
    pub engine_args: JsValue,
    /// The app's render worker setup export, if any, called with the `OffscreenCanvas` before
    /// the render world is created there.
    pub render_setup: Option<String>,
    /// The render worker's wasm stack size in bytes; `None` takes the module's default.
    pub render_stack_size: Option<u32>,
    /// The engine worker's wasm stack size in bytes; `None` takes the module's default.
    pub engine_stack_size: Option<u32>,
}

/// A worker for [`spawn_worker`].
pub struct WorkerSpec {
    /// The app export the worker calls once its wasm instance is up.
    pub entry: String,
    /// The export's arguments.
    pub args: Vec<JsValue>,
    /// Names the worker in the page's console.
    pub tag: String,
    /// The worker's wasm stack size in bytes; `None` takes the module's default.
    pub stack_size: Option<u32>,
    /// Objects in `args` to transfer rather than clone.
    pub transfer: Vec<JsValue>,
}

/// The workers [`start`] spawned. The page may add its own listeners; the engine's relay
/// listeners are already installed.
pub struct WebWorkers {
    /// The render worker.
    pub render: Worker,
    /// The engine worker.
    pub engine: Worker,
}

/// The render worker [`start_render`] spawned, for [`start_with_render`].
pub struct RenderWorker {
    /// The worker.
    pub worker: Worker,
    worker_id: u32,
}

/// Page side: prepares `canvas` for the engine and spawns its workers.
pub fn start(canvas: HtmlCanvasElement, config: &WebWorkerConfig) -> Result<WebWorkers, JsValue> {
    let render = start_render(canvas, config)?;
    start_with_render(render, config)
}

/// Page side: prepares `canvas` for the engine and spawns the render worker alone, ahead of
/// [`start_with_render`]. Its setup export runs and its device is created as soon as it is
/// up; it then waits for the engine worker's render world, so the app can spawn it well before
/// it starts the engine (to warm a pipeline cache, say).
pub fn start_render(
    canvas: HtmlCanvasElement,
    config: &WebWorkerConfig,
) -> Result<RenderWorker, JsValue> {
    let (worker_id, offscreen) = bevy_winit::prepare_worker(canvas)?;
    let canvas_arg: JsValue = offscreen.clone().into();
    let mut render_calls = Vec::new();
    if let Some(setup) = &config.render_setup {
        render_calls.push((setup.as_str(), vec![canvas_arg.clone()]));
    }
    render_calls.push((RENDER_ENTRY, vec![canvas_arg]));
    let worker = spawn(
        &config.glue_url,
        &render_calls,
        "render",
        config.render_stack_size,
        &[offscreen.into()],
        false,
    )?;
    Ok(RenderWorker { worker, worker_id })
}

/// Page side: spawns the engine worker for a render worker from [`start_render`].
pub fn start_with_render(
    render: RenderWorker,
    config: &WebWorkerConfig,
) -> Result<WebWorkers, JsValue> {
    let RenderWorker {
        worker: render,
        worker_id,
    } = render;
    let engine = spawn(
        &config.glue_url,
        &[
            (ENABLE_THREADS_ENTRY, vec![config.glue_url.as_str().into()]),
            (ATTACH_ENTRY, vec![worker_id.into()]),
            (config.engine_entry.as_str(), vec![config.engine_args.clone()]),
        ],
        "engine",
        config.engine_stack_size,
        &[],
        false,
    )?;
    Ok(WebWorkers {
        render,
        engine,
    })
}

/// Page side: spawns one worker on the engine's terms (see the module docs).
pub fn spawn_worker(glue_url: &str, spec: &WorkerSpec) -> Result<Worker, JsValue> {
    spawn(
        glue_url,
        &[(spec.entry.as_str(), spec.args.clone())],
        &spec.tag,
        spec.stack_size,
        &spec.transfer,
        false,
    )
}

/// Spawns a worker that calls `calls` in order (each an export and its arguments) once its wasm
/// instance is up.
fn spawn(
    glue_url: &str,
    calls: &[(&str, Vec<JsValue>)],
    tag: &str,
    stack_size: Option<u32>,
    transfer: &[JsValue],
    close_when_done: bool,
) -> Result<Worker, JsValue> {
    let init = init_message(glue_url, calls, tag, stack_size, transfer, close_when_done)?;
    spawn_worker_js(&init)
}

/// The init message of a worker that calls `calls` in order.
fn init_message(
    glue_url: &str,
    calls: &[(&str, Vec<JsValue>)],
    tag: &str,
    stack_size: Option<u32>,
    transfer: &[JsValue],
    close_when_done: bool,
) -> Result<Object, JsValue> {
    let init = Object::new();
    let set = |key: &str, value: &JsValue| Reflect::set(&init, &key.into(), value).map(|_| ());
    set("glueUrl", &glue_url.into())?;
    let calls_array = Array::new();
    for (entry, args) in calls {
        let call = Object::new();
        Reflect::set(&call, &"entry".into(), &(*entry).into())?;
        Reflect::set(&call, &"args".into(), &args.iter().collect::<Array>())?;
        calls_array.push(&call);
    }
    set("calls", &calls_array)?;
    set("tag", &tag.into())?;
    set("module", &wasm_bindgen::module())?;
    set("memory", &wasm_bindgen::memory())?;
    if let Some(stack_size) = stack_size {
        set("stackSize", &stack_size.into())?;
    }
    set("transfer", &transfer.iter().collect::<Array>())?;
    set("closeWhenDone", &close_when_done.into())?;
    Ok(init)
}

const RENDER_ENTRY: &str = "__bevy_render_worker_main";
const ATTACH_ENTRY: &str = "__bevy_attach_engine_worker";
const ENABLE_THREADS_ENTRY: &str = "__bevy_enable_threads";
const THREAD_ENTRY: &str = "__bevy_thread_main";

/// The stack of a thread started without a stack size, the size of a native thread's.
const THREAD_STACK_BYTES: usize = 2 << 20;

/// The glue url [`enable_threads`] was given, in the shared memory so every worker can spawn.
static GLUE_URL: OnceLock<String> = OnceLock::new();

/// A thread's body waits here, under a number of its own, until its worker takes it.
static BODIES: Mutex<BTreeMap<u32, Box<dyn FnOnce() + Send>>> = Mutex::new(BTreeMap::new());
static NEXT_BODY: AtomicU32 = AtomicU32::new(0);

fn bodies() -> std::sync::MutexGuard<'static, BTreeMap<u32, Box<dyn FnOnce() + Send>>> {
    BODIES.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

std::thread_local! {
    static KEEP_OPEN: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Makes every thread `bevy_tasks` starts a worker of this module, sized to the browser's
/// core count. The engine worker calls it before the app is built; an app that starts its
/// workers some other way calls it before it builds its app.
pub fn enable_threads(glue_url: &str) {
    let _ = GLUE_URL.set(glue_url.into());
    let cores = Reflect::get(&js_sys::global(), &"navigator".into())
        .and_then(|navigator| Reflect::get(&navigator, &"hardwareConcurrency".into()))
        .ok()
        .and_then(|cores| cores.as_f64())
        .map_or(1, |cores| cores as usize);
    bevy_tasks::thread::set_thread_spawner(spawn_thread, cores);
}

/// Keeps the worker of the calling thread open after the thread's body returns, so the futures
/// the body spawned on the worker's event loop go on running. A thread that waits on JS, such
/// as one that holds a network socket, returns from its body and runs on its event loop
/// instead of blocking. Its [`ThreadHandle`](bevy_tasks::thread::ThreadHandle) counts it
/// finished once its body returns.
pub fn keep_this_worker_open() {
    KEEP_OPEN.with(|keep| keep.set(true));
}

/// The [`bevy_tasks::thread::ThreadSpawner`] of a web build: a worker that runs `body`, which
/// the page spawns.
fn spawn_thread(
    name: &str,
    stack_size: Option<usize>,
    body: Box<dyn FnOnce() + Send>,
) -> Result<(), String> {
    let glue_url = GLUE_URL.get().ok_or("threads are not enabled")?;
    let number = NEXT_BODY.fetch_add(1, Ordering::Relaxed);
    bodies().insert(number, body);
    let stack_size = stack_size.unwrap_or(THREAD_STACK_BYTES) as u32;
    let request = init_message(
        glue_url,
        &[(THREAD_ENTRY, vec![number.into()])],
        name,
        Some(stack_size),
        &[],
        true,
    )
    .and_then(|init| spawn_thread_worker_js(&init));
    match request {
        Ok(()) => Ok(()),
        Err(error) => {
            bodies().remove(&number);
            Err(error.as_string().unwrap_or_else(|| "the worker failed to start".to_string()))
        }
    }
}

// The worker entries, called by the worker script.

#[doc(hidden)]
#[wasm_bindgen(js_name = "__bevy_render_worker_main")]
pub async fn render_worker_entry(canvas: OffscreenCanvas) -> Result<(), JsValue> {
    render_worker_main(canvas, WgpuSettings::default()).await
}

#[doc(hidden)]
#[wasm_bindgen(js_name = "__bevy_enable_threads")]
pub fn enable_threads_entry(glue_url: String) {
    enable_threads(&glue_url);
}

#[doc(hidden)]
#[wasm_bindgen(js_name = "__bevy_thread_main")]
pub fn thread_entry(number: u32) -> bool {
    let body = bodies()
        .remove(&number)
        .expect("each body's worker takes it once");
    body();
    // The worker script closes the worker when the last entry returns anything but `false`.
    !KEEP_OPEN.with(core::cell::Cell::get)
}

#[doc(hidden)]
#[wasm_bindgen(js_name = "__bevy_attach_engine_worker")]
pub fn attach_engine_worker_entry(worker_id: u32) -> Result<(), JsValue> {
    attach_engine_worker(worker_id)
}

/// Engine worker side, before the app is built: attaches winit's event loop to the page's
/// canvas.
fn attach_engine_worker(worker_id: u32) -> Result<(), JsValue> {
    // The real canvas is on the render worker and the surface comes from there, so the window
    // handle winit needs is a placeholder.
    bevy_winit::attach_worker(worker_id, OffscreenCanvas::new(1, 1)?)
}

