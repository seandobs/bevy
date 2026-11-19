//! Starting the threads of the task pools and of the app itself.
//!
//! A native build starts a thread with [`std::thread`]. A target without it, such as the web,
//! where a thread is a web worker over shared memory, registers a [`ThreadSpawner`] with
//! [`set_thread_spawner`] before the first pool is built. [`spawn`] then starts every thread
//! through it, so the pools and the app place their work on threads the same way on every
//! target.

use alloc::{
    boxed::Box,
    string::{String, ToString},
};
use core::{any::Any, panic::AssertUnwindSafe};
use std::sync::{mpsc, Mutex, OnceLock};

/// How a thread ended: `Err` holds the payload of the panic that ended it.
type Ending = Result<(), Box<dyn Any + Send + 'static>>;

/// A function that runs `body` on a new thread named `name`, with a stack of `stack_size`
/// bytes when one is given.
pub type ThreadSpawner =
    fn(name: &str, stack_size: Option<usize>, body: Box<dyn FnOnce() + Send>) -> Result<(), String>;

struct Registered {
    spawner: ThreadSpawner,
    parallelism: usize,
}

static REGISTERED: OnceLock<Registered> = OnceLock::new();

/// Makes [`spawn`] start every thread with `spawner`, and makes
/// [`available_parallelism`](crate::available_parallelism) return `parallelism`.
///
/// Returns `false`, and changes nothing, if a spawner is already registered.
pub fn set_thread_spawner(spawner: ThreadSpawner, parallelism: usize) -> bool {
    REGISTERED
        .set(Registered {
            spawner,
            parallelism: parallelism.max(1),
        })
        .is_ok()
}

/// The core count the registered spawner gave, if one is registered.
pub(crate) fn registered_parallelism() -> Option<usize> {
    REGISTERED.get().map(|registered| registered.parallelism)
}

/// A thread that [`spawn`] started.
#[derive(Debug)]
pub struct ThreadHandle(Inner);

#[derive(Debug)]
enum Inner {
    Std(std::thread::JoinHandle<()>),
    // The mutex makes the handle `Sync`, as a pool that holds it must be.
    Spawned(Mutex<mpsc::Receiver<Ending>>),
}

impl ThreadHandle {
    /// Waits for the thread to finish. An `Err` holds the payload of the panic that ended it,
    /// or, for a thread the registered spawner started that ended without unwinding, a message
    /// saying so.
    pub fn join(self) -> Ending {
        match self.0 {
            Inner::Std(handle) => handle.join(),
            Inner::Spawned(finished) => finished
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .recv()
                .unwrap_or_else(|_| Err(Box::new("the thread ended without finishing"))),
        }
    }
}

/// Runs `body` on a new thread named `name`, with the registered spawner if there is one and
/// with [`std::thread`] otherwise. Returns why if the thread didn't start.
pub fn spawn(
    name: &str,
    stack_size: Option<usize>,
    body: impl FnOnce() + Send + 'static,
) -> Result<ThreadHandle, String> {
    if let Some(registered) = REGISTERED.get() {
        let (finished, on_finish) = mpsc::sync_channel(1);
        let body = move || {
            let ending = std::panic::catch_unwind(AssertUnwindSafe(body));
            // When the handle is gone, nothing waits for the thread, and a panic goes on
            // unwinding.
            if let Err(mpsc::SendError(Err(payload))) = finished.send(ending) {
                std::panic::resume_unwind(payload);
            }
        };
        (registered.spawner)(name, stack_size, Box::new(body))?;
        return Ok(ThreadHandle(Inner::Spawned(Mutex::new(on_finish))));
    }
    let mut builder = std::thread::Builder::new().name(name.into());
    if let Some(stack_size) = stack_size {
        builder = builder.stack_size(stack_size);
    }
    builder
        .spawn(body)
        .map(|handle| ThreadHandle(Inner::Std(handle)))
        .map_err(|error| error.to_string())
}
