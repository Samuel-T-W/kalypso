class Worker:
    def __init__(self, name, executor, manager, url=None, model = None):
        self.name = name
        self.executor = executor
        self.manager = manager
        self.url = url
        self.model = model

    def __repr__(self):
        return f"Worker({self.name})"


class WorkerPool:
    def __init__(self, workers, car):
        workers = list(workers)
        if not workers:
            raise ValueError("a pool needs at least one worker")
        names = [w.name for w in workers]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate worker names: {names}")
        
        urls = [w.url for w in workers if w.url is not None]
        if len(set(urls)) != len(urls):
            raise ValueError(f"duplicate worker urls: {urls}")

        if len({id(w.manager) for w in workers}) != len(workers):
            raise ValueError("every worker needs its own KVMemoryManager")
        self.workers = workers

    @classmethod
    def single(cls, executor, manager, name="w0"):
        """One worker wrapping the existing singleton manager and executor."""
        return cls([Worker(name, executor, manager)])

    @classmethod
    def simulated(cls, n, executor, manager_factory):
        """n logical workers sharing one real engine, each with its own budget.

        manager_factory() must return a fresh KVMemoryManager each call.
        """
        return cls([Worker(f"w{i}", executor, manager_factory()) for i in range(n)])

    def manager(self, w):
        return w.manager

    def managers(self):
        return [w.manager for w in self.workers]

    def outstanding(self, w):
        return w.manager.total_inflight()

    def total_capacity(self):
       return sum(m.capacity() for m in self.managers())