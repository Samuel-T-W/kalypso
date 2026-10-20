import json
from dataclasses import dataclass
from pathlib import Path



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






@dataclass
class WorkerSpec:
    """A worker we intend to use, before we know whether it is up."""
    name: str
    url: str
 
 
@dataclass
class PoolConfig:
    workers: list[WorkerSpec]
    policy: str | None = None  # None = use --policy or the default
 
 
def _normalize_url(url):
    # "localhost:8001/" -> "http://localhost:8001", so duplicates are caught.
    url = str(url).strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url
 
 
def _check_specs(specs):
    if not specs:
        raise ValueError("no workers configured")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate worker names: {names}")
    urls = [s.url for s in specs]
    if len(set(urls)) != len(urls):
        raise ValueError(f"duplicate worker urls: {urls}")
    return specs
 
 
def parse_urls(urls):
    """'http://a:8001,http://b:8002' (the --workers flag) -> [WorkerSpec]."""
    if isinstance(urls, str):
        urls = urls.split(",")
    urls = [u.strip() for u in urls if u.strip()]
    return _check_specs(
        [WorkerSpec(f"w{i}", _normalize_url(u)) for i, u in enumerate(urls)]
    )
 
 
def parse_config(data):
    """A dict loaded from YAML/JSON -> PoolConfig, with every entry checked."""
    if not isinstance(data, dict) or not isinstance(data.get("workers"), list):
        raise ValueError("config must have a 'workers' list")
    specs = []
    for i, entry in enumerate(data["workers"]):
        if isinstance(entry, str):          # allow "- localhost:8001" shorthand
            entry = {"url": entry}
        if not isinstance(entry, dict) or "url" not in entry:
            raise ValueError(f"worker #{i} needs a 'url'")
        specs.append(WorkerSpec(
            name=str(entry.get("name", f"w{i}")),
            url=_normalize_url(entry["url"]),
        ))
    return PoolConfig(workers=_check_specs(specs), policy=data.get("policy"))
 
 
def load_config(path):
    """Read a .yaml/.yml or .json pool config file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # imported here so the rest of pool.py works without pyyaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return parse_config(data)
 
