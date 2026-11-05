from vllm.kalypso.context import ExecutionState, SemContext, SemanticInput
from vllm.kalypso.controller.plan import SemanticPlan  # noqa: F401 (import before sem_ops to avoid a cycle)
from vllm.kalypso.sem_ops.joins import CartesianProduct


def test_cartesian_product_rows_keep_the_left_rows_index_and_worker():
    contract = SemContext(
        input=SemanticInput(data=[{"role": "user", "content": "contract 4"}]),
        state=ExecutionState(idx=4, worker="w2"),
    )
    hypotheses = [
        SemContext(input=SemanticInput(data=[{"role": "user", "content": "h1"}])),
        SemContext(input=SemanticInput(data=[{"role": "user", "content": "h2"}])),
    ]

    rows = CartesianProduct(hypotheses)(contract)

    assert [(r.state.idx, r.state.worker) for r in rows] == [(4, "w2"), (4, "w2")]
    assert [r.input.right_data for r in rows] == [
        [{"role": "user", "content": "h1"}],
        [{"role": "user", "content": "h2"}],
    ]
