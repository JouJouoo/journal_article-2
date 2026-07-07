from __future__ import annotations

import numpy as np

from tecsf.chain import REJECTED, REVERTED, SETTLED, ConsortiumChainLedger, SettlementEngine
from tecsf.config import ExperimentConfig, ScenarioConfig
from tecsf.data import generate_synthetic_scenario
from tecsf.market import ActionBatch, clear_market
from tecsf.utils import stable_hash


def _package():
    config = ExperimentConfig(scenario=ScenarioConfig(num_agents=3, num_nodes=3, horizon=4))
    scenario = generate_synthetic_scenario(config, seed=5)
    actions = ActionBatch(
        q_buy=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        q_sell=np.asarray([0.0, 0.0, 1.5], dtype=np.float32),
        price_buy=np.asarray([0.9, 0.8, 0.1], dtype=np.float32),
        price_sell=np.asarray([0.1, 0.1, 0.4], dtype=np.float32),
        charge=np.zeros(3, dtype=np.float32),
        discharge=np.zeros(3, dtype=np.float32),
    )
    return config, clear_market(
        scenario,
        config,
        0,
        np.full(3, config.storage.init_soc, dtype=np.float32),
        actions,
    )


def test_settlement_rejects_hash_mismatch():
    config, package = _package()
    package.package_hash = "bad"
    engine = SettlementEngine(config, num_agents=3)
    record = engine.settle(package)
    assert record.state == REJECTED


def test_settlement_mints_once_and_reverts_duplicate():
    config, package = _package()
    engine = SettlementEngine(config, num_agents=3)
    record = engine.settle(package)
    assert record.state == SETTLED
    assert sum(record.lccoins.values()) >= 0.0
    balances = engine.energy_balances.copy()
    duplicate = engine.settle(package)
    assert duplicate.state == REVERTED
    assert np.allclose(engine.energy_balances, balances)


def test_lccoins_candidate_uses_paper_contribution_formula():
    config, package = _package()
    expected_contribution = (
        config.lccoins.clean_energy_weight * package.carbon.q_lc
        + config.lccoins.carbon_reduction_weight * package.carbon.carbon_reduction
    )
    expected_mint = config.lccoins.minting_coefficient * expected_contribution
    assert np.allclose(package.carbon.low_carbon_contribution, expected_contribution)
    assert np.allclose(package.carbon.lccoins_candidate, expected_mint)


def test_lccoins_consensus_threshold_controls_minting():
    config, package = _package()
    config.lccoins.validator_count = 3
    config.lccoins.consensus_threshold = 4
    engine = SettlementEngine(config, num_agents=3)

    record = engine.settle(package)

    assert record.state == SETTLED
    assert record.consensus_threshold == 4
    assert not any(record.consensus_confirmed.values())
    assert sum(record.lccoins.values()) == 0.0
    assert np.allclose(engine.lccoins_balances, 0.0)


def test_lccoins_consensus_rejects_inconsistent_carbon_reduction():
    config, package = _package()
    agent = 0
    package.carbon.carbon_reduction[agent] += 1.0
    package.carbon.low_carbon_contribution[agent] = (
        config.lccoins.clean_energy_weight * package.carbon.q_lc[agent]
        + config.lccoins.carbon_reduction_weight
        * package.carbon.carbon_reduction[agent]
    )
    package.carbon.lccoins_candidate[agent] = (
        config.lccoins.minting_coefficient
        * package.carbon.low_carbon_contribution[agent]
    )
    package.package_hash = stable_hash(package.payload(include_hash=False))
    engine = SettlementEngine(config, num_agents=3)

    record = engine.settle(package)

    assert record.state == SETTLED
    assert record.consensus_confirmed[agent] is False
    assert agent not in record.lccoins
    assert engine.lccoins_balances[agent] == 0.0


def test_chain_produces_blocks_and_receipts(tmp_path):
    config, package = _package()
    chain = ConsortiumChainLedger(config, num_agents=3)

    record = chain.settle(package)

    assert record.state == SETTLED
    assert len(chain.blocks) == 1
    block = chain.blocks[0]
    assert block.height == 0
    assert block.prev_hash == chain.genesis_hash
    assert block.transactions[0].package_hash == package.package_hash
    assert block.receipts[0].record_id == record.record_id
    assert block.receipts[0].state == SETTLED
    assert block.block_hash
    assert block.state_root

    ledger_path = tmp_path / "ledger.json"
    chain.export_ledger(ledger_path)
    assert ledger_path.exists()


def test_chain_links_consecutive_blocks():
    config, package = _package()
    chain = ConsortiumChainLedger(config, num_agents=3)

    first = chain.settle(package)
    second = chain.settle(package)

    assert first.state == SETTLED
    assert second.state == REVERTED
    assert len(chain.blocks) == 2
    assert chain.blocks[1].prev_hash == chain.blocks[0].block_hash
    assert chain.blocks[1].receipts[0].state == REVERTED


def test_validator_confirmations_recorded_per_node():
    """验证每个验证节点都生成独立的确认标识 sigma^v。"""
    config, package = _package()
    config.lccoins.validator_count = 4
    engine = SettlementEngine(config, num_agents=3)
    assert len(engine.validators) == 4
    roles = {v.role.value for v in engine.validators}
    assert roles == {"market_clearing", "metering", "carbon_accounting", "ledger_maintenance"}

    record = engine.settle(package)
    assert record.state == SETTLED
    for agent in range(3):
        confirmations = record.validator_confirmations.get(agent, {})
        assert len(confirmations) == 4
        assert all(isinstance(v, bool) for v in confirmations.values())
        votes = sum(1 for v in confirmations.values() if v)
        assert record.consensus_votes[agent] == votes
