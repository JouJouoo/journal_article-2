from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from tecsf.config import ExperimentConfig
from tecsf.market import ClearingPackage
from tecsf.utils import stable_hash, to_jsonable


PENDING = "Pending"
VERIFIED = "Verified"
SETTLED = "Settled"
REJECTED = "Rejected"
REVERTED = "Reverted"


@dataclass
class SettlementRecord:
    record_id: str
    epoch: int
    package_hash: str
    state: str
    reason: str = ""
    energy_payments: list[dict] = field(default_factory=list)
    carbon_entries: list[dict] = field(default_factory=list)
    lccoins: dict[int, float] = field(default_factory=dict)
    consensus_confirmed: dict[int, bool] = field(default_factory=dict)
    consensus_votes: dict[int, int] = field(default_factory=dict)
    consensus_threshold: int = 0

    @property
    def settled(self) -> bool:
        return self.state == SETTLED


@dataclass
class BlockchainTransaction:
    """Settlement transaction submitted to the simulated TECS-Chain ledger."""

    tx_id: str
    epoch: int
    package_hash: str
    payload_hash: str
    tx_type: str
    sender: str
    enable_lccoins: bool
    submitted_at: float


@dataclass
class BlockchainReceipt:
    """Execution receipt emitted after a settlement transaction is processed."""

    tx_id: str
    record_id: str
    epoch: int
    state: str
    reason: str
    record_hash: str
    lccoins_total: float
    energy_payment_count: int
    carbon_entry_count: int
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Block:
    """Minimal block structure for local TECS-Chain simulation."""

    height: int
    timestamp: float
    prev_hash: str
    proposer: str
    transactions: list[BlockchainTransaction]
    receipts: list[BlockchainReceipt]
    merkle_root: str
    state_root: str
    block_hash: str


class SettlementEngine:
    """Local TECS-Chain simulator with atomic settlement semantics."""

    def __init__(self, config: ExperimentConfig, num_agents: int):
        self.config = config
        self.num_agents = num_agents
        self.energy_balances = np.full(num_agents, 1_000.0, dtype=np.float64)
        self.carbon_balances = np.zeros(num_agents, dtype=np.float64)
        self.lccoins_balances = np.zeros(num_agents, dtype=np.float64)
        self.minted_keys: set[tuple[str, int, int]] = set()
        self.records: list[SettlementRecord] = []

    def settle(
        self,
        package: ClearingPackage,
        enable_lccoins: bool = True,
        force_execution_failure: bool = False,
    ) -> SettlementRecord:
        record_id = f"epoch-{package.epoch}-{package.package_hash[:12]}"
        record = SettlementRecord(
            record_id=record_id,
            epoch=package.epoch,
            package_hash=package.package_hash,
            state=PENDING,
        )

        expected_hash = stable_hash(package.payload(include_hash=False))
        if expected_hash != package.package_hash:
            record.state = REJECTED
            record.reason = "clearing package hash mismatch"
            self.records.append(record)
            return record
        if package.max_violation() > self.config.clearing.network_tolerance:
            record.state = REJECTED
            record.reason = "constraint violation exceeds tolerance"
            self.records.append(record)
            return record

        record.state = VERIFIED
        energy_snapshot = self.energy_balances.copy()
        carbon_snapshot = self.carbon_balances.copy()
        lccoins_snapshot = self.lccoins_balances.copy()
        minted_snapshot = set(self.minted_keys)

        try:
            if force_execution_failure:
                raise RuntimeError("forced execution failure")
            self._apply_energy_payments(package, record)
            self._apply_carbon_market(package, record)
            if enable_lccoins:
                self._mint_lccoins(package, record)
            record.state = SETTLED
        except Exception as exc:  # pragma: no cover - exercised by state assertions.
            self.energy_balances = energy_snapshot
            self.carbon_balances = carbon_snapshot
            self.lccoins_balances = lccoins_snapshot
            self.minted_keys = minted_snapshot
            record.state = REVERTED
            record.reason = str(exc)

        self.records.append(record)
        return record

    def bypass_settlement(
        self, package: ClearingPackage, enable_lccoins: bool = True
    ) -> SettlementRecord:
        record = SettlementRecord(
            record_id=f"bypass-{package.epoch}-{package.package_hash[:12]}",
            epoch=package.epoch,
            package_hash=package.package_hash,
            state=SETTLED,
            reason="TECS-Chain bypassed",
        )
        self._apply_energy_payments(package, record)
        self._apply_carbon_market(package, record)
        if enable_lccoins:
            self._mint_lccoins(package, record)
        self.records.append(record)
        return record

    def _apply_energy_payments(
        self, package: ClearingPackage, record: SettlementRecord
    ) -> None:
        dt = self.config.scenario.delta_t
        for seller in range(self.num_agents):
            for buyer in range(self.num_agents):
                qty = float(package.p2p_power[seller, buyer])
                if qty <= 1e-10:
                    continue
                amount = qty * float(package.p2p_price[seller, buyer]) * dt
                if self.energy_balances[buyer] + 1e-9 < amount:
                    raise RuntimeError(f"insufficient energy balance for agent {buyer}")
                self.energy_balances[buyer] -= amount
                self.energy_balances[seller] += amount
                record.energy_payments.append(
                    {
                        "seller": seller,
                        "buyer": buyer,
                        "quantity": qty,
                        "amount": amount,
                    }
                )

    def _apply_carbon_market(
        self, package: ClearingPackage, record: SettlementRecord
    ) -> None:
        idx = package.epoch % self.config.scenario.horizon
        buy_price = self.config.market.carbon_allowance_price
        sell_price = self.config.market.low_carbon_sell_price
        for agent in range(self.num_agents):
            buy_qty = float(package.carbon.a_buy[agent])
            sell_qty = float(package.carbon.c_sell[agent])
            net_cost = buy_price * buy_qty - sell_price * sell_qty
            self.energy_balances[agent] -= net_cost
            self.carbon_balances[agent] += buy_qty - sell_qty
            record.carbon_entries.append(
                {
                    "agent": agent,
                    "epoch": idx,
                    "allowance_buy": buy_qty,
                    "low_carbon_sell": sell_qty,
                    "net_cost": net_cost,
                }
            )

    def _mint_lccoins(
        self, package: ClearingPackage, record: SettlementRecord
    ) -> None:
        threshold = self._consensus_threshold()
        record.consensus_threshold = threshold
        for agent in range(self.num_agents):
            key = (record.record_id, agent, package.epoch)
            if key in self.minted_keys:
                raise RuntimeError(f"duplicate Lccoins mint for {key}")
            amount = float(package.carbon.lccoins_candidate[agent])
            votes = self._lccoins_consensus_votes(package, agent, amount)
            confirmed = votes >= threshold
            record.consensus_votes[agent] = votes
            record.consensus_confirmed[agent] = confirmed
            if not confirmed:
                continue
            self.minted_keys.add(key)
            self.lccoins_balances[agent] += amount
            record.lccoins[agent] = amount

    def _consensus_threshold(self) -> int:
        validators = max(int(self.config.lccoins.validator_count), 1)
        configured = int(self.config.lccoins.consensus_threshold)
        if configured > 0:
            return configured
        return int(math.ceil(2.0 * validators / 3.0))

    def _lccoins_consensus_votes(
        self,
        package: ClearingPackage,
        agent: int,
        candidate_amount: float,
    ) -> int:
        validators = max(int(self.config.lccoins.validator_count), 1)
        if not self._lccoins_record_is_valid(package, agent, candidate_amount):
            return 0
        return validators

    def _lccoins_record_is_valid(
        self,
        package: ClearingPackage,
        agent: int,
        candidate_amount: float,
    ) -> bool:
        tolerance = max(float(self.config.clearing.network_tolerance), 1e-6)
        if not np.isfinite(candidate_amount) or candidate_amount < -tolerance:
            return False

        if not np.all(np.isfinite(package.p2p_power)):
            return False
        if np.any(package.p2p_power < -tolerance):
            return False
        if np.maximum(np.diag(package.p2p_power), 0.0).sum() > tolerance:
            return False

        p2p_sold_by_agent = package.p2p_power.sum(axis=1)
        p2p_sold = float(p2p_sold_by_agent.sum())
        p2p_bought = float(package.p2p_power.sum(axis=0).sum())
        if abs(p2p_sold - p2p_bought) > tolerance:
            return False

        q_lc = float(package.carbon.q_lc[agent])
        carbon_reduction = float(package.carbon.carbon_reduction[agent])
        if q_lc < -tolerance or carbon_reduction < -tolerance:
            return False

        dt = float(self.config.scenario.delta_t)
        expected_q_lc = (
            min(
                float(package.effective_pv[agent]),
                float(package.effective_load[agent])
                + float(package.charge[agent])
                + float(p2p_sold_by_agent[agent]),
            )
            * dt
        )
        if abs(q_lc - expected_q_lc) > max(tolerance, 1e-5):
            return False

        gamma_grid = float(package.carbon.grid_emission_factor)
        if not np.isfinite(gamma_grid) or gamma_grid < -tolerance:
            return False
        expected_grid_emission = gamma_grid * float(package.grid_buy[agent]) * dt
        if abs(float(package.carbon.e_grid[agent]) - expected_grid_emission) > max(
            tolerance, 1e-5
        ):
            return False

        expected_baseline = gamma_grid * (
            float(package.effective_load[agent]) + float(package.charge[agent])
        ) * dt
        if abs(float(package.carbon.baseline_emission[agent]) - expected_baseline) > max(
            tolerance, 1e-5
        ):
            return False

        expected_carbon_reduction = max(0.0, expected_baseline - expected_grid_emission)
        if abs(carbon_reduction - expected_carbon_reduction) > max(tolerance, 1e-5):
            return False

        contribution = (
            float(self.config.lccoins.clean_energy_weight) * q_lc
            + float(self.config.lccoins.carbon_reduction_weight) * carbon_reduction
        )
        if abs(
            float(package.carbon.low_carbon_contribution[agent])
            - max(contribution, 0.0)
        ) > max(tolerance, 1e-5):
            return False
        expected = float(self.config.lccoins.minting_coefficient) * max(
            contribution, 0.0
        )
        return abs(candidate_amount - expected) <= max(tolerance, 1e-5)


class SimulatedTECSChain:
    """In-memory blockchain ledger wrapped around the settlement engine.

    The class simulates transaction submission, single-proposer block production,
    receipts, block hashes, and state roots while reusing SettlementEngine as the
    deterministic contract executor.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        num_agents: int,
        chain_id: str = "tecs-chain-local",
        proposer: str = "validator-0",
    ):
        self.config = config
        self.num_agents = num_agents
        self.chain_id = chain_id
        self.proposer = proposer
        self.engine = SettlementEngine(config, num_agents)
        self.genesis_hash = stable_hash(
            {
                "chain_id": chain_id,
                "num_agents": num_agents,
                "scenario_seed": config.scenario.seed,
                "genesis": True,
            }
        )
        self.blocks: list[Block] = []
        self.mempool: list[tuple[BlockchainTransaction, ClearingPackage, bool]] = []

    @property
    def records(self) -> list[SettlementRecord]:
        return self.engine.records

    @property
    def energy_balances(self) -> np.ndarray:
        return self.engine.energy_balances

    @property
    def carbon_balances(self) -> np.ndarray:
        return self.engine.carbon_balances

    @property
    def lccoins_balances(self) -> np.ndarray:
        return self.engine.lccoins_balances

    def settle(
        self,
        package: ClearingPackage,
        enable_lccoins: bool = True,
        force_execution_failure: bool = False,
    ) -> SettlementRecord:
        tx = self.submit_settlement_transaction(package, enable_lccoins=enable_lccoins)
        block = self.produce_block(force_execution_failure=force_execution_failure)
        if not block.receipts:
            raise RuntimeError(f"no receipt produced for settlement transaction {tx.tx_id}")
        return self.records[-1]

    def bypass_settlement(
        self, package: ClearingPackage, enable_lccoins: bool = True
    ) -> SettlementRecord:
        return self.engine.bypass_settlement(package, enable_lccoins=enable_lccoins)

    def submit_settlement_transaction(
        self,
        package: ClearingPackage,
        enable_lccoins: bool = True,
        sender: str = "market-operator",
    ) -> BlockchainTransaction:
        tx_payload = {
            "chain_id": self.chain_id,
            "epoch": package.epoch,
            "package_hash": package.package_hash,
            "payload_hash": stable_hash(package.payload(include_hash=False)),
            "tx_type": "settlement",
            "sender": sender,
            "enable_lccoins": enable_lccoins,
            "nonce": len(self.blocks) + len(self.mempool),
        }
        tx = BlockchainTransaction(
            tx_id=stable_hash(tx_payload),
            epoch=package.epoch,
            package_hash=package.package_hash,
            payload_hash=tx_payload["payload_hash"],
            tx_type="settlement",
            sender=sender,
            enable_lccoins=enable_lccoins,
            submitted_at=time(),
        )
        self.mempool.append((tx, package, enable_lccoins))
        return tx

    def produce_block(
        self,
        max_transactions: int | None = None,
        force_execution_failure: bool = False,
    ) -> Block:
        if not self.mempool:
            return self._empty_block()

        tx_count = len(self.mempool) if max_transactions is None else max_transactions
        selected = self.mempool[:tx_count]
        self.mempool = self.mempool[tx_count:]

        transactions: list[BlockchainTransaction] = []
        receipts: list[BlockchainReceipt] = []
        for idx, (tx, package, enable_lccoins) in enumerate(selected):
            record = self.engine.settle(
                package,
                enable_lccoins=enable_lccoins,
                force_execution_failure=force_execution_failure and idx == 0,
            )
            transactions.append(tx)
            receipts.append(self._receipt_from_record(tx, record))

        block = self._make_block(transactions, receipts)
        self.blocks.append(block)
        return block

    def _empty_block(self) -> Block:
        block = self._make_block([], [])
        self.blocks.append(block)
        return block

    def _make_block(
        self,
        transactions: list[BlockchainTransaction],
        receipts: list[BlockchainReceipt],
    ) -> Block:
        prev_hash = self.blocks[-1].block_hash if self.blocks else self.genesis_hash
        merkle_root = stable_hash(
            {
                "transactions": [tx.tx_id for tx in transactions],
                "receipts": [receipt.record_hash for receipt in receipts],
            }
        )
        state_root = self._state_root()
        body = {
            "height": len(self.blocks),
            "prev_hash": prev_hash,
            "proposer": self.proposer,
            "merkle_root": merkle_root,
            "state_root": state_root,
            "transactions": transactions,
            "receipts": receipts,
        }
        block_hash = stable_hash(body)
        return Block(
            height=len(self.blocks),
            timestamp=time(),
            prev_hash=prev_hash,
            proposer=self.proposer,
            transactions=transactions,
            receipts=receipts,
            merkle_root=merkle_root,
            state_root=state_root,
            block_hash=block_hash,
        )

    def _receipt_from_record(
        self, tx: BlockchainTransaction, record: SettlementRecord
    ) -> BlockchainReceipt:
        events = [
            {
                "event": "SettlementState",
                "record_id": record.record_id,
                "state": record.state,
                "reason": record.reason,
            }
        ]
        if record.settled:
            events.extend(
                [
                    {
                        "event": "EnergyPaymentsApplied",
                        "record_id": record.record_id,
                        "count": len(record.energy_payments),
                    },
                    {
                        "event": "CarbonEntriesApplied",
                        "record_id": record.record_id,
                        "count": len(record.carbon_entries),
                    },
                    {
                        "event": "LccoinsMinted",
                        "record_id": record.record_id,
                        "total": float(sum(record.lccoins.values())),
                    },
                ]
            )
        return BlockchainReceipt(
            tx_id=tx.tx_id,
            record_id=record.record_id,
            epoch=record.epoch,
            state=record.state,
            reason=record.reason,
            record_hash=stable_hash(record),
            lccoins_total=float(sum(record.lccoins.values())),
            energy_payment_count=len(record.energy_payments),
            carbon_entry_count=len(record.carbon_entries),
            events=events,
        )

    def _state_root(self) -> str:
        return stable_hash(
            {
                "energy_balances": self.engine.energy_balances,
                "carbon_balances": self.engine.carbon_balances,
                "lccoins_balances": self.engine.lccoins_balances,
                "minted_keys": sorted(list(self.engine.minted_keys)),
                "record_hashes": [stable_hash(record) for record in self.engine.records],
            }
        )

    def ledger_payload(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "chain_id": self.chain_id,
                "genesis_hash": self.genesis_hash,
                "head_hash": self.blocks[-1].block_hash if self.blocks else self.genesis_hash,
                "height": len(self.blocks) - 1,
                "mempool_size": len(self.mempool),
                "blocks": self.blocks,
                "records": self.engine.records,
                "state": {
                    "energy_balances": self.engine.energy_balances,
                    "carbon_balances": self.engine.carbon_balances,
                    "lccoins_balances": self.engine.lccoins_balances,
                },
            }
        )

    def export_ledger(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.ledger_payload(), handle, ensure_ascii=True, indent=2)
